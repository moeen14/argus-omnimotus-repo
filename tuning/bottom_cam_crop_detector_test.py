import os
import sys
import threading
import time

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

try:
    import serial
except ImportError:
    print("pyserial not found. Install it or activate nanosam-env.")
    sys.exit(1)


ENGINE_PATH = os.environ.get("ASABE_CROP_ENGINE", os.path.join(os.path.dirname(__file__), "crop_detector.engine"))
BOTTOM_CAM_DEVICE = "/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_9342D410-video-index0"
IMG_SIZE    = 640
CONF_THRESH = 0.75
IOU_THRESH  = 0.75

CLASS_NAMES  = {0: "gCrop", 1: "platform", 2: "ycrop", 3: "servo"}
COLOR_MAP    = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255), 3: (255, 0, 255)}

ESP2_PORT = os.environ.get("ASABE_ESP2_PORT", "/dev/esp32_02")
BAUD = 921600

SERVO_6_INITIAL      = 50
SERVO_6_START_ANGLE  = 305
SERVO_6_MIN_ANGLE    = 290
TARGET_GAP_PX  = 100
SEEK_STEP_DEG  = 0.5
SEEK_LOOP_SLEEP_S = 0.15

_SERIAL_LOCK = threading.Lock()

cuda.init()
_DEV = cuda.Device(0)
_CTX = _DEV.retain_primary_context()
_CTX.push()


class TRTDetector:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Could not load engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        self.input_shape  = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        print(f"[TRT] input  {self.input_name}: {self.input_shape}")
        print(f"[TRT] output {self.output_name}: {self.output_shape}")

        self.d_input  = cuda.mem_alloc(int(np.prod(self.input_shape))  * 4)
        self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape)) * 4)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)
        self.context.set_tensor_address(self.input_name,  int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

    def infer(self, inp):
        inp = np.ascontiguousarray(inp, dtype=np.float32)
        cuda.memcpy_htod_async(self.d_input, inp, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return np.array(self.h_output)


def letterbox(frame, size=IMG_SIZE):
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    padded[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return padded, r, (left, top)


def preprocess(frame):
    padded, r, pad = letterbox(frame)
    img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None], r, pad


def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order, keep = scores.argsort()[::-1], []
    while order.size > 0:
        i = order[0]; keep.append(i)
        inter_w = np.maximum(0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]))
        inter_h = np.maximum(0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]))
        iou = inter_w * inter_h / (areas[i] + areas[order[1:]] - inter_w * inter_h + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def postprocess(output, r, pad, orig_shape):
    pred = output[0].transpose(1, 0)
    boxes_xywh  = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids   = np.argmax(class_scores, axis=1)
    confs       = class_scores[np.arange(len(class_scores)), class_ids]

    mask = confs > CONF_THRESH
    if not np.any(mask):
        return []
    boxes_xywh, class_ids, confs = boxes_xywh[mask], class_ids[mask], confs[mask]

    cx, cy, w, h = boxes_xywh.T
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    left, top = pad
    boxes_xyxy[:, [0, 2]] -= left
    boxes_xyxy[:, [1, 3]] -= top
    boxes_xyxy /= r
    H, W = orig_shape[:2]
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, W - 1)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, H - 1)

    results = []
    for cls_id in np.unique(class_ids):
        m = class_ids == cls_id
        keep = nms(boxes_xyxy[m], confs[m], IOU_THRESH)
        for k in keep:
            x1, y1, x2, y2 = boxes_xyxy[m][k]
            results.append((int(cls_id), float(confs[m][k]), float(x1), float(y1), float(x2), float(y2)))
    return results


PLATFORM_CLASS_ID = 1
SERVO_CLASS_ID    = 3
GAP_LINE_COLOR    = (0, 165, 255)


def draw(frame, detections, fps):
    h, w = frame.shape[:2]
    for cls_id, conf, x1, y1, x2, y2 in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLOR_MAP.get(cls_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls_id, '?')} {conf:.2f}"
        if cls_id == PLATFORM_CLASS_ID:
            label += f" area={(x2 - x1) * (y2 - y1)}px"
        lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lsz[1] - 4), (x1 + lsz[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    counts = {cid: sum(1 for d in detections if d[0] == cid) for cid in CLASS_NAMES}
    count_text = "  ".join(f"{CLASS_NAMES[c]}: {counts[c]}" for c in sorted(CLASS_NAMES))
    tsz, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, count_text, (w - tsz[0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    draw_platform_servo_gap(frame, detections)


def platform_servo_edges(detections):

    platform = next((d for d in detections if d[0] == PLATFORM_CLASS_ID), None)
    servo    = next((d for d in detections if d[0] == SERVO_CLASS_ID), None)
    if platform is None or servo is None:
        return None
    return platform, servo


def platform_servo_gap_px(detections):

    edges = platform_servo_edges(detections)
    if edges is None:
        return None
    (_, _, _, py1, _, _), (_, _, _, _, _, sy2) = edges
    return abs(py1 - sy2)


def draw_platform_servo_gap(frame, detections):

    edges = platform_servo_edges(detections)
    if edges is None:
        return
    platform, servo = edges

    _, _, px1, py1, px2, _ = platform
    _, _, sx1, _, sx2, sy2 = servo

    px1, py1, px2 = int(px1), int(py1), int(px2)
    sx1, sy2, sx2 = int(sx1), int(sy2), int(sx2)


    cv2.line(frame, (px1, py1), (px2, py1), GAP_LINE_COLOR, 3)
    cv2.line(frame, (sx1, sy2), (sx2, sy2), GAP_LINE_COLOR, 3)

    distance_px = abs(py1 - sy2)
    mid_x = (px1 + px2 + sx1 + sx2) // 4
    cv2.line(frame, (mid_x, py1), (mid_x, sy2), GAP_LINE_COLOR, 2)

    label = f"gap: {distance_px}px"
    lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_y = (py1 + sy2) // 2
    cv2.putText(frame, label, (mid_x + 8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GAP_LINE_COLOR, 2)


def send(ser, cmd):
    with _SERIAL_LOCK:
        ser.write((cmd + "\n").encode())
        ser.flush()


def open_bottom_cam():

    candidates = [BOTTOM_CAM_DEVICE, 0, 1]
    for src in candidates:
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[cam] opened bottom camera: {src}")
                return cap
        cap.release()
    return None


def main():
    print(f"[init] loading engine: {ENGINE_PATH}")
    detector = TRTDetector(ENGINE_PATH)

    cap = open_bottom_cam()
    if cap is None:
        print("[error] Could not open bottom camera.", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"[init] connecting to {ESP2_PORT} at {BAUD} baud...")
    ser2 = serial.Serial(ESP2_PORT, BAUD, timeout=0.1)
    time.sleep(0.5)
    ser2.reset_input_buffer()

    send(ser2, f"6 {SERVO_6_INITIAL}")
    time.sleep(0.2)

    print("=" * 60)
    print(" BOTTOM CAM  |  crop_detector.engine")
    print(" classes: 0=gCrop  1=platform  2=ycrop  3=servo")
    print(f" In the video window: Enter seeks servo 6 until the platform/servo gap drops below {TARGET_GAP_PX}px.")
    print(" In the video window: 'q' quits. Ctrl+C also quits.")
    print("=" * 60)

    seeking = False
    servo6_angle = None
    seek_status = "idle — press Enter (in this window) to start"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[warn] frame grab failed.")
                break
            t0 = time.time()

            inp, r, pad = preprocess(frame)
            raw = detector.infer(inp).reshape(detector.output_shape)
            detections = postprocess(raw, r, pad, frame.shape)
            fps = 1.0 / max(1e-6, time.time() - t0)

            if seeking:
                gap_px = platform_servo_gap_px(detections)
                if gap_px is None:
                    seek_status = "seeking: target lost (need platform+servo in frame)"
                elif gap_px < TARGET_GAP_PX:
                    seeking = False
                    seek_status = f"REACHED gap={gap_px:.0f}px servo6={servo6_angle:.1f}"
                elif servo6_angle <= SERVO_6_MIN_ANGLE:
                    seeking = False
                    seek_status = f"STOPPED at servo6 min={SERVO_6_MIN_ANGLE} gap={gap_px:.0f}px (target not reached)"
                else:
                    servo6_angle = max(SERVO_6_MIN_ANGLE, servo6_angle - SEEK_STEP_DEG)
                    send(ser2, f"6 {servo6_angle:.1f}")
                    seek_status = f"seeking: gap={gap_px:.0f}px servo6={servo6_angle:.1f}"
                    time.sleep(SEEK_LOOP_SLEEP_S)

            draw(frame, detections, fps)
            cv2.putText(frame, seek_status, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, GAP_LINE_COLOR, 2)
            cv2.imshow("Bottom Cam — crop_detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (13, 10):
                servo6_angle = SERVO_6_START_ANGLE
                send(ser2, f"6 {servo6_angle}")
                seeking = True
                seek_status = f"seeking: servo6 -> {servo6_angle:.1f} (start)"
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser2.close()
        del detector
        import gc; gc.collect()
        _CTX.pop()
        _DEV.retain_primary_context().detach()

    return 0


if __name__ == "__main__":
    sys.exit(main())
