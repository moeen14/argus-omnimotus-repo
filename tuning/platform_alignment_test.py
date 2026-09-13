from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import serial
except ImportError:
    print("pyserial not found. Install it or activate nanosam-env.")
    sys.exit(1)

import tensorrt as trt
import pycuda.driver as cuda


MAX_SPEED = 120
MIN_PULSE_MS = 5
MAX_PULSE_MS = 30
ERROR_DEADBAND_PX = 3
ERROR_SATURATION_PX = 200
LOOP_SLEEP_S = 0.05
EMA_ALPHA = 0.4

ENGINE_PATH = os.environ.get("ASABE_CROP_ENGINE", str(Path(__file__).resolve().parent / "crop_detector.engine"))
IMG_SIZE = 640
CONF_THRESH = 0.60
IOU_THRESH = 0.60
GCROP_CLASS_ID = 0
PLATFORM_CLASS_ID = 1
YCROP_CLASS_ID = 2
SERVO_CLASS_ID = 3
GCROP_CONF_THRESH = 0.60
CLASS_CONF_THRESH = {GCROP_CLASS_ID: GCROP_CONF_THRESH}

ESP1_PORT = os.environ.get("ASABE_ESP1_PORT", "/dev/esp32_01")
ESP2_PORT = os.environ.get("ASABE_ESP2_PORT", "/dev/esp32_02")
BAUD = 921600

SERVO_5_INITIAL = 322
SERVO_6_INITIAL = 50
SERVO_7_INITIAL = 51


SERVO_5_OFFSET_YELLOW_FARTHER = 323.5
SERVO_5_OFFSET_GREEN_FARTHER_OR_EQUAL = 321.5

_SERIAL_LOCK = threading.Lock()

cuda.init()
_CUDA_DEVICE = cuda.Device(0)
_CUDA_CONTEXT = _CUDA_DEVICE.retain_primary_context()
_CUDA_CONTEXT.push()


class TRTDetector:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        in_size = int(np.prod(self.input_shape)) * np.dtype(np.float32).itemsize
        out_size = int(np.prod(self.output_shape)) * np.dtype(np.float32).itemsize
        self.d_input = cuda.mem_alloc(in_size)
        self.d_output = cuda.mem_alloc(out_size)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

    def infer(self, input_array):
        input_array = np.ascontiguousarray(input_array, dtype=np.float32)
        cuda.memcpy_htod_async(self.d_input, input_array, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return np.array(self.h_output)


def letterbox(frame, size=IMG_SIZE):
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[top:top + nh, left:left + nw] = resized
    return padded, r, (left, top)


def preprocess(frame):
    padded, r, pad = letterbox(frame, IMG_SIZE)
    img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]
    return img, r, pad


def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def postprocess(output, r, pad, orig_shape, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    pred = output[0].transpose(1, 0)
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(class_scores)), class_ids]

    thresholds = np.array([CLASS_CONF_THRESH.get(c, conf_thresh) for c in class_ids])
    mask = confs > thresholds
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confs = confs[mask]

    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    left, top = pad
    boxes_xyxy[:, [0, 2]] -= left
    boxes_xyxy[:, [1, 3]] -= top
    boxes_xyxy /= r

    H, W = orig_shape[:2]
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, W - 1)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, H - 1)

    results = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = boxes_xyxy[cls_mask]
        cls_confs = confs[cls_mask]
        keep = nms(cls_boxes, cls_confs, iou_thresh)
        for k in keep:
            x1, y1, x2, y2 = cls_boxes[k]
            results.append((int(cls_id), float(cls_confs[k]), float(x1), float(y1), float(x2), float(y2)))
    return results


CLASS_NAMES = {GCROP_CLASS_ID: "gCrop", PLATFORM_CLASS_ID: "platform", YCROP_CLASS_ID: "ycrop", SERVO_CLASS_ID: "servo"}
CLASS_BOX_COLORS = {GCROP_CLASS_ID: (0, 255, 0), PLATFORM_CLASS_ID: (255, 0, 0), YCROP_CLASS_ID: (0, 255, 255), SERVO_CLASS_ID: (255, 0, 255)}


def draw_detections(frame, detections):
    for cls_id, conf, x1, y1, x2, y2 in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box_color = CLASS_BOX_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"{CLASS_NAMES.get(cls_id, 'unknown')} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), box_color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def compute_platform_error(detections, frame_w):

    best = None
    for cls_id, conf, x1, y1, x2, y2 in detections:
        if cls_id != PLATFORM_CLASS_ID:
            continue
        if best is None or conf > best[0]:
            best = (conf, x1, y1, x2, y2)
    if best is None:
        return None
    _, x1, y1, x2, y2 = best
    platform_cx = (x1 + x2) / 2.0
    img_cx = frame_w / 2.0
    return platform_cx - img_cx, platform_cx


def closest_edge_distance(detections, cls_id, ref_x):


    best = None
    for c, conf, x1, y1, x2, y2 in detections:
        if c != cls_id:
            continue
        if best is None or conf > best[0]:
            best = (conf, x1, x2)
    if best is None:
        return None
    _, x1, x2 = best
    return min(abs(x1 - ref_x), abs(x2 - ref_x))


def classify_case(detections):

    present = {cls_id for cls_id, conf, *_ in detections}
    if PLATFORM_CLASS_ID not in present:
        return None
    has_green = GCROP_CLASS_ID in present
    has_yellow = YCROP_CLASS_ID in present
    if has_green and has_yellow:
        return "PLATFORM + GREEN + YELLOW", (255, 0, 0), "B"
    if has_green:
        return "PLATFORM + GREEN", (0, 255, 0), "G"
    return "PLATFORM ONLY", (0, 0, 255), "R"


def compute_pulse_ms(error_px):
    mag = min(abs(error_px), ERROR_SATURATION_PX)
    if mag <= ERROR_DEADBAND_PX:
        return 0.0
    span = ERROR_SATURATION_PX - ERROR_DEADBAND_PX
    frac = (mag - ERROR_DEADBAND_PX) / span
    return MIN_PULSE_MS + frac * (MAX_PULSE_MS - MIN_PULSE_MS)


def serial_reader(ser, stop_event):

    while not stop_event.is_set():
        try:
            ser.readline()
        except Exception:
            break


def send(ser, cmd):
    with _SERIAL_LOCK:
        ser.write((cmd + "\n").encode())
        ser.flush()


def terminal_listener(ser, stop_event):

    pattern = re.compile(r"^([567])\s+(\d+)$")
    while not stop_event.is_set():
        try:
            raw = input().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        match = pattern.match(raw)
        if not match:
            print(f"  ? unrecognized servo command '{raw}' (expected '5 <angle>', '6 <angle>', or '7 <angle>')")
            continue
        servo_id, angle = match.group(1), match.group(2)
        send(ser, f"{servo_id} {angle}")
        print(f"  >>> servo {servo_id} -> {angle}")


def main():
    print(f"Connecting to {ESP1_PORT} and {ESP2_PORT} at {BAUD} baud...")
    ser1 = serial.Serial(ESP1_PORT, BAUD, timeout=0.1)
    ser2 = serial.Serial(ESP2_PORT, BAUD, timeout=0.1)
    time.sleep(0.5)
    ser1.reset_input_buffer()
    ser2.reset_input_buffer()

    stop_event = threading.Event()
    threading.Thread(target=serial_reader, args=(ser1, stop_event), daemon=True).start()
    threading.Thread(target=serial_reader, args=(ser2, stop_event), daemon=True).start()

    send(ser2, "T")
    time.sleep(0.2)
    send(ser2, f"5 {SERVO_5_INITIAL}")
    send(ser2, f"6 {SERVO_6_INITIAL}")
    send(ser2, f"7 {SERVO_7_INITIAL}")
    time.sleep(0.2)

    listener = threading.Thread(target=terminal_listener, args=(ser2, stop_event), daemon=True)
    listener.start()

    detector = TRTDetector(ENGINE_PATH)

    cap = None
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened() and cap.read()[0]:
            print(f"[init] Camera on index {idx}")
            break
        cap.release()
    if cap is None or not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("=" * 60)
    print(" PLATFORM ALIGNMENT TEST")
    print(f" MAX_SPEED={MAX_SPEED}  pulse=[{MIN_PULSE_MS},{MAX_PULSE_MS}]ms  deadband={ERROR_DEADBAND_PX}px")
    print(" Status shown in the video window. Type '6 <angle>' or '7 <angle>' + Enter for manual servo moves.")
    print(" Press 'q' (in the video window) or Ctrl+C to quit.")
    print("=" * 60)

    aligned_done = False
    case_sent = False
    case_label, case_color = None, None
    smoothed_cx = None
    servo5_offset_sent = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[warn] frame grab failed, stopping.")
                break

            inp, r, pad = preprocess(frame)
            raw = detector.infer(inp).reshape(detector.output_shape)
            detections = postprocess(raw, r, pad, frame.shape)
            platform_info = compute_platform_error(detections, frame.shape[1])

            if platform_info is None:
                smoothed_cx = None
            else:
                _, raw_cx = platform_info
                smoothed_cx = raw_cx if smoothed_cx is None else EMA_ALPHA * raw_cx + (1 - EMA_ALPHA) * smoothed_cx
                img_cx = frame.shape[1] / 2.0
                platform_info = (smoothed_cx - img_cx, smoothed_cx)

            if not aligned_done:
                error_info = platform_info
                if error_info is None:
                    send(ser2, "X")
                    status = "NO TARGET"
                    color = (0, 0, 255)
                else:
                    error, platform_cx = error_info
                    pulse_ms = compute_pulse_ms(error)
                    if pulse_ms <= 0:
                        send(ser2, "X")
                        status = f"ALIGNED err={error:+.1f}px (locked)"
                        color = (0, 255, 0)
                        aligned_done = True
                    else:
                        direction = "R" if error > 0 else "L"
                        status = f"{'RIGHT' if direction == 'R' else 'LEFT'} err={error:+.1f}px pulse={pulse_ms:.0f}ms"
                        color = (0, 165, 255)
                        send(ser2, f"{direction}{MAX_SPEED}")
                        time.sleep(pulse_ms / 1000.0)
                        send(ser2, "X")
            else:
                status = "ALIGNED (locked, no longer adjusting)"
                color = (0, 255, 0)

            if aligned_done and not case_sent:
                case = classify_case(detections)
                if case is not None:
                    case_label, case_color, case_cmd = case
                    send(ser1, case_cmd)
                    send(ser2, case_cmd)
                    case_sent = True

            if case_sent:
                cv2.circle(frame, (20, 20), 10, case_color, -1)
                cv2.putText(frame, case_label, (38, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, case_color, 2)

            draw_detections(frame, detections)

            green_dist, yellow_dist = None, None
            if platform_info is not None:
                _, platform_cx = platform_info
                dist_y = 60
                green_dist = closest_edge_distance(detections, GCROP_CLASS_ID, platform_cx)
                if green_dist is not None:
                    cv2.putText(frame, f"Green-platform dist: {green_dist:.0f}px", (10, dist_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    dist_y += 25
                yellow_dist = closest_edge_distance(detections, YCROP_CLASS_ID, platform_cx)
                if yellow_dist is not None:
                    cv2.putText(frame, f"Yellow-platform dist: {yellow_dist:.0f}px", (10, dist_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if aligned_done and not servo5_offset_sent and green_dist is not None and yellow_dist is not None:
                if yellow_dist > green_dist:
                    send(ser2, f"5 {SERVO_5_OFFSET_YELLOW_FARTHER}")
                else:
                    send(ser2, f"5 {SERVO_5_OFFSET_GREEN_FARTHER_OR_EQUAL}")
                servo5_offset_sent = True

            cv2.putText(frame, status, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("PLATFORM ALIGNMENT TEST", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[info] stopped by user.")
                break

            time.sleep(LOOP_SLEEP_S)

    except KeyboardInterrupt:
        print("\n[info] stopped by user.")
    finally:
        send(ser2, "X")
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        ser1.close()
        ser2.close()
        del detector
        import gc
        gc.collect()
        _CUDA_CONTEXT.pop()
        _CUDA_DEVICE.retain_primary_context().detach()


if __name__ == "__main__":
    main()
