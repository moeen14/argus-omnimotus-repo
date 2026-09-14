from __future__ import annotations

import gc
import os
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


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = os.environ.get("ASABE_CROP_ENGINE", str(ROOT / "models" / "crop_detector.engine"))
BOTTOM_CAM_DEVICE = os.environ.get(
    "ASABE_BOTTOM_CAMERA_DEVICE",
    "/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_9342D410-video-index0",
)
ESP1_PORT = os.environ.get("ASABE_ESP1_PORT", "/dev/esp32_01")
ESP2_PORT = os.environ.get("ASABE_ESP2_PORT", "/dev/esp32_02")
BAUD = 921600

IMG_SIZE = 640
CONF_THRESH = 0.65
IOU_THRESH = 0.65
GCROP_CLASS_ID = 0
PLATFORM_CLASS_ID = 1
YCROP_CLASS_ID = 2
SERVO_CLASS_ID = 3
GCROP_CONF_THRESH = 0.60
CLASS_CONF_THRESH = {GCROP_CLASS_ID: GCROP_CONF_THRESH}
CLASS_NAMES = {
    GCROP_CLASS_ID: "gCrop",
    PLATFORM_CLASS_ID: "platform",
    YCROP_CLASS_ID: "ycrop",
    SERVO_CLASS_ID: "servo",
}
CLASS_BOX_COLORS = {
    GCROP_CLASS_ID: (0, 255, 0),
    PLATFORM_CLASS_ID: (255, 0, 0),
    YCROP_CLASS_ID: (0, 255, 255),
    SERVO_CLASS_ID: (255, 0, 255),
}

SERVO_INITIAL_COMMANDS = [
    ("I", "swerve initial"),
    ("5 322", "arm horizontal initial"),
    ("6 50", "arm vertical initial"),
    ("7 51", "bottom camera/front pan initial"),
    ("8 170", "overhead camera idle"),
    ("9 25", "servo 9 idle"),
]

ENTER_KEY_CODES = (13, 10)
WINDOW_NAME = "Identification"
SENT_STATUS_SECONDS = 1.5
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
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape)) * np.dtype(np.float32).itemsize)
        self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape)) * np.dtype(np.float32).itemsize)
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
    return img.transpose(2, 0, 1)[None], r, pad


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


def postprocess(output, r, pad, orig_shape):
    pred = output[0].transpose(1, 0)
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(class_scores)), class_ids]

    thresholds = np.array([CLASS_CONF_THRESH.get(c, CONF_THRESH) for c in class_ids])
    mask = confs > thresholds
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confs = confs[mask]

    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
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
        cls_mask = class_ids == cls_id
        cls_boxes = boxes_xyxy[cls_mask]
        cls_confs = confs[cls_mask]
        keep = nms(cls_boxes, cls_confs, IOU_THRESH)
        for k in keep:
            x1, y1, x2, y2 = cls_boxes[k]
            results.append((int(cls_id), float(cls_confs[k]), float(x1), float(y1), float(x2), float(y2)))
    return results


def classify_detection(detections):


    present = {cls_id for cls_id, _conf, *_box in detections if cls_id != SERVO_CLASS_ID}
    if PLATFORM_CLASS_ID not in present:
        return None, "NO PLATFORM"
    has_green = GCROP_CLASS_ID in present
    has_yellow = YCROP_CLASS_ID in present
    if has_green and has_yellow:
        return "B", "PLATFORM + GCROP + YCROP"
    if has_green:
        return "G", "PLATFORM + GCROP"
    return "R", "PLATFORM ONLY"


def draw_detections(frame, detections):
    for cls_id, conf, x1, y1, x2, y2 in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = CLASS_BOX_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls_id, '?')} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_status(frame, status, color):
    cv2.rectangle(frame, (0, frame.shape[0] - 48), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, status, (10, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def send(ser, cmd):
    with _SERIAL_LOCK:
        ser.write((cmd + "\n").encode())
        ser.flush()


def serial_reader(ser, stop_event):

    while not stop_event.is_set():
        try:
            ser.readline()
        except Exception:
            break


def send_initial_positions(ser2):
    print("[init] sending servos to initial positions...")
    for cmd, label in SERVO_INITIAL_COMMANDS:
        send(ser2, cmd)
        print(f"  {cmd} ({label})")
        time.sleep(0.15)


def reset_board1_color_counters(ser1):
    print("[init] resetting Board 01 color counters (T0)...")
    send(ser1, "T0")
    time.sleep(0.15)


def open_bottom_camera():
    candidates = [BOTTOM_CAM_DEVICE, 0, 1, 2]
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

    print(f"[init] connecting to Board 01 {ESP1_PORT} and LED/servo board {ESP2_PORT} at {BAUD} baud...")
    ser1 = serial.Serial(ESP1_PORT, BAUD, timeout=0.1)
    ser2 = serial.Serial(ESP2_PORT, BAUD, timeout=0.1)
    time.sleep(0.5)
    ser1.reset_input_buffer()
    ser2.reset_input_buffer()

    stop_event = threading.Event()
    threading.Thread(target=serial_reader, args=(ser1, stop_event), daemon=True).start()
    threading.Thread(target=serial_reader, args=(ser2, stop_event), daemon=True).start()
    reset_board1_color_counters(ser1)
    send_initial_positions(ser2)

    cap = open_bottom_camera()
    if cap is None:
        print("[error] Could not open bottom camera.", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("=" * 60)
    print(" IDENTIFICATION")
    print(" Press Enter in the camera window to classify the current frame.")
    print(" Sends G/B/R to Board 02 RGB LED and Board 01 color counter.")
    print(" Press q in the camera window or Ctrl+C to quit.")
    print("=" * 60)

    last_detections = []
    wait_status = "WAITING FOR NEXT CASE - press Enter"
    status = wait_status
    status_color = (0, 255, 255)
    status_until = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[warn] frame grab failed.")
                break

            if status_until and time.time() >= status_until:
                status = wait_status
                status_color = (0, 255, 255)
                status_until = 0.0
                last_detections = []

            display = frame.copy()
            if last_detections:
                draw_detections(display, last_detections)
            draw_status(display, status, status_color)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in ENTER_KEY_CODES:
                inp, r, pad = preprocess(frame)
                raw = detector.infer(inp).reshape(detector.output_shape)
                last_detections = postprocess(raw, r, pad, frame.shape)
                cmd, label = classify_detection(last_detections)
                if cmd is None:
                    status = f"{label} - NO COMMAND"
                    status_color = (0, 165, 255)
                    status_until = time.time() + SENT_STATUS_SECONDS
                else:
                    send(ser1, cmd)
                    send(ser2, cmd)
                    status = f"SENT {cmd} - {label}"
                    status_color = (255, 0, 0) if cmd == "B" else (
                        (0, 255, 0) if cmd == "G" else (0, 0, 255)
                    )
                    status_until = time.time() + SENT_STATUS_SECONDS

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        ser1.close()
        ser2.close()
        del detector
        gc.collect()
        _CUDA_CONTEXT.pop()
        _CUDA_DEVICE.retain_primary_context().detach()

    return 0


if __name__ == "__main__":
    sys.exit(main())
