from __future__ import annotations

import contextlib
import gc
import os
import queue
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


PLATFORM_ENGINE_PATH = os.environ.get("ASABE_CROP_ENGINE", str(Path(__file__).resolve().parent / "crop_detector.engine"))
BALL_VERIFY_ENGINE_PATH = os.environ.get(
    "ASABE_BALL_VERIFY_ENGINE", str(Path(__file__).resolve().parent / "highcam_platform.engine"))

OVERHEAD_CAMERA_DEVICE = os.environ.get(
    "ASABE_OVERHEAD_CAMERA_DEVICE",
    "/dev/v4l/by-id/usb-046d_0825_3D19ADE0-video-index0",
)
BOTTOM_CAMERA_DEVICE = os.environ.get(
    "ASABE_BOTTOM_CAMERA_DEVICE",
    "/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_9342D410-video-index0",
)

ESP2_PORT = os.environ.get("ASABE_ESP2_PORT", "/dev/esp32_02")
BAUD = 921600

IMG_SIZE = 640
FRAME_W, FRAME_H = 640, 480

ENTER_KEY_CODES = (13, 10)


PLATFORM_CONF_THRESH = 0.65
PLATFORM_IOU_THRESH = 0.65
GCROP_CLASS_ID = 0
PLATFORM_CLASS_ID = 1
YCROP_CLASS_ID = 2
SERVO_CLASS_ID = 3
GCROP_CONF_THRESH = 0.75
CLASS_CONF_THRESH = {GCROP_CLASS_ID: GCROP_CONF_THRESH}
CLASS_NAMES = {GCROP_CLASS_ID: "gCrop", PLATFORM_CLASS_ID: "platform", YCROP_CLASS_ID: "ycrop", SERVO_CLASS_ID: "servo"}
CLASS_BOX_COLORS = {GCROP_CLASS_ID: (0, 255, 0), PLATFORM_CLASS_ID: (255, 0, 0), YCROP_CLASS_ID: (0, 255, 255), SERVO_CLASS_ID: (255, 0, 255)}


BALL_VERIFY_BALL_CLASS_ID = 0
BALL_VERIFY_GCROP_CLASS_ID = 1
BALL_VERIFY_PLATFORM_CLASS_ID = 2
BALL_VERIFY_YCROP_CLASS_ID = 3
BALL_VERIFY_CLASS_CONF_THRESH = {
    BALL_VERIFY_BALL_CLASS_ID: 0.30,
    BALL_VERIFY_GCROP_CLASS_ID: 0.40,
    BALL_VERIFY_PLATFORM_CLASS_ID: 0.85,
    BALL_VERIFY_YCROP_CLASS_ID: 0.85,
}
BALL_VERIFY_IOU_THRESH = 0.75


BALL_VERIFY_SETTLE_S = 2.0


BALL_VERIFY_CONFIRM_FRAMES = 5
BALL_VERIFY_CHECK_WINDOW_S = 4.0


SERVO_6_INITIAL = 50
SERVO_6_SEEK_START_ANGLE = 305
SERVO_6_SEEK_MIN_ANGLE = 290
SERVO_6_SEEK_STEP_DEG = 0.5
SERVO_6_SEEK_TARGET_GAP_PX = 70
SERVO_6_SEEK_LOOP_SLEEP_S = 0.15


R_PLACEHOLDER_DISPENSE_CMD = "D"
R_PLACEHOLDER_DISPENSE_WAIT_S = 3.0
R_PLACEHOLDER_SERVO9_TEMP_ANGLE = 145
CAMERA_FRONT_ANGLE = 266
SERVO_8_IDLE_ANGLE = 170
SERVO_9_IDLE_ANGLE = 116.5
SERVO_9_STRAIGHT_ANGLE = 116.5


SERVO_SETTLE_TIMEOUT_S = 2.0


DISPENSE_SAFE_WAIT_TIMEOUT_S = 4.5
SERVO_DONE_EVENTS = {servo_id: threading.Event() for servo_id in range(5, 10)}

_SERIAL_LOCK = threading.Lock()

cuda.init()
_CUDA_DEVICE = cuda.Device(0)
_CUDA_CONTEXT = _CUDA_DEVICE.retain_primary_context()
_CUDA_CONTEXT.push()


def log_event(*args):
    print(" ".join(str(a) for a in args))


def send(ser, cmd):
    with _SERIAL_LOCK:
        ser.write((cmd + "\n").encode())
        ser.flush()


def settle_servo_ids_for_command(cmd):


    parts = cmd.split()
    if len(parts) == 2:
        try:
            servo_id = int(parts[0])
            float(parts[1])
        except ValueError:
            return None
        if 5 <= servo_id <= 9:
            return servo_id
    return None


def wait_for_servo_settle(servo_id, label, timeout_s=SERVO_SETTLE_TIMEOUT_S):


    if not SERVO_DONE_EVENTS[servo_id].wait(timeout_s):
        log_event(f"[settle] {label} did NOT confirm within {timeout_s:.1f}s "
                  f"(servo {servo_id} never sent DONE) — proceeding anyway.")
        return False
    log_event(f"[settle] {label} confirmed by Board 02 (DONE).")
    return True


def send_and_settle(ser, cmd, timeout_s=SERVO_SETTLE_TIMEOUT_S):


    servo_id = settle_servo_ids_for_command(cmd)
    if servo_id is not None:
        SERVO_DONE_EVENTS[servo_id].clear()
    send(ser, cmd)
    if servo_id is not None:
        wait_for_servo_settle(servo_id, f"servo {servo_id} -> {cmd.split()[1]}", timeout_s=timeout_s)


def serial_reader(ser, stop_event):


    while not stop_event.is_set():
        try:
            raw = ser.readline()
        except Exception:
            break
        if not raw:
            continue
        line = raw.decode(errors="ignore").strip()
        if not line:
            continue
        if line.startswith("ERR,"):
            log_event(f"[err] Board 02 rejected command: {line}")
        elif line.startswith("DONE,"):
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    servo_id = int(parts[1])
                except ValueError:
                    servo_id = None
                if servo_id in SERVO_DONE_EVENTS:
                    SERVO_DONE_EVENTS[servo_id].set()


class ThreadedCamera:
    def __init__(self, source, width=640, height=480):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        log_event(f"[Camera] Started at {actual_w}x{actual_h}")

        self.q = queue.Queue(maxsize=3)
        self.running = True
        self.paused = False
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    break
                self.q.put(frame)
            else:
                time.sleep(0.001)

    def isOpened(self):
        return self.cap.isOpened()

    def read(self, timeout=2.0):
        if not self.running and self.q.empty():
            return False, None
        try:
            return True, self.q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def pause(self):
        self.paused = True
        with self.q.mutex:
            self.q.queue.clear()

    def resume(self):
        self.paused = False

    def release(self):
        self.running = False
        self.t.join()
        self.cap.release()


@contextlib.contextmanager
def suppress_stderr():


    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


def open_camera(label, primary_source, fallback_indices=(0, 1, 2)):
    candidates = [primary_source] + list(fallback_indices)
    for source in candidates:
        cam = ThreadedCamera(source, FRAME_W, FRAME_H)
        if cam.isOpened():
            ok, _ = cam.read(timeout=10.0)
            if ok:
                log_event(f"[init] {label} camera opened on {source}")
                return cam
        cam.release()
    raise RuntimeError(f"Could not open {label} camera (tried {candidates})")


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
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


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


def postprocess(output, r, pad, orig_shape, conf_thresh=PLATFORM_CONF_THRESH, iou_thresh=PLATFORM_IOU_THRESH,
                 class_conf_thresh=None):


    if class_conf_thresh is None:
        class_conf_thresh = CLASS_CONF_THRESH
    pred = output[0].transpose(1, 0)
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(class_scores)), class_ids]

    thresholds = np.array([class_conf_thresh.get(c, conf_thresh) for c in class_ids])
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


def draw_detections(frame, detections):
    for cls_id, conf, x1, y1, x2, y2 in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box_color = CLASS_BOX_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"{CLASS_NAMES.get(cls_id, 'unknown')} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), box_color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def platform_servo_gap_px(detections):


    platform = next((d for d in detections if d[0] == PLATFORM_CLASS_ID), None)
    servo = next((d for d in detections if d[0] == SERVO_CLASS_ID), None)
    if platform is None or servo is None:
        return None
    _, _, _, py1, _, _ = platform
    _, _, _, _, _, sy2 = servo
    return abs(py1 - sy2)


class BallVerifyProcessor:

    def __init__(self, engine_path):
        log_event(f"Loading ball-verify TRT Engine: {engine_path}")
        self.detector = TRTDetector(engine_path)
        log_event("Ball-verify processor ready.")

    def detect(self, frame):
        inp, r, pad = preprocess(frame)
        raw = self.detector.infer(inp).reshape(self.detector.output_shape)
        return postprocess(raw, r, pad, frame.shape, iou_thresh=BALL_VERIFY_IOU_THRESH,
                            class_conf_thresh=BALL_VERIFY_CLASS_CONF_THRESH)


def ball_inside_platform(detections):


    ball = max((d for d in detections if d[0] == BALL_VERIFY_BALL_CLASS_ID),
               key=lambda d: d[1], default=None)
    platform = max((d for d in detections if d[0] == BALL_VERIFY_PLATFORM_CLASS_ID),
                   key=lambda d: d[1], default=None)
    if ball is None or platform is None:
        return False
    _, _, bx1, by1, bx2, by2 = ball
    _, _, px1, py1, px2, py2 = platform
    return bx1 >= px1 and by1 >= py1 and bx2 <= px2 and by2 <= py2


def main():
    log_event(f"Connecting to {ESP2_PORT} at {BAUD} baud...")
    ser2 = serial.Serial(ESP2_PORT, BAUD, timeout=0.1)
    time.sleep(0.5)
    ser2.reset_input_buffer()

    stop_event = threading.Event()
    threading.Thread(target=serial_reader, args=(ser2, stop_event), daemon=True).start()

    log_event("[init] Sending servo 6/8/9 to idle...")


    send_and_settle(ser2, f"6 {SERVO_6_INITIAL}", timeout_s=DISPENSE_SAFE_WAIT_TIMEOUT_S)


    send_and_settle(ser2, f"8 {CAMERA_FRONT_ANGLE}")
    send_and_settle(ser2, f"9 {SERVO_9_IDLE_ANGLE}")

    platform_processor = None
    ball_verify_processor = None
    cap_bottom = None
    cap_overhead = None
    try:
        log_event(f"Loading platform TRT Engine: {PLATFORM_ENGINE_PATH}")
        platform_processor = TRTDetector(PLATFORM_ENGINE_PATH)
        ball_verify_processor = BallVerifyProcessor(BALL_VERIFY_ENGINE_PATH)

        cap_bottom = open_camera("bottom", BOTTOM_CAMERA_DEVICE)
        cap_overhead = open_camera("overhead", OVERHEAD_CAMERA_DEVICE)
        cap_bottom.pause()
        cap_overhead.pause()


        phase = "idle"
        seek_servo6_angle = SERVO_6_SEEK_START_ANGLE
        phase_start = 0.0
        attempt_count = 0


        verify_streak = 0
        verify_check_start = 0.0
        attempt_start_time = 0.0
        last_dispense_duration_s = 0.0

        def start_attempt():
            nonlocal phase, seek_servo6_angle, attempt_count, attempt_start_time
            attempt_count += 1
            attempt_start_time = time.time()
            log_event(f"[task] Attempt {attempt_count}: seeking servo6 to dispense gap.")
            seek_servo6_angle = SERVO_6_SEEK_START_ANGLE
            send(ser2, f"6 {seek_servo6_angle}")
            cap_bottom.resume()
            phase = "seeking"

        log_event("=" * 60)
        log_event(" BALL DISPENSE TEST")
        log_event(" Place the robot at the correct front-ToF distance by hand.")
        log_event(" Press Enter in the video window to attempt a dispense.")
        log_event(" 'q' in the video window or Ctrl+C to quit.")
        log_event("=" * 60)

        with suppress_stderr():
            while True:
                overlay = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

                if phase == "idle":
                    cv2.putText(overlay, "IDLE - press Enter to dispense", (10, FRAME_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                elif phase == "seeking":
                    ret, frame = cap_bottom.read()
                    if ret:
                        inp, r, pad = preprocess(frame)
                        raw = platform_processor.infer(inp).reshape(platform_processor.output_shape)
                        detections = postprocess(raw, r, pad, frame.shape)
                        overlay = frame.copy()
                        draw_detections(overlay, detections)
                        gap_px = platform_servo_gap_px(detections)
                        cv2.putText(overlay, f"gap={'n/a' if gap_px is None else f'{gap_px:.0f}px'} "
                                              f"servo6={seek_servo6_angle:.1f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        reached = gap_px is not None and gap_px < SERVO_6_SEEK_TARGET_GAP_PX
                        hit_min = seek_servo6_angle <= SERVO_6_SEEK_MIN_ANGLE
                        if reached or hit_min:
                            cap_bottom.pause()
                            send_and_settle(ser2, f"6 {seek_servo6_angle:.1f}",
                                             timeout_s=DISPENSE_SAFE_WAIT_TIMEOUT_S)
                            if reached:
                                log_event(f"[task] seek: reached gap={gap_px:.0f}px — sending '{R_PLACEHOLDER_DISPENSE_CMD}'.")
                            else:
                                log_event(f"[task] seek: hit min angle {SERVO_6_SEEK_MIN_ANGLE} "
                                          f"without reaching target — sending '{R_PLACEHOLDER_DISPENSE_CMD}' anyway.")
                            send(ser2, R_PLACEHOLDER_DISPENSE_CMD)
                            phase = "dispensing"
                            phase_start = time.time()
                        else:
                            seek_servo6_angle = max(SERVO_6_SEEK_MIN_ANGLE, seek_servo6_angle - SERVO_6_SEEK_STEP_DEG)
                            send(ser2, f"6 {seek_servo6_angle:.1f}")
                            time.sleep(SERVO_6_SEEK_LOOP_SLEEP_S)

                elif phase == "dispensing":
                    cv2.putText(overlay, f"DISPENSING - hold {R_PLACEHOLDER_DISPENSE_WAIT_S:.1f}s",
                                (10, FRAME_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    if time.time() - phase_start >= R_PLACEHOLDER_DISPENSE_WAIT_S:


                        send_and_settle(ser2, f"6 {SERVO_6_INITIAL}", timeout_s=DISPENSE_SAFE_WAIT_TIMEOUT_S)
                        send_and_settle(ser2, f"8 {CAMERA_FRONT_ANGLE}")
                        send_and_settle(ser2, f"9 {R_PLACEHOLDER_SERVO9_TEMP_ANGLE}")
                        time.sleep(BALL_VERIFY_SETTLE_S)
                        cap_overhead.resume()
                        phase = "verify_checking"
                        verify_streak = 0
                        verify_check_start = time.time()
                        log_event("[task] verify: checking ball placement.")

                elif phase == "verify_checking":
                    ret, frame = cap_overhead.read()
                    if ret:
                        detections = ball_verify_processor.detect(frame)
                        overlay = frame.copy()
                        draw_detections(overlay, detections)
                        inside = ball_inside_platform(detections)
                        verify_streak = verify_streak + 1 if inside else 0
                        elapsed = time.time() - verify_check_start
                        cv2.putText(overlay, f"ball inside platform: {inside}  "
                                              f"streak {verify_streak}/{BALL_VERIFY_CONFIRM_FRAMES}  "
                                              f"{elapsed:.1f}/{BALL_VERIFY_CHECK_WINDOW_S:.1f}s",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0) if inside else (0, 0, 255), 2)
                        if verify_streak >= BALL_VERIFY_CONFIRM_FRAMES:
                            cap_overhead.pause()
                            last_dispense_duration_s = time.time() - attempt_start_time
                            log_event(f"[task] verify: ball confirmed inside platform bounding box — SUCCESS "
                                      f"(dispense took {last_dispense_duration_s:.1f}s).")
                            send_and_settle(ser2, f"8 {SERVO_8_IDLE_ANGLE}")
                            send_and_settle(ser2, f"9 {SERVO_9_IDLE_ANGLE}")
                            phase = "done"
                        elif elapsed >= BALL_VERIFY_CHECK_WINDOW_S:
                            cap_overhead.pause()
                            log_event(f"[task] verify: ball not confirmed within {BALL_VERIFY_CHECK_WINDOW_S:.1f}s "
                                      f"— re-seeking servo6.")
                            seek_servo6_angle = SERVO_6_SEEK_START_ANGLE
                            send(ser2, f"6 {seek_servo6_angle}")
                            cap_bottom.resume()
                            phase = "verify_seeking"

                elif phase == "verify_seeking":
                    ret, frame = cap_bottom.read()
                    if ret:
                        inp, r, pad = preprocess(frame)
                        raw = platform_processor.infer(inp).reshape(platform_processor.output_shape)
                        detections = postprocess(raw, r, pad, frame.shape)
                        overlay = frame.copy()
                        draw_detections(overlay, detections)
                        gap_px = platform_servo_gap_px(detections)
                        cv2.putText(overlay, f"gap={'n/a' if gap_px is None else f'{gap_px:.0f}px'} "
                                              f"servo6={seek_servo6_angle:.1f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        reached = gap_px is not None and gap_px < SERVO_6_SEEK_TARGET_GAP_PX
                        hit_min = seek_servo6_angle <= SERVO_6_SEEK_MIN_ANGLE
                        if reached or hit_min:
                            cap_bottom.pause()
                            send_and_settle(ser2, f"6 {seek_servo6_angle:.1f}",
                                             timeout_s=DISPENSE_SAFE_WAIT_TIMEOUT_S)
                            if reached:
                                log_event(f"[task] verify: servo6 reached gap={gap_px:.0f}px — resending '{R_PLACEHOLDER_DISPENSE_CMD}'.")
                            else:
                                log_event(f"[task] verify: servo6 hit min angle without reaching target "
                                          f"— resending '{R_PLACEHOLDER_DISPENSE_CMD}' anyway.")
                            send(ser2, R_PLACEHOLDER_DISPENSE_CMD)
                            phase = "verify_dispensing"
                            phase_start = time.time()
                        else:
                            seek_servo6_angle = max(SERVO_6_SEEK_MIN_ANGLE, seek_servo6_angle - SERVO_6_SEEK_STEP_DEG)
                            send(ser2, f"6 {seek_servo6_angle:.1f}")
                            time.sleep(SERVO_6_SEEK_LOOP_SLEEP_S)

                elif phase == "verify_dispensing":
                    cv2.putText(overlay, f"RE-DISPENSING - hold {R_PLACEHOLDER_DISPENSE_WAIT_S:.1f}s",
                                (10, FRAME_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    if time.time() - phase_start >= R_PLACEHOLDER_DISPENSE_WAIT_S:
                        send_and_settle(ser2, f"6 {SERVO_6_INITIAL}", timeout_s=DISPENSE_SAFE_WAIT_TIMEOUT_S)
                        time.sleep(BALL_VERIFY_SETTLE_S)
                        cap_overhead.resume()
                        phase = "verify_checking"
                        verify_streak = 0
                        verify_check_start = time.time()
                        log_event("[task] verify: servo6 back to initial — rechecking ball placement.")

                elif phase == "done":
                    cv2.putText(overlay, f"DONE in {last_dispense_duration_s:.1f}s - press Enter to try again",
                                (10, FRAME_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(overlay, f"PHASE: {phase}", (10, FRAME_H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow("BALL DISPENSE TEST", overlay)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log_event("\n[info] stopped by user.")
                    break
                elif key in ENTER_KEY_CODES and phase in ("idle", "done"):
                    start_attempt()

    except KeyboardInterrupt:
        log_event("\n[info] stopped by user.")
    finally:
        send(ser2, "X")
        stop_event.set()
        if cap_bottom:
            cap_bottom.release()
        if cap_overhead:
            cap_overhead.release()
        cv2.destroyAllWindows()
        ser2.close()
        if platform_processor:
            del platform_processor
        if ball_verify_processor:
            del ball_verify_processor.detector
        gc.collect()
        _CUDA_CONTEXT.pop()
        _CUDA_DEVICE.retain_primary_context().detach()
        log_event("\nCleanup complete.")


if __name__ == "__main__":
    sys.exit(main() or 0)
