from __future__ import annotations

import contextlib
import gc
import math
import os
import queue
import re
import statistics
import sys
import threading
import time
from datetime import datetime
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


LANE_ENGINE_PATH = os.environ.get("ASABE_LANE_ENGINE", str(Path(__file__).resolve().parent / "highcam_rail.engine"))
PLATFORM_ENGINE_PATH = os.environ.get("ASABE_CROP_ENGINE", str(Path(__file__).resolve().parent / "crop_detector.engine"))

OVERHEAD_CAMERA_DEVICE = os.environ.get(
    "ASABE_OVERHEAD_CAMERA_DEVICE",
    "/dev/v4l/by-id/usb-046d_0825_3D19ADE0-video-index0",
)
BOTTOM_CAMERA_DEVICE = os.environ.get(
    "ASABE_BOTTOM_CAMERA_DEVICE",
    "/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_9342D410-video-index0",
)

ESP1_PORT = os.environ.get("ASABE_ESP1_PORT", "/dev/esp32_01")
ESP2_PORT = os.environ.get("ASABE_ESP2_PORT", "/dev/esp32_02")
BAUD = 921600

IMG_SIZE = 640
FRAME_W, FRAME_H = 640, 480


SWERVE_T_SCAN_DIRECTION = "L"


TOF_LABELS = ["FRS", "BRS", "BLS", "FLF", "FLS", "CAM", "FRF"]


TURNAROUND_SIDE_TOF_LABELS = {"R": ("BRS", "FRS"), "L": ("BLS", "FLS")}


IMU_CALIBRATION_MIN_SAMPLES = 300
IMU_STILL_GYRO_MAX_DPS = 2.0


TURNAROUND_TOF_THRESHOLD_MM = 120.0
TURNAROUND_TARGET_DEG = 180.0
TURNAROUND_TOLERANCE_DEG = 5.0


FRONT_TOF_LABEL = "CAM"


FRONT_TOF_TARGET_MM = 105.0


FRONT_TOF_DEADBAND_MM = 4.0


FRONT_TOF_MAX_SPEED = 140
FRONT_TOF_MIN_PULSE_MS = 20
FRONT_TOF_MAX_PULSE_MS = 40
FRONT_TOF_ERROR_SATURATION_MM = 200.0


FRONT_TOF_MIN_SPEED = 100
FRONT_TOF_SPEED_RAMP_START_MM = 30.0


FRONT_TOF_LOOP_SLEEP_S = 0.05


SERVO_6_INITIAL = 50


PLACEHOLDER_SERVO_SETTLE_S = 1.5


B_PLACEHOLDER_SG90_ANGLE_1 = 0
B_PLACEHOLDER_SG90_ANGLE_2 = 180
B_PLACEHOLDER_SERVO6_ANGLE_2 = 340


B_PLACEHOLDER_SERVO5_ANGLE_YCROP_LEFT = 312
B_PLACEHOLDER_SERVO5_ANGLE_YCROP_RIGHT = 330


SERVO_6_SEEK_START_ANGLE = 305
SERVO_6_SEEK_MIN_ANGLE = 290
SERVO_6_SEEK_STEP_DEG = 0.5
SERVO_6_SEEK_TARGET_GAP_PX = 70
SERVO_6_SEEK_LOOP_SLEEP_S = 0.15
R_PLACEHOLDER_DISPENSE_CMD = "D"
R_PLACEHOLDER_DISPENSE_WAIT_S = 3.0
R_PLACEHOLDER_SERVO9_TEMP_ANGLE = 145


BALL_VERIFY_ENGINE_PATH = os.environ.get(
    "ASABE_BALL_VERIFY_ENGINE", str(Path(__file__).resolve().parent / "highcam_platform.engine"))
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

def log_event(*args):


    print(" ".join(str(a) for a in args))


NUM_CLASSES = 1
LANE_CONF_THRESH = 0.35
LANE_IOU_THRESH = 0.45
MASK_THRESH = 0.5


SRC_PTS = np.array([[78, 291], [142, 56], [463, 57], [520, 290]], dtype=np.float32)
SRC_CM = np.array([[0, 0], [0, 30.5], [30.5, 30.5], [30.5, 0]], dtype=np.float32)
PIXELS_PER_CM = 3.0
ORIGIN_X_OFFSET = 250.0
ORIGIN_Y_OFFSET = 660.0
THRESHOLD_Y = 80

DEFAULT_RAIL_WIDTH_BEV = 111
DEFAULT_SLOPE_DIFF = 0.0
WIDTH_EMA_ALPHA = 0.1
BEV_MIN_COMPONENT_AREA = 150
BEV_MASK_CLEAN_KERNEL = np.ones((3, 3), np.uint8)


LINE_SMOOTHING_ALPHA = 0.3

IMAGE_CENTER_X = FRAME_W / 2.0


PLATFORM_CONF_THRESH = 0.65
PLATFORM_IOU_THRESH = 0.65
GCROP_CLASS_ID = 0
PLATFORM_CLASS_ID = 1
YCROP_CLASS_ID = 2
SERVO_CLASS_ID = 3
GCROP_CONF_THRESH = 0.75
CLASS_CONF_THRESH = {GCROP_CLASS_ID: GCROP_CONF_THRESH}
PLATFORM_EMA_ALPHA = 0.4


ANGLE_DEADBAND_DEG = 0.5
CENTER_DEADBAND_PX = 2.0
PLATFORM_ERROR_DEADBAND_PX = 3.0


PLATFORM_ERROR_DEADBAND_R_PX = 35.0


ALIGN_RECHECK_DELAY_S = 1.0


ALIGN_RECHECK_ENABLED = True

PLATFORM_LOOP_SLEEP_S = 0.05


MAX_CONSECUTIVE_READ_FAILURES = 5


ANGLE_MAX_SPEED = 110
ANGLE_MIN_PULSE_MS = 6
ANGLE_MAX_PULSE_MS = 20
ANGLE_ERROR_SATURATION_DEG = 15.0


CENTER_ERROR_SATURATION_PX = 200.0

PLATFORM_MAX_SPEED = 120
PLATFORM_MIN_PULSE_MS = 5
PLATFORM_MAX_PULSE_MS = 30
PLATFORM_ERROR_SATURATION_PX = 200.0


PLATFORM_REACQUIRE_PWM = 100
PLATFORM_REACQUIRE_PULSE_MS = 30


RPM_DETAIL_RE = re.compile(r"(BR|FR|BL|FL):\s*(-?\d+(?:\.\d+)?)")
WHEEL_STUCK_RPM_THRESHOLD = 5.0


WHEEL_STUCK_CHECK_DELAY_S = 0.08


WHEEL_STUCK_DURATION_S = 5.0


WHEEL_STUCK_PULSE_BOOST_INCREMENT = 1.5


WHEEL_STUCK_PWM_BOOST_START_STEP = 3
WHEEL_STUCK_PWM_BOOST_BASE = 1.2
WHEEL_STUCK_PWM_BOOST_INCREMENT = 0.2
MAX_MOTOR_PWM = 255


def stuck_pulse_boost(step):


    return 1.0 if step <= 0 else step * WHEEL_STUCK_PULSE_BOOST_INCREMENT


def stuck_pwm_boost(step):


    if step < WHEEL_STUCK_PWM_BOOST_START_STEP:
        return 1.0
    return WHEEL_STUCK_PWM_BOOST_BASE + WHEEL_STUCK_PWM_BOOST_INCREMENT * (step - WHEEL_STUCK_PWM_BOOST_START_STEP)


def stuck_boost_desc(step):


    desc = f"{stuck_pulse_boost(step):.2f}x"
    if step >= WHEEL_STUCK_PWM_BOOST_START_STEP:
        desc += f", pwm {stuck_pwm_boost(step):.2f}x"
    return desc


def wheels_appear_stuck(telemetry):


    values = [v for v in telemetry.rpm.values() if v is not None]
    if not values:
        return False
    return all(abs(v) < WHEEL_STUCK_RPM_THRESHOLD for v in values)


def wheel_stuck_duration_update(telemetry, last_movement_time, now):


    if wheels_appear_stuck(telemetry):
        if last_movement_time is None:
            return now, 0.0
        return last_movement_time, now - last_movement_time
    return now, 0.0


NAV_BASE_SPEED = 100
NAV_TOF_SLOWDOWN_START_MM = 300.0
NAV_TOF_MIN_BASE_SPEED = 60
NAV_PWM_DELTA_MIN = 10
NAV_PWM_DELTA_MAX = 40


NAV_ANGLE_ERROR_SATURATION_DEG = 12.0
NAV_STEER_DELTA_MIN_DEG = 2.0
NAV_STEER_DELTA_MAX_DEG = 20.0


NAV_STEER_RESEND_DELTA_DEG = 1.0


NAV_RAIL_LOSS_STOP_AFTER_S = 0.35


INTERPLATFORM_ENTRY_MARGIN_PX = 60.0


PLATFORM_CASE_PEEK_FRAMES = 5


PLATFORM_CASE_PEEK_MAX_ERROR_PX = FRAME_W / 4.0


def nav_base_speed_for_tof(tof_side_mean):


    if tof_side_mean is None or tof_side_mean >= NAV_TOF_SLOWDOWN_START_MM:
        return float(NAV_BASE_SPEED)
    if tof_side_mean <= TURNAROUND_TOF_THRESHOLD_MM:
        return float(NAV_TOF_MIN_BASE_SPEED)
    span = NAV_TOF_SLOWDOWN_START_MM - TURNAROUND_TOF_THRESHOLD_MM
    frac = (tof_side_mean - TURNAROUND_TOF_THRESHOLD_MM) / span
    return NAV_TOF_MIN_BASE_SPEED + frac * (NAV_BASE_SPEED - NAV_TOF_MIN_BASE_SPEED)


def new_platform_entry_ok(platform_cx, scan_direction, reference_x):


    if scan_direction == "L":
        return platform_cx <= reference_x - INTERPLATFORM_ENTRY_MARGIN_PX
    return platform_cx >= reference_x + INTERPLATFORM_ENTRY_MARGIN_PX


def front_tof_speed_for_error(error_mag):


    mag = abs(error_mag)
    if mag >= FRONT_TOF_SPEED_RAMP_START_MM:
        return float(FRONT_TOF_MAX_SPEED)
    if mag <= FRONT_TOF_DEADBAND_MM:
        return float(FRONT_TOF_MIN_SPEED)
    span = FRONT_TOF_SPEED_RAMP_START_MM - FRONT_TOF_DEADBAND_MM
    frac = (mag - FRONT_TOF_DEADBAND_MM) / span
    return FRONT_TOF_MIN_SPEED + frac * (FRONT_TOF_MAX_SPEED - FRONT_TOF_MIN_SPEED)


CENTERLINE_COLOR = (255, 255, 255)
CAMERA_REF_COLOR = (0, 0, 255)
TARGET_DOT_COLOR = (0, 0, 255)
MASK_COLOR = np.array([0, 180, 0], dtype=np.float32)


CAMERA_FRONT_ANGLE = 266
CAMERA_LEFT_ANGLE = 179
CAMERA_RIGHT_ANGLE = 358
NO_RAILS_SWITCH_SECONDS = 1.5


CAMERA_HUNT_SETTLE_EXTRA_WAIT_S = 1.0

SERVO_5_INITIAL = 322


SERVO_5_OFFSET_YELLOW_FARTHER = SERVO_5_INITIAL + 1.0
SERVO_5_OFFSET_GREEN_FARTHER_OR_EQUAL = SERVO_5_INITIAL - 1.0


SERVO_8_IDLE_ANGLE = 170
SERVO_9_IDLE_ANGLE = 25
SERVO_9_STRAIGHT_ANGLE = 116.5


SERVO_INITIAL_COMMANDS = [
    ("I", "swerve initial"),
    (f"5 {SERVO_5_INITIAL}", "arm horizontal initial"),
    ("6 50", "arm vertical initial"),
    ("7 51", "front camera initial"),
    (f"8 {SERVO_8_IDLE_ANGLE}", "overhead camera idle"),
    (f"9 {SERVO_9_IDLE_ANGLE}", "servo 9 idle"),
]

cuda.init()
_CUDA_DEVICE = cuda.Device(0)
_CUDA_CONTEXT = _CUDA_DEVICE.retain_primary_context()
_CUDA_CONTEXT.push()


def build_bev_transform(frame_w, frame_h):
    dst = np.zeros_like(SRC_CM)
    dst[:, 0] = SRC_CM[:, 0] * PIXELS_PER_CM + ORIGIN_X_OFFSET
    dst[:, 1] = ORIGIN_Y_OFFSET - SRC_CM[:, 1] * PIXELS_PER_CM
    H, _ = cv2.findHomography(SRC_PTS, dst)
    corners = np.float32([[0, THRESHOLD_Y], [frame_w, THRESHOLD_Y],
                          [frame_w, frame_h], [0, frame_h]]).reshape(-1, 1, 2)
    wc = cv2.perspectiveTransform(corners, H)
    xmin, ymin = np.int32(wc.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(wc.max(axis=0).ravel() + 0.5)
    xmin -= 10
    xmax += 10
    bev_w, bev_h = xmax - xmin, ymax - ymin
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
    return T.dot(H).astype(np.float32), int(bev_w), int(bev_h), xmin, ymin


H_IMG2BEV, BEV_W, BEV_H, BEV_XMIN, BEV_YMIN = build_bev_transform(FRAME_W, FRAME_H)
H_BEV2IMG = np.linalg.inv(H_IMG2BEV)


_BEV_IMG_CENTER_PT = np.array([[[FRAME_W / 2.0, FRAME_H]]], dtype=np.float32)
TRUE_BEV_CENTER_X = cv2.perspectiveTransform(_BEV_IMG_CENTER_PT, H_IMG2BEV)[0, 0, 0]


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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


class TRTSegDetector:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = None
        self.out_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.out_names.append(name)

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))

        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.context.set_tensor_address(self.input_name, int(self.d_input))

        self.out = {}
        for name in self.out_names:
            shp = tuple(self.engine.get_tensor_shape(name))
            h_buf = cuda.pagelocked_empty(shp, dtype=np.float32)
            d_buf = cuda.mem_alloc(h_buf.nbytes)
            self.context.set_tensor_address(name, int(d_buf))
            role = "proto" if len(shp) == 4 else "det"
            self.out[role] = dict(shape=shp, host=h_buf, dev=d_buf)

    def infer(self):
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for o in self.out.values():
            cuda.memcpy_dtoh_async(o["host"], o["dev"], self.stream)
        self.stream.synchronize()
        det = self.out["det"]["host"].reshape(self.out["det"]["shape"])
        proto = self.out["proto"]["host"].reshape(self.out["proto"]["shape"])
        return det, proto


def preprocess_into_buffer(frame, out_buffer, size=IMG_SIZE):
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2

    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[top:top + nh, left:left + nw] = resized

    img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    out_buffer[0] = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return r, (left, top)


def seg_postprocess(det, proto, r, pad, orig_shape):
    H, W = orig_shape[:2]
    p = det[0].T
    conf = p[:, 4:4 + NUM_CLASSES].max(axis=1)
    keep = conf > LANE_CONF_THRESH
    if not np.any(keep):
        return np.zeros((H, W), dtype=np.uint8), []

    p_keep = p[keep]
    conf = conf[keep]
    cx, cy, bw, bh = p_keep[:, 0], p_keep[:, 1], p_keep[:, 2], p_keep[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    coeffs = p_keep[:, 4 + NUM_CLASSES:]

    idx = nms(boxes, conf, LANE_IOU_THRESH)
    if not idx:
        return np.zeros((H, W), dtype=np.uint8), []
    boxes, coeffs = boxes[idx], coeffs[idx]

    protos = proto[0]
    pc, ph, pw = protos.shape
    masks_160 = sigmoid(coeffs @ protos.reshape(pc, ph * pw)).reshape(-1, ph, pw)

    left, top = pad
    binary_union = np.zeros((H, W), dtype=np.uint8)
    bool_masks = []

    for i, m in enumerate(masks_160):
        bx1, by1, bx2, by2 = boxes[i]

        orig_x1 = int(np.clip((bx1 - left) / r, 0, W))
        orig_y1 = int(np.clip((by1 - top) / r, 0, H))
        orig_x2 = int(np.clip((bx2 - left) / r, 0, W))
        orig_y2 = int(np.clip((by2 - top) / r, 0, H))

        target_w = orig_x2 - orig_x1
        target_h = orig_y2 - orig_y1
        if target_w <= 0 or target_h <= 0:
            continue

        mx1 = int(np.clip(bx1 / IMG_SIZE * pw, 0, pw))
        my1 = int(np.clip(by1 / IMG_SIZE * ph, 0, ph))
        mx2 = int(np.clip(bx2 / IMG_SIZE * pw, 0, pw))
        my2 = int(np.clip(by2 / IMG_SIZE * ph, 0, ph))

        mask_slice = m[my1:my2, mx1:mx2]

        if mask_slice.size > 0:
            resized_mask = cv2.resize(mask_slice, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            bm_slice = resized_mask > MASK_THRESH

            bm_full = np.zeros((H, W), dtype=bool)
            bm_full[orig_y1:orig_y2, orig_x1:orig_x2] = bm_slice

            binary_union[orig_y1:orig_y2, orig_x1:orig_x2][bm_slice] = 255
            bool_masks.append(bm_full)

    return binary_union, bool_masks


def get_bev_skeleton_fast(bev_mask):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bev_mask, connectivity=8)

    valid = [(i, centroids[i, 0])
             for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] > BEV_MIN_COMPONENT_AREA]
    valid.sort(key=lambda x: x[1])

    def extract_edge(comp_id, edge_type):
        rows, cols = np.where(labels == comp_id)
        if rows.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        unique_rows = np.unique(rows)
        sort_idx = np.argsort(rows, kind='stable')
        s_rows = rows[sort_idx]
        s_cols = cols[sort_idx]

        boundaries = np.concatenate([[0], np.where(np.diff(s_rows))[0] + 1, [len(s_rows)]])
        edge_points = []
        for k in range(len(unique_rows)):
            seg = s_cols[boundaries[k]:boundaries[k + 1]]
            if seg.size == 0:
                continue
            seg_sorted = np.sort(seg)
            gaps = np.where(np.diff(seg_sorted) > 1)[0] + 1
            runs = np.split(seg_sorted, gaps)
            best = max(runs, key=len)

            if edge_type == 'inner_left':
                x = best[-1]
            elif edge_type == 'inner_right':
                x = best[0]
            else:
                x = int((best[0] + best[-1]) // 2)

            edge_points.append([int(x), int(unique_rows[k])])

        return np.array(edge_points, dtype=np.int32) if edge_points\
            else np.empty((0, 2), dtype=np.int32)

    left_pts = np.empty((0, 2), dtype=np.int32)
    right_pts = np.empty((0, 2), dtype=np.int32)

    if len(valid) >= 2:
        left_pts = extract_edge(valid[0][0], 'inner_left')
        right_pts = extract_edge(valid[-1][0], 'inner_right')
    elif len(valid) == 1:
        comp_id, comp_x = valid[0]
        if comp_x < TRUE_BEV_CENTER_X:
            left_pts = extract_edge(comp_id, 'inner_left')
        else:
            right_pts = extract_edge(comp_id, 'inner_right')

    return left_pts, right_pts


def fit_line_stable(pts):
    if pts is None or len(pts) < 5:
        return None, None
    [vx, vy, x0, y0] = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if abs(vy) < 1e-6:
        return None, None
    m = vx / vy
    b = x0 - m * y0
    return float(m), float(b)


def project_line_safe(m, b, y_start, y_end, H_inv):
    ys = np.linspace(y_start, y_end, 20)
    xs = m * ys + b
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, H_inv).astype(np.int32)


def centerline_angle_and_distance(centerline_pts, image_center_x, mc):


    far_x = float(centerline_pts[0, 0, 0])
    mid_x = float(centerline_pts[len(centerline_pts) // 2, 0, 0])
    near_x = float(centerline_pts[-1, 0, 0])
    signed_dist = ((far_x - image_center_x) + (mid_x - image_center_x) + (near_x - image_center_x)) / 3.0
    dist_px = abs(signed_dist)
    side_label = "RIGHT" if signed_dist > 0 else ("LEFT" if signed_dist < 0 else "CENTER")

    signed_angle_deg = float(np.degrees(np.arctan(mc)))

    return signed_angle_deg, dist_px, side_label


def rotation_for_angle(angle_deg):


    if angle_deg > ANGLE_DEADBAND_DEG:
        return "CCW"
    if angle_deg < -ANGLE_DEADBAND_DEG:
        return "CW"
    return "ALIGNED"


def movement_for_distance(dist_px, side_label):


    if dist_px <= CENTER_DEADBAND_PX:
        return "ALIGNED"
    return "B" if side_label == "LEFT" else "F"


def flip_fb_for_camera_side(direction, camera_side):


    if camera_side != "right":
        return direction
    if direction == "F":
        return "B"
    if direction == "B":
        return "F"
    return direction


def compute_pulse_ms(error_mag, deadband, saturation, min_pulse_ms, max_pulse_ms):


    mag = min(abs(error_mag), saturation)
    if mag <= deadband:
        return 0.0
    span = saturation - deadband
    frac = (mag - deadband) / span
    return min_pulse_ms + frac * (max_pulse_ms - min_pulse_ms)


def draw_dashed_vline(img, x, y_top, y_bottom, color, thickness=2, dash_len=12, gap_len=10):
    x = int(x)
    y = float(y_top)
    while y < y_bottom:
        y_end = min(y + dash_len, y_bottom)
        cv2.line(img, (x, int(y)), (x, int(y_end)), color, thickness)
        y += dash_len + gap_len


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


def platform_servo_gap_px(detections):


    platform = next((d for d in detections if d[0] == PLATFORM_CLASS_ID), None)
    servo = next((d for d in detections if d[0] == SERVO_CLASS_ID), None)
    if platform is None or servo is None:
        return None
    _, _, _, py1, _, _ = platform
    _, _, _, _, _, sy2 = servo
    return abs(py1 - sy2)


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


def ycrop_side_of_frame_center(detections):


    best = None
    for cls_id, conf, x1, y1, x2, y2 in detections:
        if cls_id != YCROP_CLASS_ID:
            continue
        if best is None or conf > best[0]:
            best = (conf, x1, x2)
    if best is None:
        return None
    _, x1, x2 = best
    ycrop_cx = (x1 + x2) / 2.0
    return "L" if ycrop_cx < IMAGE_CENTER_X else "R"


def build_front_tof_steps(case_cmd, ycrop_position, camera_side):


    if case_cmd == "G":
        return []

    if case_cmd == "B":
        servo5_angle = (B_PLACEHOLDER_SERVO5_ANGLE_YCROP_LEFT if ycrop_position == "L"
                         else B_PLACEHOLDER_SERVO5_ANGLE_YCROP_RIGHT)
        return [
            ("1", "placeholder_actions", [
                ("sg90_board1", B_PLACEHOLDER_SG90_ANGLE_1, PLACEHOLDER_SERVO_SETTLE_S),
                ("servo5", servo5_angle, None),
                ("servo6", B_PLACEHOLDER_SERVO6_ANGLE_2, None),
                ("sg90_board1", B_PLACEHOLDER_SG90_ANGLE_2, PLACEHOLDER_SERVO_SETTLE_S),
                ("servo5_async", SERVO_5_INITIAL, None),
                ("servo6_async", SERVO_6_INITIAL, None),
            ]),
        ]


    servo8_rest_angle = CAMERA_RIGHT_ANGLE if camera_side == "right" else CAMERA_LEFT_ANGLE
    return [
        ("1", "align", FRONT_TOF_TARGET_MM),
        ("2", "placeholder_actions", [
            ("servo6_seek", SERVO_6_SEEK_TARGET_GAP_PX, None),


            ("servo_batch", [(8, CAMERA_FRONT_ANGLE), (9, R_PLACEHOLDER_SERVO9_TEMP_ANGLE)], None),
            ("esp2_raw", R_PLACEHOLDER_DISPENSE_CMD, R_PLACEHOLDER_DISPENSE_WAIT_S),
            ("servo6", SERVO_6_INITIAL, None),
            ("verify", None, None),
            ("servo8_async", servo8_rest_angle, None),
            ("servo9_async", SERVO_9_STRAIGHT_ANGLE, None),
        ]),
    ]


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


TASK_IDLE = 0
TASK_LINE_FOLLOW = 1
TASK_FRONT_TOF_ALIGN = 2


TASK_TURNAROUND = 3
TASK_NAMES = {
    TASK_IDLE: "idle (press Enter in the video window to start)",
    TASK_LINE_FOLLOW: "line-following (overhead+bottom, T mode)",
    TASK_FRONT_TOF_ALIGN: "front-camera ToF alignment (no vision, T or I mode)",
    TASK_TURNAROUND: "turnaround (drive to ToF wall, rotate 180, then resume line-following)",
}


NO_VISION_TASKS = (TASK_TURNAROUND, TASK_FRONT_TOF_ALIGN)
ENTER_KEY_CODES = (13, 10)


_SERIAL_LOCK = threading.Lock()


def send(ser, cmd):
    with _SERIAL_LOCK:
        ser.write((cmd + "\n").encode())
        ser.flush()


TOF_RESET_RETRY_DELAY_S = 2.0


def reset_esp1_tofs(ser1):


    attempt = 0
    while True:
        attempt += 1
        log_event(f"[init] Resetting Board 01 ToF sensors (attempt {attempt})...")
        send(ser1, "F")

        deadline = time.time() + 6.0
        ok = total = None
        while time.time() < deadline:
            raw = ser1.readline()
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if not line:
                continue
            if line.startswith("ACK,TOF_RESET,DONE"):
                parts = line.split(",")[-1].split("/")
                if len(parts) == 2:
                    try:
                        ok, total = int(parts[0]), int(parts[1])
                    except ValueError:
                        ok = total = None
                break

        if ok is not None and ok == total:
            log_event(f"[init] Board 01 ToF reset OK — {ok}/{total} sensors reporting. Safe to proceed.")
            return

        seen = f"{ok}/{total}" if ok is not None else "no ACK reply seen"
        log_event(f"[init] Board 01 ToF reset INCOMPLETE ({seen}) — the robot will NOT start until "
                  f"every sensor resets successfully. Retrying in {TOF_RESET_RETRY_DELAY_S:.1f}s...")
        time.sleep(TOF_RESET_RETRY_DELAY_S)


SERVO_SETTLE_TOLERANCE_DEG = 3.0
SERVO_SETTLE_POLL_INTERVAL_S = 0.05
SERVO_SETTLE_STABLE_POLLS = 3
SERVO_SETTLE_TIMEOUT_S = 2.0


DISPENSE_SAFE_WAIT_TIMEOUT_S = 4.5


SERVO_DONE_EVENTS = {servo_id: threading.Event() for servo_id in range(5, 10)}


def angle_diff_deg(a, b):


    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def settle_servo_ids_for_command(cmd):


    if cmd in ("I", "M", "T"):
        return (1, 2, 3, 4), f"swerve mode {cmd}"
    parts = cmd.split()
    if len(parts) == 2:
        try:
            servo_id = int(parts[0])
            float(parts[1])
        except ValueError:
            return None, None
        if 5 <= servo_id <= 9:
            return (servo_id,), f"servo {servo_id} -> {parts[1]}"
    return None, None


def wait_for_swerve_settle(ser, servo_pos, servo_ids, label, timeout_s=SERVO_SETTLE_TIMEOUT_S):


    last_seen = {sid: None for sid in servo_ids}
    stable_count = 0
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for sid in servo_ids:
            servo_pos[sid] = None
        send(ser, "P")
        fresh_deadline = min(deadline, time.time() + SERVO_SETTLE_POLL_INTERVAL_S * 6)
        while time.time() < fresh_deadline and any(servo_pos.get(sid) is None for sid in servo_ids):
            time.sleep(0.01)
        current = {sid: servo_pos.get(sid) for sid in servo_ids}
        if all(v is not None for v in current.values()) and all(
                last_seen[sid] is not None and angle_diff_deg(current[sid], last_seen[sid]) <= SERVO_SETTLE_TOLERANCE_DEG
                for sid in servo_ids):
            stable_count += 1
            if stable_count >= SERVO_SETTLE_STABLE_POLLS:
                log_event(f"[settle] {label} settled {current}.")
                return True
        else:
            stable_count = 0
        last_seen = current
        time.sleep(SERVO_SETTLE_POLL_INTERVAL_S)
    log_event(f"[settle] {label} did NOT settle within {timeout_s:.1f}s "
              f"(last={last_seen}) — proceeding anyway.")
    return False


def wait_for_servos_settle(servo_ids, label, timeout_s=SERVO_SETTLE_TIMEOUT_S):


    deadline = time.time() + timeout_s
    for sid in servo_ids:
        remaining = deadline - time.time()
        if remaining <= 0 or not SERVO_DONE_EVENTS[sid].wait(remaining):
            log_event(f"[settle] {label} did NOT confirm within {timeout_s:.1f}s "
                      f"(servo {sid} never sent DONE) — proceeding anyway.")
            return False
    log_event(f"[settle] {label} confirmed by Board 02 (DONE).")
    return True


def send_and_settle(ser, servo_pos, cmd, timeout_s=SERVO_SETTLE_TIMEOUT_S):


    servo_ids, label = settle_servo_ids_for_command(cmd)
    if cmd in ("I", "M", "T"):
        send(ser, cmd)
        if servo_ids:
            wait_for_swerve_settle(ser, servo_pos, servo_ids, label, timeout_s=timeout_s)
        return
    if servo_ids:
        for sid in servo_ids:
            SERVO_DONE_EVENTS[sid].clear()
    send(ser, cmd)
    if servo_ids:
        wait_for_servos_settle(servo_ids, label, timeout_s=timeout_s)


def parse_pos_line(line):


    result = {}
    for servo_id, part in enumerate(line.split(",")[1:10], start=1):
        try:
            result[servo_id] = float(part)
        except ValueError:
            pass
    return result


def serial_reader(ser, stop_event, servo_pos):


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
        elif line.startswith("POS,"):
            servo_pos.update(parse_pos_line(line))
        elif line.startswith("DONE,"):
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    servo_id = int(parts[1])
                except ValueError:
                    servo_id = None
                if servo_id in SERVO_DONE_EVENTS:
                    SERVO_DONE_EVENTS[servo_id].set()


class ImuFusion:


    def __init__(self):
        self.last_t = None
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw_gyro_integrated = 0.0
        self.acc = [0.0, 0.0, 9.81]
        self.mag = [1.0, 0.0, 0.0]
        self.calibration_samples = []
        self.bias = {"gx": 0.0, "gy": 0.0, "gz": 0.0}
        self.yaw_reference = None

    @staticmethod
    def wrap(angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def update(self, values, now):
        ax, ay, az, gx, gy, gz, mx, my, mz = values
        dt = 0.02 if self.last_t is None else max(0.005, min(0.15, now - self.last_t))
        self.last_t = now

        alpha = 0.16
        self.acc = [
            self.acc[0] * (1 - alpha) + ax * alpha,
            self.acc[1] * (1 - alpha) + ay * alpha,
            self.acc[2] * (1 - alpha) + az * alpha,
        ]
        self.mag = [
            self.mag[0] * (1 - alpha) + mx * alpha,
            self.mag[1] * (1 - alpha) + my * alpha,
            self.mag[2] * (1 - alpha) + mz * alpha,
        ]

        mag_yaw = self.wrap(math.degrees(math.atan2(self.mag[1], self.mag[0])))
        if (
            len(self.calibration_samples) < IMU_CALIBRATION_MIN_SAMPLES
            and abs(gx) < IMU_STILL_GYRO_MAX_DPS
            and abs(gy) < IMU_STILL_GYRO_MAX_DPS
            and abs(gz) < IMU_STILL_GYRO_MAX_DPS
        ):
            self.calibration_samples.append({"gx": gx, "gy": gy, "gz": gz, "yaw": mag_yaw})
            for axis in ("gx", "gy", "gz"):
                self.bias[axis] = statistics.fmean(sample[axis] for sample in self.calibration_samples)
            self.yaw_reference = self.circular_mean(sample["yaw"] for sample in self.calibration_samples)

        gx_corrected = gx - self.bias["gx"]
        gy_corrected = gy - self.bias["gy"]
        gz_corrected = gz - self.bias["gz"]

        ax, ay, az = self.acc
        acc_roll = math.degrees(math.atan2(ay, az))
        acc_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))

        comp = 0.96
        self.roll = comp * (self.roll + gx_corrected * dt) + (1 - comp) * acc_roll
        self.pitch = comp * (self.pitch + gy_corrected * dt) + (1 - comp) * acc_pitch
        self.yaw_gyro_integrated = self.wrap(self.yaw_gyro_integrated + gz_corrected * dt)

        return {"yaw": self.yaw_gyro_integrated, "roll": self.roll, "pitch": self.pitch}

    @classmethod
    def circular_mean(cls, angles):
        angles = list(angles)
        if not angles:
            return 0.0
        sin_sum = sum(math.sin(math.radians(angle)) for angle in angles)
        cos_sum = sum(math.cos(math.radians(angle)) for angle in angles)
        return cls.wrap(math.degrees(math.atan2(sin_sum, cos_sum)))


class Telemetry:


    def __init__(self):
        self.tof = {label: None for label in TOF_LABELS}
        self.heading = None


        self.rpm = {"BR": None, "FR": None, "BL": None, "FL": None}
        self._imu_fusion = ImuFusion()


        self.tof_active = False

    def update_imu(self, values, now):
        self.heading = self._imu_fusion.update(values, now)["yaw"]


def esp1_reader(ser, stop_event, telemetry):


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
        if line.startswith("TOF,["):
            if not telemetry.tof_active:
                continue
            body = line[line.find("[") + 1:line.rfind("]")]
            for item in body.split(","):
                if ":" not in item:
                    continue
                label, value = item.split(":", 1)
                label = label.strip()
                if label in telemetry.tof:
                    try:
                        telemetry.tof[label] = float(value.strip())
                    except ValueError:
                        pass
        elif line.startswith("IMU,"):
            parts = line.split(",")
            if len(parts) == 10 and parts[1] != "ERR":
                try:
                    values = [float(v) for v in parts[1:]]
                except ValueError:
                    continue
                telemetry.update_imu(values, time.time())
        elif line.startswith(("RPM,", "RPM [")):
            for label, value in RPM_DETAIL_RE.findall(line):
                try:
                    telemetry.rpm[label] = float(value)
                except ValueError:
                    pass


def send_all_servos_initial(ser2, servo_pos):
    log_event("[init] Sending all servos to initial position...")
    for command, label in SERVO_INITIAL_COMMANDS:
        log_event(f"  >>> {command}  ({label})")
        send_and_settle(ser2, servo_pos, command)


def terminal_listener(ser, stop_event):


    pattern = re.compile(r"^([567])\s+(\d+(?:\.\d+)?)$")
    while not stop_event.is_set():
        try:
            raw = input().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        match = pattern.match(raw)
        if not match:
            log_event(f"  ? unrecognized servo command '{raw}' (expected '5 <angle>', '6 <angle>', or '7 <angle>')")
            continue
        servo_id, angle = match.group(1), match.group(2)
        send(ser, f"{servo_id} {angle}")
        log_event(f"  >>> servo {servo_id} -> {angle}")


class LaneAlignmentProcessor:
    def __init__(self, engine_path, image_center_x, warmup_iters=5):
        self.image_center_x = image_center_x

        log_event(f"Loading lane TRT Engine: {engine_path}")
        self.detector = TRTSegDetector(engine_path)

        self.last_known_rail_width = DEFAULT_RAIL_WIDTH_BEV
        self.last_known_slope_diff = DEFAULT_SLOPE_DIFF

        self.smoothed_m1 = self.smoothed_b1 = None
        self.smoothed_m2 = self.smoothed_b2 = None
        self._was_tracking = False

        self.fps_s = 0.0
        self._t_prev = None

        self._warmup(warmup_iters)
        log_event("Lane processor ready.")

    def _warmup(self, iters):
        dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(max(1, iters)):
            preprocess_into_buffer(dummy, self.detector.h_input)
            self.detector.infer()
        log_event(f"Lane GPU warm-up done ({iters} iters)")

    def process_frame(self, frame):
        t_now = time.time()
        if self._t_prev:
            dt = t_now - self._t_prev
            if dt > 1e-6:
                self.fps_s = 0.2 / dt + 0.8 * self.fps_s if self.fps_s else 1.0 / dt
        self._t_prev = t_now

        if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        h, w = frame.shape[:2]
        img_cx = w // 2

        r, pad = preprocess_into_buffer(frame, self.detector.h_input)
        det, proto = self.detector.infer()
        binary_mask, bool_masks = seg_postprocess(det, proto, r, pad, frame.shape)

        bev_mask = cv2.warpPerspective(binary_mask, H_IMG2BEV, (BEV_W, BEV_H))
        bev_mask = cv2.morphologyEx(bev_mask, cv2.MORPH_CLOSE, BEV_MASK_CLEAN_KERNEL)
        bev_mask = cv2.morphologyEx(bev_mask, cv2.MORPH_OPEN, BEV_MASK_CLEAN_KERNEL)
        left_bev_pts, right_bev_pts = get_bev_skeleton_fast(bev_mask)

        m1, b1 = fit_line_stable(left_bev_pts)
        m2, b2 = fit_line_stable(right_bev_pts)

        if m1 is not None and m2 is not None:
            current_width = b2 - b1
            current_slope_diff = m2 - m1
            if abs(current_width) > 50:
                self.last_known_rail_width = (WIDTH_EMA_ALPHA * current_width +
                    (1 - WIDTH_EMA_ALPHA) * self.last_known_rail_width)
                self.last_known_slope_diff = (WIDTH_EMA_ALPHA * current_slope_diff +
                    (1 - WIDTH_EMA_ALPHA) * self.last_known_slope_diff)
        elif m1 is not None:
            m2 = m1 + self.last_known_slope_diff
            b2 = b1 + self.last_known_rail_width
        elif m2 is not None:
            m1 = m2 - self.last_known_slope_diff
            b1 = b2 - self.last_known_rail_width

        have_rails = (m1 is not None and m2 is not None)

        if have_rails:
            if not self._was_tracking:
                self.smoothed_m1, self.smoothed_b1 = m1, b1
                self.smoothed_m2, self.smoothed_b2 = m2, b2
            else:
                a = LINE_SMOOTHING_ALPHA
                self.smoothed_m1 = a * m1 + (1 - a) * self.smoothed_m1
                self.smoothed_b1 = a * b1 + (1 - a) * self.smoothed_b1
                self.smoothed_m2 = a * m2 + (1 - a) * self.smoothed_m2
                self.smoothed_b2 = a * b2 + (1 - a) * self.smoothed_b2
            m1, b1, m2, b2 = self.smoothed_m1, self.smoothed_b1, self.smoothed_m2, self.smoothed_b2
        self._was_tracking = have_rails
        bottom_cx = img_cx
        mc = bc = None

        if have_rails:
            mc, bc = (m1 + m2) / 2.0, (b1 + b2) / 2.0
            y_near_bev = float(cv2.perspectiveTransform(
                np.array([[[img_cx, FRAME_H]]], np.float32), H_IMG2BEV)[0, 0, 1])
            cl_bev = np.array([[[mc * y_near_bev + bc, y_near_bev]]], np.float32)
            bottom_cx = float(cv2.perspectiveTransform(cl_bev, H_BEV2IMG)[0, 0, 0])

        disp = frame.copy()
        for m in bool_masks:
            disp[m] = (0.5 * MASK_COLOR + 0.5 * disp[m]).astype(np.uint8)

        draw_dashed_vline(disp, self.image_center_x, THRESHOLD_Y, FRAME_H, CAMERA_REF_COLOR)

        angle_deg, dist_px, side_label = 0.0, 0.0, "N/A"
        if have_rails:
            limit_img = np.array([[[img_cx, THRESHOLD_Y]], [[img_cx, FRAME_H]]], dtype=np.float32)
            limit_bev = cv2.perspectiveTransform(limit_img, H_IMG2BEV)
            y_far, y_near = limit_bev[0][0][1], limit_bev[1][0][1]

            centerline_pts = project_line_safe(mc, bc, y_far, y_near, H_BEV2IMG)
            angle_deg, dist_px, side_label = centerline_angle_and_distance(
                centerline_pts, self.image_center_x, mc)

            cv2.polylines(disp, [centerline_pts], False, CENTERLINE_COLOR, 3)
            cv2.circle(disp, (int(bottom_cx), FRAME_H - 5), 6, TARGET_DOT_COLOR, -1)

        cv2.putText(disp, f"ANGLE:{angle_deg:+.1f}deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(disp, f"DIST:{dist_px:.1f}px ({side_label})",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(disp, f"FPS:{self.fps_s:.1f}",
                    (disp.shape[1] - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return disp, have_rails, angle_deg, dist_px, side_label


class PlatformProcessor:
    def __init__(self, engine_path, image_center_x):
        self.image_center_x = image_center_x

        log_event(f"Loading platform TRT Engine: {engine_path}")
        self.detector = TRTDetector(engine_path)

        self.smoothed_cx = None
        self.fps_s = 0.0
        self._t_prev = None
        log_event("Platform processor ready.")

    def process_frame(self, frame):
        t_now = time.time()
        if self._t_prev:
            dt = t_now - self._t_prev
            if dt > 1e-6:
                self.fps_s = 0.2 / dt + 0.8 * self.fps_s if self.fps_s else 1.0 / dt
        self._t_prev = t_now

        inp, r, pad = preprocess(frame)
        raw = self.detector.infer(inp).reshape(self.detector.output_shape)
        detections = postprocess(raw, r, pad, frame.shape)
        any_detected = len(detections) > 0

        platform_info = compute_platform_error(detections, frame.shape[1])
        if platform_info is None:
            self.smoothed_cx = None
            error_px = None
        else:
            _, raw_cx = platform_info
            self.smoothed_cx = (raw_cx if self.smoothed_cx is None
                                 else PLATFORM_EMA_ALPHA * raw_cx + (1 - PLATFORM_EMA_ALPHA) * self.smoothed_cx)
            error_px = self.smoothed_cx - self.image_center_x

        disp = frame.copy()
        draw_detections(disp, detections)
        cv2.putText(disp, f"FPS:{self.fps_s:.1f}",
                    (disp.shape[1] - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if error_px is not None:
            cv2.putText(disp, f"ERR:{error_px:+.1f}px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        return disp, any_detected, error_px, detections


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
    log_event(f"Connecting to {ESP1_PORT} and {ESP2_PORT} at {BAUD} baud...")
    ser1 = serial.Serial(ESP1_PORT, BAUD, timeout=0.1)
    ser2 = serial.Serial(ESP2_PORT, BAUD, timeout=0.1)
    time.sleep(0.5)
    ser1.reset_input_buffer()
    ser2.reset_input_buffer()
    reset_esp1_tofs(ser1)


    send(ser1, "T")
    send(ser1, "T0")

    stop_event = threading.Event()
    telemetry = Telemetry()
    servo_pos = {}
    threading.Thread(target=esp1_reader, args=(ser1, stop_event, telemetry), daemon=True).start()
    threading.Thread(target=serial_reader, args=(ser2, stop_event, servo_pos), daemon=True).start()
    threading.Thread(target=terminal_listener, args=(ser2, stop_event), daemon=True).start()

    send_all_servos_initial(ser2, servo_pos)

    lane_processor = None
    platform_processor = None
    ball_verify_processor = None
    cap_overhead = None
    cap_bottom = None
    try:
        lane_processor = LaneAlignmentProcessor(LANE_ENGINE_PATH, IMAGE_CENTER_X)
        platform_processor = PlatformProcessor(PLATFORM_ENGINE_PATH, IMAGE_CENTER_X)
        ball_verify_processor = BallVerifyProcessor(BALL_VERIFY_ENGINE_PATH)

        cap_overhead = open_camera("overhead", OVERHEAD_CAMERA_DEVICE)
        cap_bottom = open_camera("bottom", BOTTOM_CAMERA_DEVICE)


        cap_bottom.pause()


        scan_direction = SWERVE_T_SCAN_DIRECTION


        scan_direction_locked = False

        task = TASK_IDLE
        previous_task = TASK_IDLE


        servo_ready_sent = False


        camera_side = "left"
        no_rails_since = None


        consecutive_read_failures_overhead = 0
        consecutive_read_failures_bottom = 0


        line_follow_phase = "following"


        old_platform_reference_x = IMAGE_CENTER_X


        last_steer_direction = None
        last_steer_delta_deg = 0.0


        rail_loss_stop_sent = False


        task_aligned = False
        task_pending_recheck = False

        lane_have_rails, angle_deg, dist_px, side_label = False, 0.0, 0.0, "N/A"
        platform_any_detected, platform_error_px, platform_detections = False, None, []


        case_sent = False
        case_label, case_color = None, None


        servo5_offset_sent = False


        case_peek_cmd = None
        case_peek_streak = 0


        platform_last_movement_time = None
        platform_stuck_step = 0
        front_tof_last_movement_time = None
        front_tof_stuck_step = 0


        gbr_count = 0


        turnaround_phase = "rotating"
        turnaround_target_heading = None


        last_case_cmd = None


        last_ycrop_position = None
        front_tof_steps = []
        front_tof_step_index = 0

        front_tof_phase = "waiting_enter"
        front_tof_target = None
        front_tof_pending_recheck = False


        front_tof_action_list = []
        front_tof_action_index = 0
        front_tof_action_start = 0.0


        seek_servo6_angle = None


        verify_subphase = "checking"
        verify_action_start = 0.0
        verify_streak = 0
        verify_check_start = 0.0

        def enter_front_tof_step():


            nonlocal front_tof_phase, front_tof_target, front_tof_pending_recheck
            nonlocal front_tof_action_list, front_tof_action_index
            nonlocal front_tof_last_movement_time, front_tof_stuck_step
            if front_tof_step_index < len(front_tof_steps):
                step_label, kind, value = front_tof_steps[front_tof_step_index]
                front_tof_phase = kind
                if kind == "align":
                    front_tof_target = value
                    front_tof_pending_recheck = False
                    front_tof_last_movement_time = None
                    front_tof_stuck_step = 0
                    log_event(f"[task] {step_label}: aligning front ToF (CAM) to {value:.0f}mm.")
                else:
                    front_tof_action_list = value
                    front_tof_action_index = 0
                    log_event(f"[task] {step_label}: starting placeholder sequence "
                              f"({len(front_tof_action_list)} action(s)).")
                    enter_front_tof_action()
            else:
                front_tof_phase = "waiting_enter"
                send(ser2, "X")
                log_event("[task] Front ToF sequence complete. Press Enter to resume line-following.")

        def enter_front_tof_action():


            nonlocal front_tof_action_start, front_tof_step_index, front_tof_action_index, seek_servo6_angle
            nonlocal verify_subphase, verify_streak, verify_check_start
            if front_tof_action_index >= len(front_tof_action_list):
                front_tof_step_index += 1
                enter_front_tof_step()
                return
            kind, value, hold_s = front_tof_action_list[front_tof_action_index]
            step_label = front_tof_steps[front_tof_step_index][0]
            if kind in ("servo5", "servo6", "servo8", "servo9"):


                log_event(f"[task] {step_label} action {front_tof_action_index + 1}/"
                          f"{len(front_tof_action_list)}: {kind}={value} (waiting for arrival)")
                timeout_s = DISPENSE_SAFE_WAIT_TIMEOUT_S if kind == "servo6" else SERVO_SETTLE_TIMEOUT_S
                send_and_settle(ser2, servo_pos, f"{kind[-1]} {value}", timeout_s=timeout_s)
                front_tof_action_index += 1
                enter_front_tof_action()
                return
            if kind in ("servo5_async", "servo6_async", "servo8_async", "servo9_async"):


                servo_id = kind.split("_")[0][-1]
                log_event(f"[task] {step_label} action {front_tof_action_index + 1}/"
                          f"{len(front_tof_action_list)}: {kind}={value} (fire-and-forget)")
                send(ser2, f"{servo_id} {value}")
                front_tof_action_index += 1
                enter_front_tof_action()
                return
            if kind == "servo_batch":


                servo_ids = tuple(sid for sid, _angle in value)
                log_event(f"[task] {step_label} action {front_tof_action_index + 1}/"
                          f"{len(front_tof_action_list)}: batch {value} (waiting for arrival)")
                for sid in servo_ids:
                    SERVO_DONE_EVENTS[sid].clear()
                for sid, angle in value:
                    send(ser2, f"{sid} {angle}")
                wait_for_servos_settle(servo_ids, f"batch {value}")
                front_tof_action_index += 1
                enter_front_tof_action()
                return
            if kind == "sg90_board1":
                send(ser1, f"SG90,{value}")
            elif kind == "esp2_raw":
                send(ser2, value)
            elif kind == "servo6_seek":


                seek_servo6_angle = SERVO_6_SEEK_START_ANGLE
                send(ser2, f"6 {seek_servo6_angle}")
                cap_bottom.resume()
            elif kind == "verify":


                time.sleep(BALL_VERIFY_SETTLE_S)
                verify_subphase = "checking"
                verify_streak = 0
                verify_check_start = time.time()
                cap_overhead.resume()


            front_tof_action_start = time.time()
            log_event(f"[task] {step_label} action {front_tof_action_index + 1}/"
                      f"{len(front_tof_action_list)}: {kind}"
                      f"{f'={value}' if value is not None else ''}"
                      f"{f' (hold {hold_s:.1f}s)' if hold_s is not None else ' (vision seek)'}")

        log_event("=" * 60)
        log_event(f"RUN START {datetime.now().isoformat(timespec='seconds')}")
        log_event(" LANE + PLATFORM ALIGNMENT TEST")
        log_event(" Click the video window; press Enter there to start line-following.")
        log_event(" Platform stops, dispense/topple, and resuming are all automatic.")
        log_event(" Enter also restarts line-following after a turnaround completes.")
        log_event(" 'q' in the video window or Ctrl+C to quit.")
        log_event("=" * 60)

        with suppress_stderr():
            while True:


                skip_control_this_frame = False


                if task != previous_task:
                    task_aligned = False
                    task_pending_recheck = False
                    skip_control_this_frame = True
                    if task == TASK_LINE_FOLLOW:


                        send(ser2, "X")
                        cap_overhead.resume()
                        cap_bottom.resume()
                        send_and_settle(ser2, servo_pos, "T")


                        angle = CAMERA_RIGHT_ANGLE if camera_side == "right" else CAMERA_LEFT_ANGLE
                        send_and_settle(ser2, servo_pos, f"8 {angle}")
                        no_rails_since = None


                        last_steer_direction = None
                        last_steer_delta_deg = 0.0
                        rail_loss_stop_sent = False
                        platform_processor.smoothed_cx = None
                        case_peek_cmd, case_peek_streak = None, 0


                        telemetry.tof_active = True


                        log_event(f"[task] Line-following: servo 8 -> {camera_side} ({angle}), "
                                  f"scan_direction={scan_direction}, phase={line_follow_phase}.")
                    elif task == TASK_FRONT_TOF_ALIGN:
                        send(ser2, "X")


                        if last_case_cmd == "R":
                            send_and_settle(ser2, servo_pos, "I")


                        cap_overhead.pause()
                        cap_bottom.pause()
                        front_tof_step_index = 0
                        front_tof_steps = build_front_tof_steps(last_case_cmd, last_ycrop_position, camera_side)
                        enter_front_tof_step()
                    elif task == TASK_TURNAROUND:


                        cap_overhead.pause()
                        cap_bottom.pause()
                        log_event(f"[task] Turnaround: rotating {TURNAROUND_TARGET_DEG:.0f} deg.")
                    previous_task = task

                if task in NO_VISION_TASKS:


                    overlay = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
                    if task == TASK_TURNAROUND:
                        label = {"rotating": "ROTATING", "waiting_enter": "DONE"}.get(turnaround_phase, "-")
                        cv2.putText(overlay, f"GBR:{gbr_count}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 0), 2)
                        side_a, side_b = TURNAROUND_SIDE_TOF_LABELS[scan_direction]
                        vals = {lbl: telemetry.tof.get(lbl) for lbl in (side_a, side_b)}
                        tof_text = " ".join(
                            f"{k}:{'--' if v is None else f'{v:.0f}'}" for k, v in vals.items())
                        cv2.putText(overlay, f"TOF {tof_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 0, 255), 2)
                        heading_text = "--" if telemetry.heading is None else f"{telemetry.heading:+.1f}"
                        cv2.putText(overlay, f"HEADING:{heading_text}", (10, 135),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    else:
                        label = {"align": "ALIGNING", "placeholder_actions": "ACTIONS",
                                 "waiting_enter": "DONE"}.get(front_tof_phase, "-")
                        step_label = (front_tof_steps[front_tof_step_index][0]
                                      if front_tof_step_index < len(front_tof_steps) else "-")
                        cv2.putText(overlay, f"STEP {step_label}  (case={last_case_cmd or '?'})",
                                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cam_reading = telemetry.tof.get(FRONT_TOF_LABEL)
                        cam_text = "--" if cam_reading is None else f"{cam_reading:.0f}mm"
                        target_text = "--" if front_tof_target is None else f"{front_tof_target:.0f}mm"
                        cv2.putText(overlay, f"CAM:{cam_text} TARGET:{target_text}", (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        if front_tof_phase == "placeholder_actions":
                            action_text = f"action {front_tof_action_index + 1}/{len(front_tof_action_list)}"
                            cv2.putText(overlay, action_text, (10, 160),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                elif task == TASK_IDLE:


                    ret_o, frame_o = cap_overhead.read()
                    if not ret_o:
                        consecutive_read_failures_overhead += 1
                        if consecutive_read_failures_overhead >= MAX_CONSECUTIVE_READ_FAILURES:
                            log_event(f"\n[warn] overhead frame grab failed "
                                      f"{consecutive_read_failures_overhead}x in a row, stopping.")
                            break
                        log_event(f"[warn] overhead frame grab timed out "
                                  f"({consecutive_read_failures_overhead}/{MAX_CONSECUTIVE_READ_FAILURES}), retrying...")
                        continue
                    consecutive_read_failures_overhead = 0
                    overlay = frame_o
                    lane_have_rails = False
                    label = "WAITING (press Enter)"

                elif task == TASK_LINE_FOLLOW and line_follow_phase == "following":


                    ret_o, frame_o = cap_overhead.read()
                    if not ret_o:
                        consecutive_read_failures_overhead += 1
                        if consecutive_read_failures_overhead >= MAX_CONSECUTIVE_READ_FAILURES:
                            log_event(f"\n[warn] overhead frame grab failed "
                                      f"{consecutive_read_failures_overhead}x in a row, stopping.")
                            break
                        log_event(f"[warn] overhead frame grab timed out "
                                  f"({consecutive_read_failures_overhead}/{MAX_CONSECUTIVE_READ_FAILURES}), retrying...")
                        continue
                    consecutive_read_failures_overhead = 0

                    ret_b, frame_b = cap_bottom.read()
                    if not ret_b:
                        consecutive_read_failures_bottom += 1
                        if consecutive_read_failures_bottom >= MAX_CONSECUTIVE_READ_FAILURES:
                            log_event(f"\n[warn] bottom frame grab failed "
                                      f"{consecutive_read_failures_bottom}x in a row, stopping.")
                            break
                        log_event(f"[warn] bottom frame grab timed out "
                                  f"({consecutive_read_failures_bottom}/{MAX_CONSECUTIVE_READ_FAILURES}), retrying...")
                        continue
                    consecutive_read_failures_bottom = 0

                    overlay, lane_have_rails, angle_deg, dist_px, side_label =\
                        lane_processor.process_frame(frame_o)
                    _, platform_any_detected, platform_error_px, platform_detections =\
                        platform_processor.process_frame(frame_b)
                    label = "FOLLOWING" if lane_have_rails else "NO RAILS"
                    if case_sent:
                        cv2.circle(overlay, (20, 20), 10, case_color, -1)
                        cv2.putText(overlay, case_label, (38, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, case_color, 2)
                    cv2.putText(overlay, f"GBR:{gbr_count}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 0), 2)
                    if platform_any_detected and platform_error_px is not None:
                        accepted = (old_platform_reference_x is not None and new_platform_entry_ok(
                            platform_error_px + IMAGE_CENTER_X, scan_direction, old_platform_reference_x))
                        cv2.putText(overlay, "PLATFORM DETECTED" + (" (accepted)" if accepted else ""),
                                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                else:
                    ret, frame = cap_bottom.read()
                    if not ret:


                        consecutive_read_failures_bottom += 1
                        if consecutive_read_failures_bottom >= MAX_CONSECUTIVE_READ_FAILURES:
                            log_event(f"\n[warn] bottom frame grab failed "
                                      f"{consecutive_read_failures_bottom}x in a row, stopping.")
                            break
                        log_event(f"[warn] bottom frame grab timed out "
                                  f"({consecutive_read_failures_bottom}/{MAX_CONSECUTIVE_READ_FAILURES}), retrying...")
                        continue
                    consecutive_read_failures_bottom = 0

                    overlay, platform_any_detected, platform_error_px, platform_detections =\
                        platform_processor.process_frame(frame)
                    if line_follow_phase == "waiting_enter":
                        label = "WAITING (press Enter)"
                    elif platform_error_px is None:
                        label = "NO TARGET"
                    elif abs(platform_error_px) <= PLATFORM_ERROR_DEADBAND_PX:
                        label = "ALIGNED"
                    else:
                        label = "R" if platform_error_px > 0 else "L"
                    if case_sent:
                        cv2.circle(overlay, (20, 20), 10, case_color, -1)
                        cv2.putText(overlay, case_label, (38, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, case_color, 2)
                    cv2.putText(overlay, f"GBR:{gbr_count}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 0), 2)

                label_color = (0, 255, 0) if label in ("ALIGNED", "FOLLOWING", "DONE") else (
                    (0, 0, 255) if label in ("NO RAILS", "NO TARGET") else (0, 165, 255))
                cv2.putText(overlay, f"ALIGN: {label}", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
                cv2.putText(overlay, f"TASK: {TASK_NAMES[task]}", (10, FRAME_H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow("LANE + PLATFORM ALIGNMENT TEST", overlay)


                now = time.time()
                if (servo_ready_sent and task == TASK_LINE_FOLLOW
                        and line_follow_phase == "following"):
                    if lane_have_rails:
                        no_rails_since = None
                        rail_loss_stop_sent = False
                        if not scan_direction_locked:


                            scan_direction = "R" if camera_side == "right" else "L"
                            scan_direction_locked = True
                            log_event(f"[init] Overhead camera found the lane on the "
                                      f"{camera_side} side -> scan_direction={scan_direction}.")
                    else:
                        if no_rails_since is None:
                            no_rails_since = now
                        elif now - no_rails_since >= NO_RAILS_SWITCH_SECONDS:


                            camera_side = "right" if camera_side == "left" else "left"
                            angle = CAMERA_RIGHT_ANGLE if camera_side == "right" else CAMERA_LEFT_ANGLE
                            send(ser2, "X")
                            send_and_settle(ser2, servo_pos, f"8 {angle}")
                            time.sleep(CAMERA_HUNT_SETTLE_EXTRA_WAIT_S)
                            last_steer_direction = None
                            last_steer_delta_deg = 0.0
                            no_rails_since = None
                            rail_loss_stop_sent = False
                            log_event(f"[hunt] Rails lost -> servo 8 -> {angle} ({camera_side}).")
                            continue


                if skip_control_this_frame:
                    pass

                elif task == TASK_LINE_FOLLOW and line_follow_phase == "following":


                    side_a, side_b = TURNAROUND_SIDE_TOF_LABELS[scan_direction]
                    tof_a, tof_b = telemetry.tof.get(side_a), telemetry.tof.get(side_b)
                    tof_side_mean = (tof_a + tof_b) / 2.0 if tof_a is not None and tof_b is not None else None


                    platform_entry_accepted = False
                    if platform_any_detected and platform_error_px is not None:
                        platform_cx = platform_error_px + IMAGE_CENTER_X
                        if old_platform_reference_x is None:


                            old_platform_reference_x = platform_cx
                            log_event(f"[task] First platform position after resuming "
                                      f"line-following: {platform_cx:.0f}px — using as reference.")
                        elif new_platform_entry_ok(platform_cx, scan_direction, old_platform_reference_x):
                            platform_entry_accepted = True

                    if tof_side_mean is not None and tof_side_mean <= TURNAROUND_TOF_THRESHOLD_MM:
                        send(ser2, "X")
                        send_and_settle(ser2, servo_pos, "M")
                        task = TASK_TURNAROUND
                        turnaround_phase = "rotating"
                        turnaround_target_heading = None


                        camera_side = "right" if camera_side == "left" else "left"
                        turnaround_camera_angle = CAMERA_RIGHT_ANGLE if camera_side == "right" else CAMERA_LEFT_ANGLE
                        send(ser2, f"8 {turnaround_camera_angle}")
                        log_event(f"[task] ToF tripped during line-follow "
                                  f"({side_a}={tof_a} {side_b}={tof_b}, under "
                                  f"{TURNAROUND_TOF_THRESHOLD_MM:.0f}mm) — rotating "
                                  f"{TURNAROUND_TARGET_DEG:.0f} deg; servo 8 -> "
                                  f"{turnaround_camera_angle} ({camera_side}) fired now, not awaited.")

                    elif platform_entry_accepted:
                        send(ser2, "X")
                        line_follow_phase = "platform_align"
                        cap_overhead.pause()
                        task_aligned = False
                        task_pending_recheck = False
                        case_sent = False
                        case_label, case_color = None, None
                        servo5_offset_sent = False
                        platform_last_movement_time = None
                        platform_stuck_step = 0
                        case_peek_cmd = None
                        case_peek_streak = 0
                        log_event("[task] New platform confirmed entering from the expected side — "
                                  "switching to platform alignment.")

                    elif not lane_have_rails:


                        if (no_rails_since is not None
                                and now - no_rails_since >= NAV_RAIL_LOSS_STOP_AFTER_S
                                and not rail_loss_stop_sent):
                            send(ser2, "X")
                            rail_loss_stop_sent = True

                    else:


                        dist_dir = flip_fb_for_camera_side(
                            movement_for_distance(dist_px, side_label), camera_side)
                        if dist_dir == "ALIGNED":
                            new_steer_direction, new_steer_delta = None, 0.0
                        else:
                            new_steer_direction = dist_dir
                            new_steer_delta = compute_pulse_ms(
                                dist_px, CENTER_DEADBAND_PX, CENTER_ERROR_SATURATION_PX,
                                NAV_STEER_DELTA_MIN_DEG, NAV_STEER_DELTA_MAX_DEG)
                        if (new_steer_direction != last_steer_direction
                                or abs(new_steer_delta - last_steer_delta_deg) >= NAV_STEER_RESEND_DELTA_DEG):
                            send(ser2, "DF0" if new_steer_direction is None
                                 else f"D{new_steer_direction}{new_steer_delta:.1f}")
                            last_steer_direction = new_steer_direction
                            last_steer_delta_deg = new_steer_delta


                        rotation = rotation_for_angle(angle_deg)
                        pwm_delta = compute_pulse_ms(
                            angle_deg, ANGLE_DEADBAND_DEG, NAV_ANGLE_ERROR_SATURATION_DEG,
                            NAV_PWM_DELTA_MIN, NAV_PWM_DELTA_MAX)
                        base = nav_base_speed_for_tof(tof_side_mean)
                        if rotation == "CW":
                            br, fr, bl, fl = base + pwm_delta, base, base + pwm_delta, base
                        elif rotation == "CCW":
                            br, fr, bl, fl = base, base + pwm_delta, base, base + pwm_delta
                        else:
                            br, fr, bl, fl = base, base, base, base
                        send(ser2, f"{scan_direction}{br:.0f},{fr:.0f},{bl:.0f},{fl:.0f}")

                elif task == TASK_LINE_FOLLOW and line_follow_phase == "platform_align":


                    peek_cmd = None
                    if (platform_error_px is not None
                            and abs(platform_error_px) <= PLATFORM_CASE_PEEK_MAX_ERROR_PX):
                        peek_case = classify_case(platform_detections)
                        peek_cmd = peek_case[2] if peek_case is not None else None
                    if peek_cmd is not None and peek_cmd == case_peek_cmd:
                        case_peek_streak += 1
                    else:
                        case_peek_cmd = peek_cmd
                        case_peek_streak = 1 if peek_cmd is not None else 0
                    if case_peek_streak >= PLATFORM_CASE_PEEK_FRAMES and case_peek_cmd == "G":
                        send(ser2, "X")
                        send(ser1, "G")
                        send(ser2, "G")
                        last_case_cmd = "G"
                        gbr_count += 1
                        current_platform_cx = (platform_error_px + IMAGE_CENTER_X
                                                if platform_error_px is not None else IMAGE_CENTER_X)
                        old_platform_reference_x = current_platform_cx
                        case_peek_cmd, case_peek_streak = None, 0
                        line_follow_phase = "following"
                        cap_overhead.resume()
                        last_steer_direction = None
                        last_steer_delta_deg = 0.0
                        skip_control_this_frame = True
                        log_event(f"[task] G-case fast path: {PLATFORM_CASE_PEEK_FRAMES} consistent reads "
                                  f"during alignment -> sent 'G' to both ESPs, aborting align early "
                                  f"(GBR:{gbr_count}); resuming line-following.")


                    elif platform_error_px is None:


                        send(ser2, f"{scan_direction}{PLATFORM_REACQUIRE_PWM}")
                        time.sleep(PLATFORM_REACQUIRE_PULSE_MS / 1000.0)
                        send(ser2, "X")
                        task_pending_recheck = False
                    else:


                        align_deadband_px = (
                            PLATFORM_ERROR_DEADBAND_R_PX
                            if case_peek_streak >= PLATFORM_CASE_PEEK_FRAMES and case_peek_cmd == "R"
                            else PLATFORM_ERROR_DEADBAND_PX)
                        pulse_ms = compute_pulse_ms(
                            platform_error_px, align_deadband_px, PLATFORM_ERROR_SATURATION_PX,
                            PLATFORM_MIN_PULSE_MS, PLATFORM_MAX_PULSE_MS)
                        if pulse_ms <= 0:
                            send(ser2, "X")
                            if not ALIGN_RECHECK_ENABLED or task_pending_recheck:
                                task_aligned = True
                                log_event(f"[task] Platform aligned (err={platform_error_px:+.1f}px, "
                                          f"locked, deadband={align_deadband_px:.0f}px).")
                            else:
                                task_pending_recheck = True
                                time.sleep(ALIGN_RECHECK_DELAY_S)
                        else:
                            task_pending_recheck = False
                            direction = "R" if platform_error_px > 0 else "L"
                            boosted_speed = min(MAX_MOTOR_PWM, round(
                                PLATFORM_MAX_SPEED * stuck_pwm_boost(platform_stuck_step)))
                            send(ser2, f"{direction}{boosted_speed}")


                            check_s = min(WHEEL_STUCK_CHECK_DELAY_S, pulse_ms / 1000.0)
                            time.sleep(check_s)
                            now_wheel = time.time()
                            moving_now = not wheels_appear_stuck(telemetry)
                            platform_last_movement_time, stuck_s = wheel_stuck_duration_update(
                                telemetry, platform_last_movement_time, now_wheel)
                            if moving_now and platform_stuck_step > 0:
                                platform_stuck_step -= 1
                                log_event(f"[safety] Wheels moving again during platform align — "
                                          f"stepping boost down to {stuck_boost_desc(platform_stuck_step)}.")
                            time.sleep(max(0.0, pulse_ms * stuck_pulse_boost(platform_stuck_step) / 1000.0 - check_s))
                            send(ser2, "X")
                            if stuck_s >= WHEEL_STUCK_DURATION_S:
                                platform_stuck_step += 1
                                platform_last_movement_time = now_wheel
                                log_event(f"[safety] Wheels appear stuck during platform align for "
                                          f">= {WHEEL_STUCK_DURATION_S:.0f}s — "
                                          f"boosting pulse to {stuck_boost_desc(platform_stuck_step)}.")
                    time.sleep(PLATFORM_LOOP_SLEEP_S)

                elif task == TASK_TURNAROUND:
                    if turnaround_phase == "rotating":
                        if telemetry.heading is None:
                            send(ser2, "X")
                        else:
                            if turnaround_target_heading is None:
                                turnaround_target_heading = ImuFusion.wrap(
                                    telemetry.heading + TURNAROUND_TARGET_DEG)
                            error = ImuFusion.wrap(turnaround_target_heading - telemetry.heading)
                            if abs(error) <= TURNAROUND_TOLERANCE_DEG:
                                send(ser2, "X")
                                turnaround_phase = "waiting_enter"
                                log_event("[task] Turnaround: rotation complete. "
                                          "Press Enter to resume line-following.")
                            else:
                                direction = "CCW" if error > 0 else "CW"
                                pulse_ms = compute_pulse_ms(
                                    error, TURNAROUND_TOLERANCE_DEG, ANGLE_ERROR_SATURATION_DEG,
                                    ANGLE_MIN_PULSE_MS, ANGLE_MAX_PULSE_MS)


                                send(ser2, f"{direction}{ANGLE_MAX_SPEED}")
                                time.sleep(pulse_ms / 1000.0)
                                send(ser2, "X")


                elif task == TASK_FRONT_TOF_ALIGN:
                    if front_tof_phase == "align":
                        reading = telemetry.tof.get(FRONT_TOF_LABEL)
                        step_label = (front_tof_steps[front_tof_step_index][0]
                                      if front_tof_step_index < len(front_tof_steps) else "?")
                        if reading is None:
                            send(ser2, "X")
                            time.sleep(FRONT_TOF_LOOP_SLEEP_S)
                        else:
                            error = reading - front_tof_target
                            pulse_ms = compute_pulse_ms(
                                error, FRONT_TOF_DEADBAND_MM, FRONT_TOF_ERROR_SATURATION_MM,
                                FRONT_TOF_MIN_PULSE_MS, FRONT_TOF_MAX_PULSE_MS)
                            if pulse_ms <= 0:
                                send(ser2, "X")
                                if not ALIGN_RECHECK_ENABLED or front_tof_pending_recheck:
                                    log_event(f"[task] {step_label}: CAM aligned ({reading:.0f}mm, "
                                              f"target={front_tof_target:.0f}mm, locked).")
                                    front_tof_step_index += 1
                                    enter_front_tof_step()
                                else:
                                    front_tof_pending_recheck = True
                                    time.sleep(ALIGN_RECHECK_DELAY_S)
                            else:
                                front_tof_pending_recheck = False
                                direction = "F" if error > 0 else "B"
                                speed = front_tof_speed_for_error(error)
                                boosted_speed = min(MAX_MOTOR_PWM, round(
                                    speed * stuck_pwm_boost(front_tof_stuck_step)))
                                send(ser2, f"{direction}{boosted_speed}")


                                check_s = min(WHEEL_STUCK_CHECK_DELAY_S, pulse_ms / 1000.0)
                                time.sleep(check_s)
                                now_wheel = time.time()
                                moving_now = not wheels_appear_stuck(telemetry)
                                front_tof_last_movement_time, stuck_s = wheel_stuck_duration_update(
                                    telemetry, front_tof_last_movement_time, now_wheel)
                                if moving_now and front_tof_stuck_step > 0:
                                    front_tof_stuck_step -= 1
                                    log_event(f"[safety] Wheels moving again during front-ToF approach — "
                                              f"stepping boost down to {stuck_boost_desc(front_tof_stuck_step)}.")
                                time.sleep(max(0.0, pulse_ms * stuck_pulse_boost(front_tof_stuck_step) / 1000.0 - check_s))
                                send(ser2, "X")
                                if stuck_s >= WHEEL_STUCK_DURATION_S:
                                    front_tof_stuck_step += 1
                                    front_tof_last_movement_time = now_wheel
                                    log_event(f"[safety] Wheels appear stuck during front-ToF approach "
                                              f"for >= {WHEEL_STUCK_DURATION_S:.0f}s — "
                                              f"boosting pulse to {stuck_boost_desc(front_tof_stuck_step)}.")
                                time.sleep(FRONT_TOF_LOOP_SLEEP_S)
                    elif front_tof_phase == "placeholder_actions":
                        kind, target_gap_px, hold_s = front_tof_action_list[front_tof_action_index]
                        if kind == "servo6_seek":
                            ret_seek, frame_seek = cap_bottom.read()
                            if ret_seek:
                                _, _, _, seek_detections = platform_processor.process_frame(frame_seek)
                                gap_px = platform_servo_gap_px(seek_detections)
                                if gap_px is not None and gap_px < target_gap_px:
                                    cap_bottom.pause()


                                    send_and_settle(ser2, servo_pos, f"6 {seek_servo6_angle:.1f}",
                                                     timeout_s=DISPENSE_SAFE_WAIT_TIMEOUT_S)
                                    log_event(f"[task] servo6_seek: reached gap={gap_px:.0f}px "
                                              f"(target={target_gap_px}px) at servo6={seek_servo6_angle:.1f}.")
                                    front_tof_action_index += 1
                                    enter_front_tof_action()
                                elif seek_servo6_angle <= SERVO_6_SEEK_MIN_ANGLE:
                                    cap_bottom.pause()
                                    send_and_settle(ser2, servo_pos, f"6 {seek_servo6_angle:.1f}",
                                                     timeout_s=DISPENSE_SAFE_WAIT_TIMEOUT_S)
                                    log_event(f"[task] servo6_seek: hit min angle "
                                              f"{SERVO_6_SEEK_MIN_ANGLE} without reaching target "
                                              f"(last gap={'n/a' if gap_px is None else f'{gap_px:.0f}px'}).")
                                    front_tof_action_index += 1
                                    enter_front_tof_action()
                                else:
                                    seek_servo6_angle = max(
                                        SERVO_6_SEEK_MIN_ANGLE, seek_servo6_angle - SERVO_6_SEEK_STEP_DEG)
                                    send(ser2, f"6 {seek_servo6_angle:.1f}")
                                    time.sleep(SERVO_6_SEEK_LOOP_SLEEP_S)
                        elif kind == "verify":
                            if verify_subphase == "checking":
                                ret_v, frame_v = cap_overhead.read()
                                if ret_v:
                                    verify_detections = ball_verify_processor.detect(frame_v)
                                    inside = ball_inside_platform(verify_detections)
                                    verify_streak = verify_streak + 1 if inside else 0
                                    if verify_streak >= BALL_VERIFY_CONFIRM_FRAMES:
                                        cap_overhead.pause()
                                        log_event("[task] verify: ball confirmed inside "
                                                  "platform bounding box — moving on.")
                                        front_tof_action_index += 1
                                        enter_front_tof_action()
                                    elif time.time() - verify_check_start >= BALL_VERIFY_CHECK_WINDOW_S:


                                        cap_overhead.pause()
                                        log_event(f"[task] verify: ball not confirmed within "
                                                  f"{BALL_VERIFY_CHECK_WINDOW_S:.1f}s — "
                                                  f"servo6 seeking dispense position again.")
                                        verify_subphase = "seeking"
                                        seek_servo6_angle = SERVO_6_SEEK_START_ANGLE
                                        send(ser2, f"6 {seek_servo6_angle}")
                                        cap_bottom.resume()
                            elif verify_subphase == "seeking":
                                ret_seek, frame_seek = cap_bottom.read()
                                if ret_seek:
                                    _, _, _, seek_detections = platform_processor.process_frame(frame_seek)
                                    gap_px = platform_servo_gap_px(seek_detections)
                                    reached = gap_px is not None and gap_px < SERVO_6_SEEK_TARGET_GAP_PX
                                    hit_min = seek_servo6_angle <= SERVO_6_SEEK_MIN_ANGLE
                                    if reached or hit_min:
                                        cap_bottom.pause()


                                        send_and_settle(ser2, servo_pos, f"6 {seek_servo6_angle:.1f}",
                                                         timeout_s=DISPENSE_SAFE_WAIT_TIMEOUT_S)
                                        send_and_settle(ser2, servo_pos, f"9 {R_PLACEHOLDER_SERVO9_TEMP_ANGLE}",
                                                         timeout_s=DISPENSE_SAFE_WAIT_TIMEOUT_S)
                                        if reached:
                                            log_event(f"[task] verify: servo6 reached gap={gap_px:.0f}px "
                                                      f"— resending '{R_PLACEHOLDER_DISPENSE_CMD}'.")
                                        else:
                                            log_event(f"[task] verify: servo6 hit min angle "
                                                      f"{SERVO_6_SEEK_MIN_ANGLE} without reaching target "
                                                      f"— resending '{R_PLACEHOLDER_DISPENSE_CMD}' anyway.")
                                        send(ser2, R_PLACEHOLDER_DISPENSE_CMD)
                                        verify_subphase = "dispensing"
                                        verify_action_start = time.time()
                                    else:
                                        seek_servo6_angle = max(
                                            SERVO_6_SEEK_MIN_ANGLE, seek_servo6_angle - SERVO_6_SEEK_STEP_DEG)
                                        send(ser2, f"6 {seek_servo6_angle:.1f}")
                                        time.sleep(SERVO_6_SEEK_LOOP_SLEEP_S)
                            elif verify_subphase == "dispensing":
                                if time.time() - verify_action_start >= R_PLACEHOLDER_DISPENSE_WAIT_S:
                                    send_and_settle(ser2, servo_pos, f"6 {SERVO_6_INITIAL}",
                                                     timeout_s=DISPENSE_SAFE_WAIT_TIMEOUT_S)
                                    time.sleep(BALL_VERIFY_SETTLE_S)
                                    verify_subphase = "checking"
                                    verify_streak = 0
                                    verify_check_start = time.time()
                                    cap_overhead.resume()
                                    log_event("[task] verify: servo6 back to initial — "
                                              "rechecking ball placement.")
                        elif time.time() - front_tof_action_start >= hold_s:
                            front_tof_action_index += 1
                            enter_front_tof_action()


                if (task == TASK_LINE_FOLLOW and line_follow_phase == "platform_align"
                        and task_aligned and not case_sent):
                    case = classify_case(platform_detections)
                    if case is not None:
                        case_label, case_color, case_cmd = case
                        send(ser1, case_cmd)
                        send(ser2, case_cmd)
                        case_sent = True
                        last_case_cmd = case_cmd


                        if case_cmd == "B":
                            last_ycrop_position = ycrop_side_of_frame_center(platform_detections)
                        gbr_count += 1
                        log_event(f"[task] Crop case: {case_label} -> sent '{case_cmd}' to both ESPs. "
                                  f"(G/B/R count: {gbr_count}"
                                  f"{f', ycrop={last_ycrop_position}' if case_cmd == 'B' else ''}).")


                if (task == TASK_LINE_FOLLOW and line_follow_phase == "platform_align"
                        and task_aligned and not servo5_offset_sent and platform_error_px is not None):
                    platform_cx = platform_error_px + IMAGE_CENTER_X
                    green_dist = closest_edge_distance(platform_detections, GCROP_CLASS_ID, platform_cx)
                    yellow_dist = closest_edge_distance(platform_detections, YCROP_CLASS_ID, platform_cx)
                    if green_dist is not None and yellow_dist is not None:
                        if yellow_dist > green_dist:
                            send_and_settle(ser2, servo_pos, f"5 {SERVO_5_OFFSET_YELLOW_FARTHER}")
                        else:
                            send_and_settle(ser2, servo_pos, f"5 {SERVO_5_OFFSET_GREEN_FARTHER_OR_EQUAL}")
                        servo5_offset_sent = True
                        log_event(f"[task] Servo 5 offset sent (green_dist={green_dist:.0f}px, "
                                  f"yellow_dist={yellow_dist:.0f}px).")


                if task == TASK_LINE_FOLLOW and line_follow_phase == "platform_align" and case_sent:
                    send(ser2, "X")
                    line_follow_phase = "waiting_enter"
                    log_event("[task] Platform aligned + case sent — "
                              "press Enter to run the dispense/topple sequence.")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log_event("\n[info] stopped by user.")
                    break
                elif key in ENTER_KEY_CODES:
                    if task == TASK_TURNAROUND:
                        if turnaround_phase == "waiting_enter":
                            task = TASK_LINE_FOLLOW
                            line_follow_phase = "following"


                            old_platform_reference_x = IMAGE_CENTER_X


                            log_event(f"[task] Enter pressed -> resuming line-following "
                                      f"(scan_direction={scan_direction})")

                    elif task == TASK_IDLE and not servo_ready_sent:
                        send_and_settle(ser2, servo_pos, f"8 {CAMERA_FRONT_ANGLE}")
                        send_and_settle(ser2, servo_pos, f"9 {SERVO_9_STRAIGHT_ANGLE}")


                        angle = CAMERA_RIGHT_ANGLE if camera_side == "right" else CAMERA_LEFT_ANGLE
                        send_and_settle(ser2, servo_pos, f"8 {angle}")
                        time.sleep(CAMERA_HUNT_SETTLE_EXTRA_WAIT_S)
                        servo_ready_sent = True
                        log_event(f"[init] Enter pressed -> servo 8 -> {CAMERA_FRONT_ANGLE} (front) then "
                                  f"{angle} ({camera_side}), servo 9 -> {SERVO_9_STRAIGHT_ANGLE} (straight). "
                                  f"Press Enter again to start line-following.")
                    elif task == TASK_IDLE and servo_ready_sent:
                        task = TASK_LINE_FOLLOW
                        line_follow_phase = "following"


                        old_platform_reference_x = IMAGE_CENTER_X
                        log_event(f"[task] Enter pressed -> {TASK_NAMES[TASK_LINE_FOLLOW]}")
                    elif task == TASK_LINE_FOLLOW and line_follow_phase == "waiting_enter":
                        task = TASK_FRONT_TOF_ALIGN
                        log_event(f"[task] Enter pressed -> {TASK_NAMES[TASK_FRONT_TOF_ALIGN]}")
                    elif task == TASK_FRONT_TOF_ALIGN and front_tof_phase == "waiting_enter":
                        task = TASK_LINE_FOLLOW
                        line_follow_phase = "following"


                        old_platform_reference_x = None
                        log_event("[task] Enter pressed -> resuming line-following.")


    except KeyboardInterrupt:
        log_event("\n[info] stopped by user.")
    finally:
        send(ser2, "X")
        stop_event.set()
        if cap_overhead:
            cap_overhead.release()
        if cap_bottom:
            cap_bottom.release()
        cv2.destroyAllWindows()
        ser1.close()
        ser2.close()
        if lane_processor:
            del lane_processor.detector
        if platform_processor:
            del platform_processor.detector
        if ball_verify_processor:
            del ball_verify_processor.detector
        gc.collect()
        _CUDA_CONTEXT.pop()
        _CUDA_DEVICE.retain_primary_context().detach()
        log_event("\nCleanup complete.")


if __name__ == "__main__":
    main()
