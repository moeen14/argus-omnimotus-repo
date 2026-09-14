from __future__ import annotations

import contextlib
import gc
import os
import queue
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


ENGINE_PATH = os.environ.get("ASABE_LANE_ENGINE", str(Path(__file__).resolve().parent / "highcam_rail.engine"))
OVERHEAD_CAMERA_DEVICE = os.environ.get(
    "ASABE_OVERHEAD_CAMERA_DEVICE",
    "/dev/v4l/by-id/usb-046d_0825_3D19ADE0-video-index0",
)
ESP1_PORT = os.environ.get("ASABE_ESP1_PORT", "/dev/esp32_01")
ESP2_PORT = os.environ.get("ASABE_ESP2_PORT", "/dev/esp32_02")
BAUD = 921600

IMG_SIZE = 640
NUM_CLASSES = 1
CONF_THRESH = 0.45
IOU_THRESH = 0.55
MASK_THRESH = 0.7


SRC_PTS = np.array([[78, 291], [142, 56], [463, 57], [520, 290]], dtype=np.float32)
SRC_CM = np.array([[0, 0], [0, 30.5], [30.5, 30.5], [30.5, 0]], dtype=np.float32)
PIXELS_PER_CM = 3.0
ORIGIN_X_OFFSET = 250.0
ORIGIN_Y_OFFSET = 660.0
THRESHOLD_Y = 80
FRAME_W, FRAME_H = 640, 480

DEFAULT_RAIL_WIDTH_BEV = 111
DEFAULT_SLOPE_DIFF = 0.0
WIDTH_EMA_ALPHA = 0.1
BEV_MIN_COMPONENT_AREA = 150
BEV_MASK_CLEAN_KERNEL = np.ones((3, 3), np.uint8)

LINE_SMOOTHING_ALPHA = 0.3

IMAGE_CENTER_X = FRAME_W / 2.0

ANGLE_DEADBAND_DEG = 0.5
CENTER_DEADBAND_PX = 2.0
ANGLE_ERROR_SATURATION_DEG = 12.0
CENTER_ERROR_SATURATION_PX = 200.0


NAV_SWITCH_FINE_ALIGN_DEADBAND_PX = 3.0
CENTER_MAX_SPEED = 120
CENTER_MIN_PULSE_MS = 10
CENTER_MAX_PULSE_MS = 60


ALIGN_RECHECK_ENABLED = True
ALIGN_RECHECK_DELAY_S = 1.0

CENTERLINE_COLOR = (255, 255, 255)
CAMERA_REF_COLOR = (0, 0, 255)
TARGET_DOT_COLOR = (0, 0, 255)
OFFSET_CENTERLINE_COLOR = (0, 165, 255)
OFFSET_CENTERLINE_COLOR_PINK = (203, 0, 255)


RIGHT_CAMERA_CENTERLINE_OFFSET_PX = 40.0


NAV_LANE_PROGRESS_FIRST_3_4_MM = 250.0


FIRST_LANE_LEFT_CENTERLINE_OFFSET_PX = 40.0


LAST_LANE_RIGHT_CENTERLINE_OFFSET_PX = 50.0
MASK_COLOR = np.array([0, 180, 0], dtype=np.float32)


CAMERA_FRONT_ANGLE = 266
CAMERA_LEFT_ANGLE = 179


CAMERA_LEFT_ANGLE_INITIAL = 180
CAMERA_RIGHT_ANGLE = 358
NO_RAILS_SWITCH_SECONDS = 1.5


CAMERA_HUNT_SETTLE_EXTRA_WAIT_S = 1.0
SERVO_8_IDLE_ANGLE = 170
SERVO_9_IDLE_ANGLE = 25
SERVO_9_STRAIGHT_ANGLE = 116.5

SERVO_5_INITIAL = 322
SERVO_INITIAL_COMMANDS = [
    ("T", "swerve initial"),
    (f"5 {SERVO_5_INITIAL}", "arm horizontal initial"),
    ("6 50", "arm vertical initial"),
    ("7 51", "front camera initial"),
    (f"8 {SERVO_8_IDLE_ANGLE}", "overhead camera idle"),
    (f"9 {SERVO_9_IDLE_ANGLE}", "servo 9 idle"),
]


def initial_nav_direction(camera_side):
    return "R" if camera_side == "right" else "L"


NAV_BASE_SPEED = 130
NAV_TOF_SLOWDOWN_START_MM = 300.0
NAV_TOF_MIN_BASE_SPEED = 70


NAV_TOF_MIN_SPEED_MM = 200.0


NAV_PWM_DELTA_MIN = 10
NAV_PWM_DELTA_MAX = 50


NAV_STEER_DELTA_MIN_DEG = 3.0
NAV_STEER_DELTA_MAX_DEG = 30.0


NAV_STEER_RESEND_DELTA_DEG = 1.0


NAV_RAIL_LOSS_STOP_AFTER_S = 0.35


NAV_LAST_LANE_FRONT_STEER_TRIGGER_MM = 100.0
NAV_LAST_LANE_FRONT_STEER_MAX_DEG = 20.0


TOF_LABELS = ["FRS", "BRS", "BLS", "FLF", "FLS", "CAM", "FRF"]
NAV_SIDE_TOF_LABELS = {"R": ("BRS", "FRS"), "L": ("BLS", "FLS")}


NAV_TOF_THRESHOLD_MM_L = 120.0
NAV_TOF_THRESHOLD_MM_R = 150.0


def nav_tof_threshold_mm(nav_direction):
    return NAV_TOF_THRESHOLD_MM_L if nav_direction == "L" else NAV_TOF_THRESHOLD_MM_R


NAV_NOTCH_FRF_THRESHOLD_MM = 150.0

NAV_NOTCH_RECOVERY_RIGHT_TOF_SLOWDOWN_START_MM = 300.0
NAV_NOTCH_RECOVERY_RIGHT_TOF_STOP_MM = 450.0
NAV_NOTCH_RECOVERY_MIN_SPEED = 60


NAV_END_FRONT_TOF_THRESHOLD_MM = 150.0
NAV_END_RIGHT_TOF_THRESHOLD_MM = 150.0


NAV_SWITCH_FORWARD_BASE_PWM = 100
NAV_SWITCH_APPROACH_MIN_PWM = 60
NAV_SWITCH_APPROACH_SLOWDOWN_START_PX = 300.0
NAV_SWITCH_ANGLE_PWM_DELTA_MIN = 8
NAV_SWITCH_ANGLE_PWM_DELTA_MAX = 40
NAV_SWITCH_DIST_PWM_DELTA_MIN = 8
NAV_SWITCH_DIST_PWM_DELTA_MAX = 35
NAV_SWITCH_TARGET_CENTER_DEADBAND_PX = 8.0
NAV_SWITCH_BACKWARD_SEARCH_AFTER_S = 3.0
NAV_SWITCH_BACKWARD_SEARCH_PWM = 70
NAV_SWITCH_OLD_CENTERLINE_LOSS_CONFIRM_S = 0.30


NAV_SWITCH_NEW_LANE_ENTRY_MARGIN_PX = 60.0


NAV_SWITCH_MAX_FORWARD_DURATION_S = 3.0


NAV_SWITCH_FRONT_TOF_STOP_MM = 100.0


NAV_SWITCH_FRONT_TOF_SLOWDOWN_START_MM = 300.0
NAV_SWITCH_FRONT_TOF_MIN_PWM = 70


NAV_SWITCH_FRONT_TOF_ALIGN_DEADBAND_MM = 10.0
NAV_SWITCH_FRONT_TOF_ALIGN_SATURATION_MM = 100.0
NAV_SWITCH_FRONT_TOF_ALIGN_MIN_PULSE_MS = 20
NAV_SWITCH_FRONT_TOF_ALIGN_MAX_PULSE_MS = 40
NAV_SWITCH_FRONT_TOF_ALIGN_SPEED = 100


def log_event(*args):
    message = " ".join(str(a) for a in args)
    print(f"{datetime.now().isoformat(timespec='milliseconds')} {message}")


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
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while self.running:
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

    def read(self):
        if not self.running and self.q.empty():
            return False, None
        return True, self.q.get()

    def flush(self):


        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

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


def open_overhead_camera():
    candidates = [OVERHEAD_CAMERA_DEVICE, 0, 1, 2]
    for source in candidates:
        cam = ThreadedCamera(source, FRAME_W, FRAME_H)
        if cam.isOpened():
            ok, _ = cam.read()
            if ok:
                log_event(f"[init] Overhead camera opened on {source}")
                return cam
        cam.release()
    raise RuntimeError(f"Could not open overhead camera (tried {candidates})")


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


def seg_postprocess(det, proto, r, pad, orig_shape):
    H, W = orig_shape[:2]
    p = det[0].T
    conf = p[:, 4:4 + NUM_CLASSES].max(axis=1)
    keep = conf > CONF_THRESH
    if not np.any(keep):
        return np.zeros((H, W), dtype=np.uint8), []

    p_keep = p[keep]
    conf = conf[keep]
    cx, cy, bw, bh = p_keep[:, 0], p_keep[:, 1], p_keep[:, 2], p_keep[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    coeffs = p_keep[:, 4 + NUM_CLASSES:]

    idx = nms(boxes, conf, IOU_THRESH)
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


    mid_x = float(centerline_pts[len(centerline_pts) // 2, 0, 0])
    signed_dist = mid_x - image_center_x
    dist_px = abs(signed_dist)
    side_label = "RIGHT" if signed_dist > 0 else ("LEFT" if signed_dist < 0 else "CENTER")

    signed_angle_deg = float(np.degrees(np.arctan(mc)))

    return signed_angle_deg, dist_px, side_label


def last_lane_front_steer_bias_deg(front_tof_mean):


    if front_tof_mean is None or front_tof_mean >= NAV_LAST_LANE_FRONT_STEER_TRIGGER_MM:
        return 0.0
    frac = (NAV_LAST_LANE_FRONT_STEER_TRIGGER_MM - front_tof_mean) / NAV_LAST_LANE_FRONT_STEER_TRIGGER_MM
    return max(0.0, min(1.0, frac)) * NAV_LAST_LANE_FRONT_STEER_MAX_DEG


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


def flipped_direction(direction):
    flips = {"F": "B", "B": "F", "CW": "CCW", "CCW": "CW"}
    return flips.get(direction, direction)


def nav_base_speed_for_tof(tof_side_mean):


    if tof_side_mean is None or tof_side_mean >= NAV_TOF_SLOWDOWN_START_MM:
        return float(NAV_BASE_SPEED)
    if tof_side_mean <= NAV_TOF_MIN_SPEED_MM:
        return float(NAV_TOF_MIN_BASE_SPEED)

    span = NAV_TOF_SLOWDOWN_START_MM - NAV_TOF_MIN_SPEED_MM
    frac = (tof_side_mean - NAV_TOF_MIN_SPEED_MM) / span
    return NAV_TOF_MIN_BASE_SPEED + frac * (NAV_BASE_SPEED - NAV_TOF_MIN_BASE_SPEED)


def nav_notch_recovery_speed_for_right_tof(right_tof_value):


    if right_tof_value is None or right_tof_value <= NAV_NOTCH_RECOVERY_RIGHT_TOF_SLOWDOWN_START_MM:
        return float(NAV_BASE_SPEED)
    if right_tof_value >= NAV_NOTCH_RECOVERY_RIGHT_TOF_STOP_MM:
        return float(NAV_NOTCH_RECOVERY_MIN_SPEED)

    span = NAV_NOTCH_RECOVERY_RIGHT_TOF_STOP_MM - NAV_NOTCH_RECOVERY_RIGHT_TOF_SLOWDOWN_START_MM
    frac = (right_tof_value - NAV_NOTCH_RECOVERY_RIGHT_TOF_SLOWDOWN_START_MM) / span
    return NAV_BASE_SPEED - frac * (NAV_BASE_SPEED - NAV_NOTCH_RECOVERY_MIN_SPEED)


def compute_pulse_ms(error_mag, deadband, saturation, min_pulse_ms, max_pulse_ms):


    mag = min(abs(error_mag), saturation)
    if mag <= deadband:
        return 0.0
    span = saturation - deadband
    frac = (mag - deadband) / span
    return min_pulse_ms + frac * (max_pulse_ms - min_pulse_ms)


def clamp_pwm(value):
    return max(0, min(255, int(round(value))))


def nav_switch_front_tof_pwm_scale(front_tof_mean):


    if front_tof_mean is None or front_tof_mean >= NAV_SWITCH_FRONT_TOF_SLOWDOWN_START_MM:
        return 1.0
    min_fraction = NAV_SWITCH_FRONT_TOF_MIN_PWM / NAV_SWITCH_FORWARD_BASE_PWM
    if front_tof_mean <= NAV_SWITCH_FRONT_TOF_STOP_MM:
        return min_fraction
    span = NAV_SWITCH_FRONT_TOF_SLOWDOWN_START_MM - NAV_SWITCH_FRONT_TOF_STOP_MM
    frac = (front_tof_mean - NAV_SWITCH_FRONT_TOF_STOP_MM) / span
    return min_fraction + frac * (1.0 - min_fraction)


def nav_switch_approach_base_pwm(dist_px):
    if dist_px >= NAV_SWITCH_APPROACH_SLOWDOWN_START_PX:
        return float(NAV_SWITCH_FORWARD_BASE_PWM)
    if dist_px <= NAV_SWITCH_TARGET_CENTER_DEADBAND_PX:
        return float(NAV_SWITCH_APPROACH_MIN_PWM)

    span = NAV_SWITCH_APPROACH_SLOWDOWN_START_PX - NAV_SWITCH_TARGET_CENTER_DEADBAND_PX
    frac = (dist_px - NAV_SWITCH_TARGET_CENTER_DEADBAND_PX) / span
    return NAV_SWITCH_APPROACH_MIN_PWM + ((NAV_SWITCH_FORWARD_BASE_PWM - NAV_SWITCH_APPROACH_MIN_PWM) *
                                          max(0.0, min(1.0, frac)))


def effective_side_label(side_label, camera_side):


    if camera_side != "right" or side_label not in ("LEFT", "RIGHT"):
        return side_label
    return "RIGHT" if side_label == "LEFT" else "LEFT"


def nav_switch_forward_pwms(angle_deg, dist_px, side_label, camera_side, enable_slowdown, pwm_boost=1.0):


    left_boost = 0.0
    right_boost = 0.0
    base_pwm = nav_switch_approach_base_pwm(dist_px) if enable_slowdown else float(NAV_SWITCH_FORWARD_BASE_PWM)
    base_pwm *= pwm_boost
    correction_scale = base_pwm / NAV_SWITCH_FORWARD_BASE_PWM if NAV_SWITCH_FORWARD_BASE_PWM else 0.0

    rotation = rotation_for_angle(angle_deg)
    angle_delta = compute_pulse_ms(
        angle_deg,
        ANGLE_DEADBAND_DEG,
        ANGLE_ERROR_SATURATION_DEG,
        NAV_SWITCH_ANGLE_PWM_DELTA_MIN,
        NAV_SWITCH_ANGLE_PWM_DELTA_MAX,
    ) * correction_scale
    if rotation == "CW":
        left_boost += angle_delta
    elif rotation == "CCW":
        right_boost += angle_delta

    if dist_px > CENTER_DEADBAND_PX:
        dist_delta = compute_pulse_ms(
            dist_px,
            CENTER_DEADBAND_PX,
            CENTER_ERROR_SATURATION_PX,
            NAV_SWITCH_DIST_PWM_DELTA_MIN,
            NAV_SWITCH_DIST_PWM_DELTA_MAX,
        ) * correction_scale
        effective_label = effective_side_label(side_label, camera_side)
        if effective_label == "LEFT":
            right_boost += dist_delta
        elif effective_label == "RIGHT":
            left_boost += dist_delta

    br = clamp_pwm(base_pwm + right_boost)
    fr = clamp_pwm(base_pwm + right_boost)
    bl = clamp_pwm(base_pwm + left_boost)
    fl = clamp_pwm(base_pwm + left_boost)
    return f"F{br},{fr},{bl},{fl}", rotation, left_boost, right_boost, base_pwm


def nav_switch_angle_pwms(base_pwm, angle_deg):


    rotation = rotation_for_angle(angle_deg)
    angle_delta = compute_pulse_ms(
        angle_deg, ANGLE_DEADBAND_DEG, ANGLE_ERROR_SATURATION_DEG,
        NAV_SWITCH_ANGLE_PWM_DELTA_MIN, NAV_SWITCH_ANGLE_PWM_DELTA_MAX)
    left_boost = angle_delta if rotation == "CW" else 0.0
    right_boost = angle_delta if rotation == "CCW" else 0.0
    br = fr = clamp_pwm(base_pwm + right_boost)
    bl = fl = clamp_pwm(base_pwm + left_boost)
    return f"F{br},{fr},{bl},{fl}", br, bl


def new_lane_entry_ok(bottom_cx, camera_side):


    if camera_side == "right":
        return bottom_cx <= IMAGE_CENTER_X - NAV_SWITCH_NEW_LANE_ENTRY_MARGIN_PX
    return bottom_cx >= IMAGE_CENTER_X + NAV_SWITCH_NEW_LANE_ENTRY_MARGIN_PX


def draw_dashed_vline(img, x, y_top, y_bottom, color, thickness=2, dash_len=12, gap_len=10):
    x = int(x)
    y = float(y_top)
    while y < y_bottom:
        y_end = min(y + dash_len, y_bottom)
        cv2.line(img, (x, int(y)), (x, int(y_end)), color, thickness)
        y += dash_len + gap_len


_SERIAL_LOCK = threading.Lock()

NAV_PHASE_IDLE = "idle"
NAV_PHASE_FOLLOWING = "following"


NAV_PHASE_SWITCHING = "switching"


NAV_PHASE_SWITCH_FRONT_ALIGN = "switch_front_align"
NAV_PHASE_SWITCH_SAFETY_ALIGN = "switch_safety_align"
NAV_PHASE_SWITCH_STOPPED = "switch_stopped"


NAV_PHASE_NOTCH_RECOVERY = "notch_recovery"


NAV_PHASE_DONE = "done"


def sequence_label(nav_phase, lane_number, on_last_lane, tof_side_mean, near_front):


    lane_tag = "LAST LANE" if on_last_lane else ("FIRST LANE" if lane_number == 1 else f"LANE {lane_number}")
    if nav_phase == NAV_PHASE_IDLE:
        return "STARTUP"
    if nav_phase == NAV_PHASE_FOLLOWING:
        if tof_side_mean is None:
            progress = "-"
        elif tof_side_mean < NAV_TOF_SLOWDOWN_START_MM:
            progress = "LAST 1/4"
        else:
            progress = "FIRST 3/4"
        return f"LINE FOLLOWING - {lane_tag} ({progress})"
    if nav_phase == NAV_PHASE_SWITCH_FRONT_ALIGN:
        return f"FRONT ToF ALIGNMENT - {lane_tag} (correcting overshoot toward target)"
    if nav_phase == NAV_PHASE_SWITCH_SAFETY_ALIGN:
        return f"LANE ALIGNMENT - {lane_tag} (centering on new centerline)"
    if nav_phase == NAV_PHASE_SWITCHING:
        if near_front:
            return (f"INTERLANE TRANSITION - leaving {lane_tag} "
                    "(FRONT ToF CLOSE — ramping down to stop, no alignment)")
        return f"INTERLANE TRANSITION - leaving {lane_tag}"
    if nav_phase == NAV_PHASE_SWITCH_STOPPED:
        return f"INTERLANE TRANSITION - stopped on {lane_tag}, waiting for Enter"
    if nav_phase == NAV_PHASE_NOTCH_RECOVERY:
        return f"NOTCH RECOVERY - {lane_tag}"
    if nav_phase == NAV_PHASE_DONE:
        return "FINISHED - LAST LANE"
    return "-"


ENTER_KEY_CODES = (13, 10)


AUTONOMOUS_MODE = True


SERVO_SETTLE_TIMEOUT_S = 2.0


SERVO_DONE_EVENTS = {servo_id: threading.Event() for servo_id in range(5, 10)}


SWERVE_MODE_DONE_EVENT = threading.Event()


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


def wait_for_swerve_mode_done(label, timeout_s=SERVO_SETTLE_TIMEOUT_S):


    if SWERVE_MODE_DONE_EVENT.wait(timeout_s):
        log_event(f"[settle] {label} confirmed by Board 02 (DONE,MODE).")
        return True
    log_event(f"[settle] {label} did NOT confirm within {timeout_s:.1f}s "
              "(no DONE,MODE seen — confirm board02 is flashed with the matching "
              "firmware) — proceeding anyway.")
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


def send_and_settle(ser, cmd, timeout_s=SERVO_SETTLE_TIMEOUT_S):


    servo_ids, label = settle_servo_ids_for_command(cmd)
    if cmd in ("I", "M", "T"):
        SWERVE_MODE_DONE_EVENT.clear()
        send(ser, cmd)
        wait_for_swerve_mode_done(label, timeout_s=timeout_s)
        return
    if servo_ids:
        for sid in servo_ids:
            SERVO_DONE_EVENTS[sid].clear()
    send(ser, cmd)
    if servo_ids:
        wait_for_servos_settle(servo_ids, label, timeout_s=timeout_s)


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
        with _SERIAL_LOCK:
            ser1.write(b"F\n")
            ser1.flush()

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


def serial_reader(ser, stop_event):


    while not stop_event.is_set():
        try:
            raw = ser.readline()
        except Exception:
            break
        if not raw:
            continue
        line = raw.decode(errors="ignore").strip()
        if line:
            if line.startswith("ERR,"):
                log_event(f"[err] Board 02 rejected command: {line}")
            elif line.startswith("DONE,MODE,"):
                SWERVE_MODE_DONE_EVENT.set()
            elif line.startswith("DONE,"):
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        servo_id = int(parts[1])
                    except ValueError:
                        servo_id = None
                    if servo_id in SERVO_DONE_EVENTS:
                        SERVO_DONE_EVENTS[servo_id].set()


def esp1_reader(ser, stop_event, tof, tof_reset_request, tof_reset_done):


    while not stop_event.is_set():
        if tof_reset_request.is_set():
            tof_reset_request.clear()
            log_event("[notch] Sending ToF reset (F) to Board 01...")
            with _SERIAL_LOCK:
                ser.write(b"F\n")
                ser.flush()
            deadline = time.time() + 6.0
            saw_done = False
            while time.time() < deadline:
                try:
                    raw = ser.readline()
                except Exception:
                    break
                if not raw:
                    continue
                line = raw.decode(errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("ACK,TOF_RESET,DONE"):
                    saw_done = True
                    break
            log_event(f"[notch] ToF reset done (ack_done={saw_done}).")
            tof_reset_done.set()
            continue

        try:
            raw = ser.readline()
        except Exception:
            break
        if not raw:
            continue
        line = raw.decode(errors="ignore").strip()
        if not line:
            continue

        if not line.startswith("TOF,["):
            continue
        body = line[line.find("[") + 1:line.rfind("]")]
        for item in body.split(","):
            if ":" not in item:
                continue
            label, value = item.split(":", 1)
            label = label.strip()
            if label in tof:
                try:
                    tof[label] = float(value.strip())
                except ValueError:
                    pass


def send_all_servos_initial(ser2):
    log_event("[init] Sending all servos to initial position...")
    for command, label in SERVO_INITIAL_COMMANDS:
        send_and_settle(ser2, command)
        log_event(f"  ({label})")


class LaneAlignmentProcessor:


    def __init__(self, engine_path, image_center_x, warmup_iters=5):
        self.engine_path = engine_path
        self.image_center_x = image_center_x

        log_event(f"Loading TRT Engine: {self.engine_path}")
        self.detector = TRTSegDetector(self.engine_path)

        self.last_known_rail_width = DEFAULT_RAIL_WIDTH_BEV
        self.last_known_slope_diff = DEFAULT_SLOPE_DIFF

        self.smoothed_m1 = self.smoothed_b1 = None
        self.smoothed_m2 = self.smoothed_b2 = None
        self._was_tracking = False

        self.fps_s = 0.0
        self._t_prev = None

        self._warmup(warmup_iters)
        log_event("Initialization complete. Ready to process feed.")

    def _warmup(self, iters):
        dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(max(1, iters)):
            preprocess_into_buffer(dummy, self.detector.h_input)
            self.detector.infer()
        log_event(f"GPU warm-up done ({iters} iters)")

    def process_frame(self, frame, reference_x=None, centerline_offset_px=0.0,
                       centerline_offset_color=OFFSET_CENTERLINE_COLOR,
                       extra_offset_px=0.0, extra_offset_color=OFFSET_CENTERLINE_COLOR_PINK):


        t_now = time.time()
        if self._t_prev:
            dt = t_now - self._t_prev
            if dt > 1e-6:
                self.fps_s = 0.2 / dt + 0.8 * self.fps_s if self.fps_s else 1.0 / dt
        self._t_prev = t_now

        if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        if reference_x is None:
            reference_x = self.image_center_x

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

        draw_dashed_vline(disp, reference_x, THRESHOLD_Y, FRAME_H, CAMERA_REF_COLOR)

        angle_deg, dist_px, side_label = 0.0, 0.0, "N/A"
        if have_rails:
            limit_img = np.array([[[img_cx, THRESHOLD_Y]], [[img_cx, FRAME_H]]], dtype=np.float32)
            limit_bev = cv2.perspectiveTransform(limit_img, H_IMG2BEV)
            y_far, y_near = limit_bev[0][0][1], limit_bev[1][0][1]

            centerline_pts = project_line_safe(mc, bc, y_far, y_near, H_BEV2IMG)
            tracked_pts = centerline_pts
            if centerline_offset_px:
                tracked_pts = centerline_pts + np.array(
                    [int(round(centerline_offset_px)), 0], dtype=centerline_pts.dtype)
            angle_deg, dist_px, side_label = centerline_angle_and_distance(
                tracked_pts, reference_x, mc)

            cv2.polylines(disp, [centerline_pts], False, CENTERLINE_COLOR, 3)
            if centerline_offset_px:
                cv2.polylines(disp, [tracked_pts], False, centerline_offset_color, 3)
            if extra_offset_px:
                extra_pts = centerline_pts + np.array(
                    [int(round(extra_offset_px)), 0], dtype=centerline_pts.dtype)
                cv2.polylines(disp, [extra_pts], False, extra_offset_color, 3)
            cv2.circle(disp, (int(bottom_cx), FRAME_H - 5), 6, TARGET_DOT_COLOR, -1)

        cv2.putText(disp, f"ANGLE:{angle_deg:+.1f}deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(disp, f"DIST:{dist_px:.1f}px ({side_label})",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(disp, f"FPS:{self.fps_s:.1f}",
                    (disp.shape[1] - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return disp, have_rails, angle_deg, dist_px, side_label, bottom_cx


def main():
    log_event("=" * 60)
    log_event(f"RUN START {datetime.now().isoformat(timespec='seconds')}")
    log_event(f"Connecting to ESP1 (sensors) {ESP1_PORT} and ESP2 (actuators) {ESP2_PORT} at {BAUD} baud...")
    ser1 = serial.Serial(ESP1_PORT, BAUD, timeout=0.1)
    ser2 = serial.Serial(ESP2_PORT, BAUD, timeout=0.1)
    time.sleep(0.5)
    ser1.reset_input_buffer()
    ser2.reset_input_buffer()
    reset_esp1_tofs(ser1)

    stop_event = threading.Event()
    tof = {label: None for label in TOF_LABELS}
    tof_reset_request = threading.Event()
    tof_reset_done = threading.Event()
    threading.Thread(target=esp1_reader, args=(ser1, stop_event, tof, tof_reset_request, tof_reset_done), daemon=True).start()
    threading.Thread(target=serial_reader, args=(ser2, stop_event), daemon=True).start()

    send_all_servos_initial(ser2)

    processor = None
    cap = None
    try:
        processor = LaneAlignmentProcessor(
            engine_path=ENGINE_PATH,
            image_center_x=IMAGE_CENTER_X,
        )

        cap = open_overhead_camera()


        camera_side = "left"
        no_rails_since = None
        rail_loss_stop_sent = False
        hunt_reacquire_waiting_for_enter = False
        tof_recovery_waiting_for_enter = False
        notch_waiting_for_enter = False


        notch_used = False
        notch_armed = False


        notch_reset_pending = False


        notch_switch_skip_flip = False

        nav_phase = NAV_PHASE_IDLE
        nav_direction = initial_nav_direction(camera_side)


        travel_stage = "leaving_old"
        movement_direction = "forward"
        old_loss_since = None
        new_search_started_at = None
        continuous_forward_since = None


        align_pending_recheck = False

        front_align_pending_recheck = False


        switch_safety_align_last_lane_trigger = False


        lane_number = 1
        on_last_lane = False


        near_front = False


        last_steer_direction = None
        last_steer_delta_deg = 0.0


        hud_wheel_mode = None
        hud_br = hud_fr = hud_bl = hud_fl = 0.0

        def reached_new_lane(reason, is_last_lane_trigger=False):


            nonlocal nav_phase, nav_direction, notch_switch_skip_flip
            nonlocal last_steer_direction, last_steer_delta_deg, lane_number, on_last_lane
            lane_number += 1
            if is_last_lane_trigger:
                on_last_lane = True
            if notch_switch_skip_flip:


                notch_switch_skip_flip = False
            else:
                nav_direction = "R" if nav_direction == "L" else "L"
            send_and_settle(ser2, "T")
            last_steer_direction = None
            last_steer_delta_deg = 0.0
            nav_phase = NAV_PHASE_SWITCH_STOPPED


            cap.flush()
            log_event(f"[nav] {reason} — swerve -> T; stopped, waiting for Enter "
                      f"(direction -> {nav_direction}, camera_side unchanged: {camera_side}).")

        def arm_notch_if_needed():


            nonlocal notch_armed
            if notch_used or nav_direction != "R":
                notch_armed = False
                return
            frf = tof.get("FRF")
            notch_armed = frf is not None and frf > NAV_NOTCH_FRF_THRESHOLD_MM
            log_event(f"[notch] R-direction FOLLOWING starting — FRF={frf} -> "
                      f"{'armed' if notch_armed else 'not armed'}.")

        log_event("=" * 60)
        log_event(" NAVIGATION TEST (line-following)")
        log_event(" Enter: servo 8/9 to ready pose, then continuous line-following.")
        log_event(" 'q' in the video window or Ctrl+C to quit.")
        log_event("=" * 60)

        with suppress_stderr():
            while True:
                ret, frame = cap.read()
                if not ret:
                    log_event("\n[warn] frame grab failed, stopping.")
                    break


                side_a, side_b = NAV_SIDE_TOF_LABELS[nav_direction]
                tof_a, tof_b = tof.get(side_a), tof.get(side_b)
                tof_side_mean = ((tof_a + tof_b) / 2.0
                                 if tof_a is not None and tof_b is not None
                                 else None)


                tof_threshold_mm = nav_tof_threshold_mm(nav_direction)


                lane_first_3_4 = tof_side_mean is None or tof_side_mean >= NAV_LANE_PROGRESS_FIRST_3_4_MM
                offset_extra_px, offset_extra_color = 0.0, OFFSET_CENTERLINE_COLOR_PINK
                if nav_phase != NAV_PHASE_FOLLOWING:
                    offset_px, offset_color = (
                        (RIGHT_CAMERA_CENTERLINE_OFFSET_PX, OFFSET_CENTERLINE_COLOR)
                        if camera_side == "right" else (0.0, OFFSET_CENTERLINE_COLOR))
                elif on_last_lane and camera_side == "right" and lane_first_3_4:
                    offset_px, offset_color = LAST_LANE_RIGHT_CENTERLINE_OFFSET_PX, OFFSET_CENTERLINE_COLOR_PINK
                    offset_extra_px, offset_extra_color = RIGHT_CAMERA_CENTERLINE_OFFSET_PX, OFFSET_CENTERLINE_COLOR
                elif lane_number == 1 and not on_last_lane and camera_side == "left" and lane_first_3_4:
                    offset_px, offset_color = FIRST_LANE_LEFT_CENTERLINE_OFFSET_PX, OFFSET_CENTERLINE_COLOR_PINK
                elif camera_side == "right":
                    offset_px, offset_color = RIGHT_CAMERA_CENTERLINE_OFFSET_PX, OFFSET_CENTERLINE_COLOR
                else:
                    offset_px, offset_color = 0.0, OFFSET_CENTERLINE_COLOR

                overlay, have_rails, angle_deg, dist_px, side_label, bottom_cx = processor.process_frame(
                    frame, centerline_offset_px=offset_px, centerline_offset_color=offset_color,
                    extra_offset_px=offset_extra_px, extra_offset_color=offset_extra_color)
                entry_ok = have_rails and new_lane_entry_ok(bottom_cx, camera_side)


                camera_mismatched = camera_side != ("right" if nav_direction == "R" else "left")


                now = time.time()
                if nav_phase in (NAV_PHASE_FOLLOWING, NAV_PHASE_NOTCH_RECOVERY):
                    if have_rails:
                        no_rails_since = None
                        rail_loss_stop_sent = False
                    else:
                        if no_rails_since is None:
                            no_rails_since = now
                            rail_loss_stop_sent = False
                        elif now - no_rails_since >= NO_RAILS_SWITCH_SECONDS:
                            camera_side = "right" if camera_side == "left" else "left"
                            angle = CAMERA_RIGHT_ANGLE if camera_side == "right" else CAMERA_LEFT_ANGLE
                            send(ser2, "X")
                            send_and_settle(ser2, f"8 {angle}")


                            time.sleep(CAMERA_HUNT_SETTLE_EXTRA_WAIT_S)
                            if nav_phase == NAV_PHASE_FOLLOWING:
                                last_steer_direction = None
                                last_steer_delta_deg = 0.0
                            hunt_reacquire_waiting_for_enter = True
                            no_rails_since = None
                            rail_loss_stop_sent = False
                            log_event(f"[hunt] Rails lost — servo 8 -> {angle} ({camera_side}); "
                                      f"waited {CAMERA_HUNT_SETTLE_EXTRA_WAIT_S:.1f}s for vision to catch up; "
                                      "waiting for rails, then Enter.")
                            continue

                rail_loss_elapsed = (now - no_rails_since) if no_rails_since is not None else 0.0


                frf, flf = tof.get("FRF"), tof.get("FLF")
                front_tof_mean = (frf + flf) / 2.0 if frf is not None and flf is not None else None
                right_tof_a, right_tof_b = tof.get("BRS"), tof.get("FRS")
                right_tof_mean = ((right_tof_a + right_tof_b) / 2.0
                                  if right_tof_a is not None and right_tof_b is not None
                                  else None)


                right_tof_candidates = [v for v in (right_tof_a, right_tof_b) if v is not None]
                right_tof_max = max(right_tof_candidates) if right_tof_candidates else None


                if hunt_reacquire_waiting_for_enter:
                    label = "WAITING (press Enter)" if have_rails else "HUNTING"
                elif tof_recovery_waiting_for_enter:
                    label = f"TOF WAIT {nav_direction}"
                elif notch_reset_pending:
                    label = "NOTCH TRIPPED - RESETTING TOFS"
                elif notch_waiting_for_enter:
                    label = "NOTCH TRIPPED - WAIT (press Enter)"
                elif nav_phase == NAV_PHASE_IDLE:
                    label = "WAITING (press Enter)"
                elif nav_phase == NAV_PHASE_FOLLOWING:
                    if (nav_direction == "R" and front_tof_mean is not None
                            and front_tof_mean < NAV_END_FRONT_TOF_THRESHOLD_MM
                            and right_tof_mean is not None
                            and right_tof_mean < NAV_END_RIGHT_TOF_THRESHOLD_MM):
                        label = f"TRACK FINISHED (front {front_tof_mean:.0f}mm, right {right_tof_mean:.0f}mm)"
                    elif (nav_direction == "R" and notch_armed
                            and frf is not None and frf < NAV_NOTCH_FRF_THRESHOLD_MM):
                        label = f"NOTCH TRIPPED (FRF {frf:.0f}mm)"
                    elif tof_side_mean is not None and tof_side_mean <= tof_threshold_mm:
                        label = f"STOPPING {nav_direction} (mean {tof_side_mean:.0f}mm)"
                    elif not have_rails:
                        label = (f"VISION HOLD {nav_direction}"
                                 if rail_loss_elapsed < NAV_RAIL_LOSS_STOP_AFTER_S
                                 else f"WAITING HUNT {nav_direction}")
                    else:
                        base_preview = nav_base_speed_for_tof(tof_side_mean)
                        if base_preview < NAV_BASE_SPEED:
                            label = f"FOLLOWING {nav_direction} SLOW {base_preview:.0f}"
                        else:
                            label = f"FOLLOWING {nav_direction}"
                elif nav_phase == NAV_PHASE_SWITCHING:
                    if travel_stage == "leaving_old":
                        label = "LEAVING OLD CENTERLINE"
                    elif travel_stage == "searching_new":
                        if have_rails and not entry_ok:
                            label = f"SEARCHING NEW CENTERLINE (rejected, wrong side x={bottom_cx:.0f}px)"
                        else:
                            label = "SEARCHING NEW CENTERLINE"
                    elif travel_stage == "tracking_new":
                        label = "TRACKING NEW CENTERLINE" if have_rails else "TRACKING NEW CENTERLINE NO RAILS"
                    elif travel_stage == "backtracking_new":
                        if have_rails and not entry_ok:
                            label = f"BACKTRACKING (rejected, wrong side x={bottom_cx:.0f}px)"
                        else:
                            label = "BACKTRACKING FOR NEW CENTERLINE" if have_rails else "BACKTRACKING NO RAILS"
                    else:
                        label = travel_stage
                elif nav_phase == NAV_PHASE_SWITCH_FRONT_ALIGN:
                    if front_tof_mean is None:
                        label = "FRONT ALIGN: NO READING, WAITING"
                    elif abs(front_tof_mean - NAV_SWITCH_FRONT_TOF_STOP_MM) <= NAV_SWITCH_FRONT_TOF_ALIGN_DEADBAND_MM:
                        label = "FRONT ALIGN: DISTANCE OK, CONFIRMING"
                    else:
                        label = f"FRONT ALIGN: CORRECTING (front={front_tof_mean:.0f}mm)"
                elif nav_phase == NAV_PHASE_SWITCH_SAFETY_ALIGN:
                    if not have_rails:
                        label = "SAFETY ALIGN: NO RAILS, WAITING"
                    elif dist_px <= NAV_SWITCH_FINE_ALIGN_DEADBAND_PX:
                        label = "SAFETY ALIGN: DISTANCE OK, CONFIRMING"
                    else:
                        label = f"SAFETY ALIGN: CENTERING (dist={dist_px:.0f}px)"
                elif nav_phase == NAV_PHASE_SWITCH_STOPPED:
                    label = "STOPPED ON NEXT LANE (press Enter)"
                elif nav_phase == NAV_PHASE_NOTCH_RECOVERY:
                    if not have_rails:
                        label = (f"NOTCH RECOVERY {nav_direction} NO RAILS"
                                 if rail_loss_elapsed < NAV_RAIL_LOSS_STOP_AFTER_S
                                 else f"NOTCH RECOVERY {nav_direction} WAITING HUNT")
                    else:
                        label = (f"NOTCH RECOVERY {nav_direction} "
                                 f"(right max {'--' if right_tof_max is None else f'{right_tof_max:.0f}'}mm)")
                elif nav_phase == NAV_PHASE_DONE:
                    label = "TRACK FINISHED"
                else:
                    label = "-"
                label_color = (0, 255, 0) if label in ("ALIGNED",) or label.startswith("FOLLOWING") else (
                    (0, 0, 255) if label in ("NO RAILS",) or label.startswith("STOPPING") or "approaching" in label
                    else (0, 165, 255))
                cv2.putText(overlay, f"STATUS: {label}", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)


                side_triggered = tof_side_mean is not None and tof_side_mean < NAV_TOF_SLOWDOWN_START_MM
                tof_text = (f"{side_a}:{'--' if tof_a is None else f'{tof_a:.0f}'} "
                            f"{side_b}:{'--' if tof_b is None else f'{tof_b:.0f}'} "
                            f"mean:{'--' if tof_side_mean is None else f'{tof_side_mean:.0f}'}")
                side_color = (0, 0, 255) if side_triggered else (255, 255, 0)
                cv2.putText(overlay,
                            f"TOF[{nav_direction}] {tof_text}" + (" TRIGGERED" if side_triggered else ""),
                            (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.6, side_color, 2)

                front_triggered = front_tof_mean is not None and front_tof_mean < NAV_SWITCH_FRONT_TOF_SLOWDOWN_START_MM
                front_text = "--" if front_tof_mean is None else f"{front_tof_mean:.0f}mm"
                front_color = (0, 0, 255) if front_triggered else (255, 255, 0)
                cv2.putText(overlay,
                            f"FRONT ToF:{front_text}" + (" TRIGGERED" if front_triggered else ""),
                            (10, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.6, front_color, 2)

                seq_text = sequence_label(nav_phase, lane_number, on_last_lane, tof_side_mean, near_front)
                cv2.putText(overlay, f"SEQUENCE: {seq_text}", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


                if hud_wheel_mode == "T" and nav_direction == "L":
                    left_label, left_vals = "BACK", (("BR", hud_br), ("BL", hud_bl))
                    right_label, right_vals = "FRONT", (("FR", hud_fr), ("FL", hud_fl))
                elif hud_wheel_mode == "T" and nav_direction == "R":
                    left_label, left_vals = "FRONT", (("FR", hud_fr), ("FL", hud_fl))
                    right_label, right_vals = "BACK", (("BR", hud_br), ("BL", hud_bl))
                elif hud_wheel_mode == "F":
                    left_label, left_vals = "LEFT", (("BL", hud_bl), ("FL", hud_fl))
                    right_label, right_vals = "RIGHT", (("BR", hud_br), ("FR", hud_fr))
                else:
                    left_label = right_label = None

                bottom_y1, bottom_y2 = FRAME_H - 34, FRAME_H - 10
                if left_label is not None:
                    cv2.putText(overlay, f"{left_label} PWM", (10, bottom_y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(overlay,
                                f"{left_vals[0][0]}:{left_vals[0][1]:.0f} {left_vals[1][0]}:{left_vals[1][1]:.0f}",
                                (10, bottom_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                if right_label is not None:
                    right_x = FRAME_W - 160
                    cv2.putText(overlay, f"{right_label} PWM", (right_x, bottom_y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(overlay,
                                f"{right_vals[0][0]}:{right_vals[0][1]:.0f} {right_vals[1][0]}:{right_vals[1][1]:.0f}",
                                (right_x, bottom_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                delta_text = ("Delta: 0.0deg" if last_steer_direction is None
                              else f"Delta{last_steer_direction}: {last_steer_delta_deg:.1f}deg")
                (delta_tw, _), _ = cv2.getTextSize(delta_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.putText(overlay, delta_text, ((FRAME_W - delta_tw) // 2, bottom_y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                cv2.imshow("NAVIGATION TEST", overlay)


                if hunt_reacquire_waiting_for_enter:


                    pass

                elif tof_recovery_waiting_for_enter:


                    pass

                elif notch_reset_pending:


                    if tof_reset_done.is_set():
                        notch_reset_pending = False
                        notch_waiting_for_enter = True
                        log_event("[notch] ToF reset confirmed — waiting for Enter.")

                elif notch_waiting_for_enter:


                    pass

                elif nav_phase == NAV_PHASE_FOLLOWING:
                    if (nav_direction == "R" and front_tof_mean is not None
                            and front_tof_mean < NAV_END_FRONT_TOF_THRESHOLD_MM
                            and right_tof_mean is not None
                            and right_tof_mean < NAV_END_RIGHT_TOF_THRESHOLD_MM):


                        send(ser2, "X")
                        send(ser1, "Z")
                        send(ser2, "Z")
                        nav_phase = NAV_PHASE_DONE
                        log_event(f"[nav] Track finished — front={front_tof_mean:.0f}mm, "
                                  f"right={right_tof_mean:.0f}mm both under threshold. "
                                  "Sent 'Z' to both boards. Stopped for good.")
                    elif (nav_direction == "R" and notch_armed
                            and frf is not None and frf < NAV_NOTCH_FRF_THRESHOLD_MM):


                        send(ser2, "X")
                        notch_armed = False
                        notch_used = True
                        tof_reset_done.clear()
                        tof_reset_request.set()
                        notch_reset_pending = True
                        log_event(f"[notch] FRF={frf:.0f}mm dropped under "
                                  f"{NAV_NOTCH_FRF_THRESHOLD_MM:.0f}mm while armed — stopped; "
                                  "requesting ToF reset before waiting for Enter.")
                    elif tof_side_mean is not None and tof_side_mean <= tof_threshold_mm:


                        send(ser2, "X")
                        tof_recovery_waiting_for_enter = True
                        log_event(f"[nav] ToF tripped (side_mean={tof_side_mean:.0f}mm, under "
                              f"{tof_threshold_mm:.0f}mm, rails={have_rails}) — stopped; waiting for Enter.")
                    elif not have_rails:


                        if (rail_loss_elapsed >= NAV_RAIL_LOSS_STOP_AFTER_S
                                and not rail_loss_stop_sent):
                            send(ser2, "X")
                            rail_loss_stop_sent = True
                    else:


                        dist_dir = movement_for_distance(dist_px, side_label)
                        if camera_mismatched:
                            dist_dir = flipped_direction(dist_dir)
                        if dist_dir == "ALIGNED":
                            new_steer_direction, new_steer_delta = None, 0.0
                        else:
                            new_steer_direction = dist_dir
                            new_steer_delta = compute_pulse_ms(
                                dist_px, CENTER_DEADBAND_PX, CENTER_ERROR_SATURATION_PX,
                                NAV_STEER_DELTA_MIN_DEG, NAV_STEER_DELTA_MAX_DEG)


                        if on_last_lane:
                            front_steer_bias_deg = last_lane_front_steer_bias_deg(front_tof_mean)
                            if front_steer_bias_deg > 0.0:
                                signed_steer_deg = (
                                    new_steer_delta if new_steer_direction == "B"
                                    else (-new_steer_delta if new_steer_direction == "F" else 0.0))
                                signed_steer_deg += front_steer_bias_deg
                                if signed_steer_deg > 0.0:
                                    new_steer_direction, new_steer_delta = "B", signed_steer_deg
                                elif signed_steer_deg < 0.0:
                                    new_steer_direction, new_steer_delta = "F", -signed_steer_deg
                                else:
                                    new_steer_direction, new_steer_delta = None, 0.0
                        if (new_steer_direction != last_steer_direction
                                or abs(new_steer_delta - last_steer_delta_deg) >= NAV_STEER_RESEND_DELTA_DEG):
                            if new_steer_direction is None:
                                send(ser2, "DF0")
                            else:
                                send(ser2, f"D{new_steer_direction}{new_steer_delta:.1f}")
                            last_steer_direction = new_steer_direction
                            last_steer_delta_deg = new_steer_delta


                        rotation = rotation_for_angle(angle_deg)
                        if nav_direction == "R":
                            rotation = flipped_direction(rotation)
                        pwm_delta = compute_pulse_ms(
                            angle_deg, ANGLE_DEADBAND_DEG, ANGLE_ERROR_SATURATION_DEG,
                            NAV_PWM_DELTA_MIN, NAV_PWM_DELTA_MAX)
                        base = nav_base_speed_for_tof(tof_side_mean)
                        if rotation == "CW":

                            br, fr, bl, fl = base + pwm_delta, base, base + pwm_delta, base
                        elif rotation == "CCW":

                            br, fr, bl, fl = base, base + pwm_delta, base, base + pwm_delta
                        else:
                            br, fr, bl, fl = base, base, base, base
                        motor_cmd = f"{nav_direction}{br:.0f},{fr:.0f},{bl:.0f},{fl:.0f}"
                        send(ser2, motor_cmd)
                        hud_wheel_mode = "T"
                        hud_br, hud_fr, hud_bl, hud_fl = br, fr, bl, fl

                elif nav_phase == NAV_PHASE_NOTCH_RECOVERY:


                    if ((right_tof_a is not None and right_tof_a >= NAV_NOTCH_RECOVERY_RIGHT_TOF_STOP_MM)
                            or (right_tof_b is not None and right_tof_b >= NAV_NOTCH_RECOVERY_RIGHT_TOF_STOP_MM)):
                        send(ser2, "X")


                        tof_recovery_waiting_for_enter = True
                        notch_switch_skip_flip = True
                        log_event(f"[notch] Recovery complete — BRS="
                                  f"{'--' if right_tof_a is None else f'{right_tof_a:.0f}'}mm FRS="
                                  f"{'--' if right_tof_b is None else f'{right_tof_b:.0f}'}mm, one reached "
                                  f"{NAV_NOTCH_RECOVERY_RIGHT_TOF_STOP_MM:.0f}mm — stopped; waiting for Enter "
                                  "(interlane lane switch next).")
                    elif not have_rails:
                        if (rail_loss_elapsed >= NAV_RAIL_LOSS_STOP_AFTER_S
                                and not rail_loss_stop_sent):
                            send(ser2, "X")
                            rail_loss_stop_sent = True
                    else:
                        dist_dir = movement_for_distance(dist_px, side_label)
                        if camera_mismatched:
                            dist_dir = flipped_direction(dist_dir)
                        if dist_dir == "ALIGNED":
                            new_steer_direction, new_steer_delta = None, 0.0
                        else:
                            new_steer_direction = dist_dir
                            new_steer_delta = compute_pulse_ms(
                                dist_px, CENTER_DEADBAND_PX, CENTER_ERROR_SATURATION_PX,
                                NAV_STEER_DELTA_MIN_DEG, NAV_STEER_DELTA_MAX_DEG)
                        if (new_steer_direction != last_steer_direction
                                or abs(new_steer_delta - last_steer_delta_deg) >= NAV_STEER_RESEND_DELTA_DEG):
                            if new_steer_direction is None:
                                send(ser2, "DF0")
                            else:
                                send(ser2, f"D{new_steer_direction}{new_steer_delta:.1f}")
                            last_steer_direction = new_steer_direction
                            last_steer_delta_deg = new_steer_delta


                        rotation = rotation_for_angle(angle_deg)
                        if nav_direction == "R":
                            rotation = flipped_direction(rotation)
                        pwm_delta = compute_pulse_ms(
                            angle_deg, ANGLE_DEADBAND_DEG, ANGLE_ERROR_SATURATION_DEG,
                            NAV_PWM_DELTA_MIN, NAV_PWM_DELTA_MAX)
                        base = nav_notch_recovery_speed_for_right_tof(right_tof_max)
                        if rotation == "CW":
                            br, fr, bl, fl = base + pwm_delta, base, base + pwm_delta, base
                        elif rotation == "CCW":
                            br, fr, bl, fl = base, base + pwm_delta, base, base + pwm_delta
                        else:
                            br, fr, bl, fl = base, base, base, base
                        motor_cmd = f"{nav_direction}{br:.0f},{fr:.0f},{bl:.0f},{fl:.0f}"
                        send(ser2, motor_cmd)
                        hud_wheel_mode = "T"
                        hud_br, hud_fr, hud_bl, hud_fl = br, fr, bl, fl

                elif nav_phase == NAV_PHASE_DONE:


                    pass

                elif nav_phase == NAV_PHASE_SWITCHING:


                    near_front = (
                        movement_direction == "forward" and front_tof_mean is not None
                        and front_tof_mean < NAV_SWITCH_FRONT_TOF_SLOWDOWN_START_MM)
                    if front_tof_mean is not None and front_tof_mean <= NAV_SWITCH_FRONT_TOF_STOP_MM:


                        send(ser2, "X")
                        nav_phase = NAV_PHASE_SWITCH_FRONT_ALIGN
                        front_align_pending_recheck = False
                        switch_safety_align_last_lane_trigger = True
                        continuous_forward_since = None
                        log_event(f"[safety] Front ToF mean {front_tof_mean:.0f}mm <= "
                                  f"{NAV_SWITCH_FRONT_TOF_STOP_MM:.0f}mm during interlane travel "
                                  f"(stage={travel_stage}) — stopping.")
                    elif travel_stage == "leaving_old":
                        if have_rails:
                            old_loss_since = None
                        else:
                            if old_loss_since is None:
                                old_loss_since = now
                            elif now - old_loss_since >= NAV_SWITCH_OLD_CENTERLINE_LOSS_CONFIRM_S:
                                travel_stage = "searching_new"
                                new_search_started_at = now
                                old_loss_since = None
                                log_event("[nav] Old centerline disappeared; searching for new centerline.")

                    elif travel_stage == "searching_new":
                        if entry_ok:
                            travel_stage = "tracking_new"
                            movement_direction = "forward"
                            log_event(f"[nav] New centerline detected entering from the expected side "
                                      f"(bottom_x={bottom_cx:.0f}px, dist={dist_px:.1f}px, "
                                      f"angle={angle_deg:+.2f}deg); enabling slowdown and alignment.")
                        if (not entry_ok and movement_direction == "forward" and new_search_started_at is not None and
                                now - new_search_started_at >= NAV_SWITCH_BACKWARD_SEARCH_AFTER_S):
                            movement_direction = "backward"
                            travel_stage = "backtracking_new"
                            log_event(f"[nav] No new centerline within "
                                      f"{NAV_SWITCH_BACKWARD_SEARCH_AFTER_S:.1f}s after old-line loss; "
                                      f"reversing slowly at PWM {NAV_SWITCH_BACKWARD_SEARCH_PWM}.")

                    elif travel_stage == "tracking_new":


                        if not near_front and have_rails and dist_px <= NAV_SWITCH_TARGET_CENTER_DEADBAND_PX:
                            send(ser2, "X")
                            nav_phase = NAV_PHASE_SWITCH_SAFETY_ALIGN
                            align_pending_recheck = False
                            switch_safety_align_last_lane_trigger = False
                            log_event(f"[nav] New centerline reached (dist={dist_px:.1f}px, "
                                      f"angle={angle_deg:+.2f}deg) — fine-aligning to "
                                      f"{NAV_SWITCH_FINE_ALIGN_DEADBAND_PX:.0f}px before completing switch.")

                    elif travel_stage == "backtracking_new":
                        if not near_front and entry_ok and dist_px <= NAV_SWITCH_TARGET_CENTER_DEADBAND_PX:
                            send(ser2, "X")
                            nav_phase = NAV_PHASE_SWITCH_SAFETY_ALIGN
                            align_pending_recheck = False
                            switch_safety_align_last_lane_trigger = False
                            log_event(f"[nav] New centerline found while backtracking "
                                      f"(dist={dist_px:.1f}px, angle={angle_deg:+.2f}deg) — fine-aligning to "
                                      f"{NAV_SWITCH_FINE_ALIGN_DEADBAND_PX:.0f}px before completing switch.")


                    if nav_phase == NAV_PHASE_SWITCHING:
                        if movement_direction == "backward":
                            send(ser2, f"B{NAV_SWITCH_BACKWARD_SEARCH_PWM}")
                            continuous_forward_since = None
                        else:


                            front_tof_scale = nav_switch_front_tof_pwm_scale(front_tof_mean)
                            if near_front:


                                scaled_pwm = clamp_pwm(NAV_SWITCH_FORWARD_BASE_PWM * front_tof_scale)
                                motor_cmd = f"F{scaled_pwm}"
                                hud_br = hud_fr = hud_bl = hud_fl = scaled_pwm
                            elif travel_stage == "tracking_new" and have_rails:
                                motor_cmd, _rot, _lb, _rb, _base = nav_switch_forward_pwms(
                                    angle_deg, dist_px, side_label, camera_side, True,
                                    pwm_boost=front_tof_scale)
                                hud_br = hud_fr = clamp_pwm(_base + _rb)
                                hud_bl = hud_fl = clamp_pwm(_base + _lb)
                            else:


                                scaled_pwm = NAV_SWITCH_FORWARD_BASE_PWM * front_tof_scale
                                if have_rails:
                                    motor_cmd, _br, _bl = nav_switch_angle_pwms(scaled_pwm, angle_deg)
                                    hud_br = hud_fr = _br
                                    hud_bl = hud_fl = _bl
                                else:
                                    motor_cmd = f"F{clamp_pwm(scaled_pwm)}"
                                    hud_br = hud_fr = hud_bl = hud_fl = clamp_pwm(scaled_pwm)
                            hud_wheel_mode = "F"

                            if continuous_forward_since is None:
                                continuous_forward_since = now
                            elif now - continuous_forward_since >= NAV_SWITCH_MAX_FORWARD_DURATION_S:
                                send(ser2, "X")
                                nav_phase = NAV_PHASE_SWITCH_SAFETY_ALIGN
                                align_pending_recheck = False
                                switch_safety_align_last_lane_trigger = False
                                continuous_forward_since = None
                                log_event(f"[safety] Drove F continuously for over "
                                          f"{NAV_SWITCH_MAX_FORWARD_DURATION_S:.1f}s (stage={travel_stage}) without "
                                          "reaching the target deadband; stopping.")
                            else:
                                send(ser2, motor_cmd)

                elif nav_phase == NAV_PHASE_SWITCH_FRONT_ALIGN:


                    if front_tof_mean is None:
                        send(ser2, "X")
                    else:
                        front_align_error = front_tof_mean - NAV_SWITCH_FRONT_TOF_STOP_MM
                        if abs(front_align_error) <= NAV_SWITCH_FRONT_TOF_ALIGN_DEADBAND_MM:
                            send(ser2, "X")
                            if not ALIGN_RECHECK_ENABLED or front_align_pending_recheck:
                                front_align_pending_recheck = False


                                reached_new_lane(
                                    f"Front ToF fine-aligned ({front_tof_mean:.0f}mm, "
                                    f"target={NAV_SWITCH_FRONT_TOF_STOP_MM:.0f}mm)",
                                    is_last_lane_trigger=switch_safety_align_last_lane_trigger)
                            else:
                                front_align_pending_recheck = True
                                time.sleep(ALIGN_RECHECK_DELAY_S)
                        else:
                            front_align_pending_recheck = False
                            front_align_direction = "F" if front_align_error > 0 else "B"
                            front_align_pulse_ms = compute_pulse_ms(
                                front_align_error, NAV_SWITCH_FRONT_TOF_ALIGN_DEADBAND_MM,
                                NAV_SWITCH_FRONT_TOF_ALIGN_SATURATION_MM,
                                NAV_SWITCH_FRONT_TOF_ALIGN_MIN_PULSE_MS, NAV_SWITCH_FRONT_TOF_ALIGN_MAX_PULSE_MS)
                            send(ser2, f"{front_align_direction}{NAV_SWITCH_FRONT_TOF_ALIGN_SPEED}")
                            time.sleep(front_align_pulse_ms / 1000.0)
                            send(ser2, "X")

                elif nav_phase == NAV_PHASE_SWITCH_SAFETY_ALIGN:


                    if not have_rails:
                        align_pending_recheck = False
                    elif dist_px <= NAV_SWITCH_FINE_ALIGN_DEADBAND_PX:
                        send(ser2, "X")
                        if not ALIGN_RECHECK_ENABLED or align_pending_recheck:
                            reached_new_lane(
                                f"Distance fine-aligned (dist={dist_px:.2f}px)",
                                is_last_lane_trigger=switch_safety_align_last_lane_trigger)
                            align_pending_recheck = False
                        else:
                            align_pending_recheck = True
                            time.sleep(ALIGN_RECHECK_DELAY_S)
                    else:
                        align_pending_recheck = False
                        direction = movement_for_distance(dist_px, effective_side_label(side_label, camera_side))
                        pulse_ms = compute_pulse_ms(
                            dist_px, NAV_SWITCH_FINE_ALIGN_DEADBAND_PX, CENTER_ERROR_SATURATION_PX,
                            CENTER_MIN_PULSE_MS, CENTER_MAX_PULSE_MS)
                        send(ser2, f"{direction}{CENTER_MAX_SPEED}")
                        time.sleep(pulse_ms / 1000.0)
                        send(ser2, "X")

                elif nav_phase == NAV_PHASE_SWITCH_STOPPED:


                    pass

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log_event("\n[info] stopped by user.")
                    break

                if AUTONOMOUS_MODE and key not in ENTER_KEY_CODES:


                    if hunt_reacquire_waiting_for_enter and have_rails and nav_phase != NAV_PHASE_IDLE:
                        key = ENTER_KEY_CODES[0]
                    elif notch_waiting_for_enter:
                        key = ENTER_KEY_CODES[0]
                    elif tof_recovery_waiting_for_enter:
                        key = ENTER_KEY_CODES[0]
                    elif nav_phase == NAV_PHASE_SWITCH_STOPPED and have_rails:
                        key = ENTER_KEY_CODES[0]

                if key in ENTER_KEY_CODES:
                    if hunt_reacquire_waiting_for_enter and have_rails:
                        hunt_reacquire_waiting_for_enter = False
                        no_rails_since = None
                        log_event("[hunt] Enter pressed after rail reacquire — resuming current sequence.")

                    if notch_waiting_for_enter:
                        notch_waiting_for_enter = False
                        nav_direction = "R" if nav_direction == "L" else "L"
                        no_rails_since = None
                        rail_loss_stop_sent = False
                        last_steer_direction = None
                        last_steer_delta_deg = 0.0
                        nav_phase = NAV_PHASE_NOTCH_RECOVERY
                        log_event(f"[notch] Enter pressed after NOTCH trip — driving {nav_direction} "
                                  f"until right-side ToF mean reaches {NAV_NOTCH_RECOVERY_RIGHT_TOF_STOP_MM:.0f}mm.")

                    if tof_recovery_waiting_for_enter:
                        tof_recovery_waiting_for_enter = False
                        send_and_settle(ser2, "I")
                        nav_phase = NAV_PHASE_SWITCHING
                        travel_stage = "leaving_old"
                        movement_direction = "forward"
                        old_loss_since = None
                        new_search_started_at = None
                        continuous_forward_since = None
                        log_event("[nav] Enter pressed after ToF trip — leaving current lane at "
                                  f"F{NAV_SWITCH_FORWARD_BASE_PWM}; old centerline corrections disabled.")
                    elif nav_phase == NAV_PHASE_SWITCH_STOPPED and have_rails:
                        nav_phase = NAV_PHASE_FOLLOWING
                        arm_notch_if_needed()
                        log_event(f"[nav] Enter pressed after lane switch — resuming line-following "
                                  f"(direction={nav_direction}).")
                    elif nav_phase == NAV_PHASE_SWITCH_STOPPED:
                        log_event("[nav] Enter pressed after lane switch, but no rails are visible; holding.")
                    elif nav_phase == NAV_PHASE_IDLE:


                        SERVO_DONE_EVENTS[8].clear()
                        SERVO_DONE_EVENTS[9].clear()
                        send(ser2, f"8 {CAMERA_LEFT_ANGLE_INITIAL}")
                        send(ser2, f"9 {SERVO_9_STRAIGHT_ANGLE}")
                        wait_for_servos_settle((8, 9), "startup camera ready pose")
                        nav_direction = initial_nav_direction(camera_side)
                        send_and_settle(ser2, "T")
                        nav_phase = NAV_PHASE_FOLLOWING
                        arm_notch_if_needed()
                        log_event(f"[init] Enter pressed -> servo 8 -> {CAMERA_LEFT_ANGLE_INITIAL} (left), "
                                  f"servo 9 -> {SERVO_9_STRAIGHT_ANGLE} (straight), both confirmed -> "
                                  f"line-following (direction={nav_direction}).")

    except KeyboardInterrupt:
        log_event("\n[info] stopped by user.")
    finally:
        send(ser2, "X")
        stop_event.set()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        ser1.close()
        ser2.close()
        if processor:
            del processor.detector
        gc.collect()
        _CUDA_CONTEXT.pop()
        _CUDA_DEVICE.retain_primary_context().detach()
        log_event("\nCleanup complete.")
        log_event(f"RUN END {datetime.now().isoformat(timespec='seconds')}")
        log_event("=" * 60)


if __name__ == "__main__":
    main()
