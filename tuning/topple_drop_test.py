import os
import time
import gc
from pathlib import Path
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

OVERHEAD_CAMERA_DEVICE = os.environ.get(
    "ASABE_OVERHEAD_CAMERA_DEVICE",
    "/dev/v4l/by-id/usb-046d_0825_3D19ADE0-video-index0",
)
FRAME_W, FRAME_H = 640, 480

ENGINE_PATH = os.environ.get("ASABE_PLATFORM_ENGINE",
                             str(Path(__file__).resolve().parent / "highcam_platform.engine"))
IMG_SIZE = 640
IOU_THRESH = 0.75
CLASS_NAMES = {0: "ball", 1: "gcrop", 2: "platform", 3: "ycrop"}
CONF_THRESH = {0: 0.30, 1: 0.40, 2: 0.85, 3: 0.85}
CLASS_BOX_COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 255, 255)}


ARUCO_DICT_NAMES = ["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000"]
MARKER_BOX_COLOR = (0, 200, 255)


MIN_MARKER_SIDE_PX = 10.0

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
            raise RuntimeError(f"Failed to deserialize engine from {engine_path}. "
                             "Check file exists, is not corrupted, and is compatible with this TensorRT/device.")
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
        print(f"[init] input {self.input_name}: {self.input_shape}")
        print(f"[init] output {self.output_name}: {self.output_shape}")

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
    pred = output[0]
    pred = pred.transpose(1, 0)

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(class_scores)), class_ids]

    per_anchor_thresh = np.array([conf_thresh[c] for c in class_ids])
    mask = confs > per_anchor_thresh
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
            results.append((int(cls_id), float(cls_confs[k]),
                            float(x1), float(y1), float(x2), float(y2)))
    return results


def build_detector_params():


    params = cv2.aruco.DetectorParameters_create()
    params.minMarkerPerimeterRate = 0.01
    params.polygonalApproxAccuracyRate = 0.06
    params.maxErroneousBitsInBorderRate = 0.5
    params.errorCorrectionRate = 0.8
    params.minOtsuStdDev = 3.0
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.1
    return params


def build_aruco_dictionaries():
    params = build_detector_params()
    return [(name, cv2.aruco.Dictionary_get(getattr(cv2.aruco, name)), params)
            for name in ARUCO_DICT_NAMES]


def detect_marker(gray, dictionaries):


    for name, dictionary, params in dictionaries:
        corners_list, ids, _rejected = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=params)
        if ids is None:
            continue
        for corners, marker_id in zip(corners_list, ids.flatten()):
            corners = corners.reshape(4, 2)
            side_lengths = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)
            if side_lengths.min() < MIN_MARKER_SIDE_PX:
                continue
            return name, int(marker_id), corners
    return None


def open_camera(preferred):
    candidates = [preferred] + [i for i in range(4)]
    for source in candidates:
        cap = cv2.VideoCapture(source)
        if cap.isOpened() and cap.read()[0]:
            print(f"[init] Camera opened: {source}")
            return cap
        cap.release()
    return None


def draw_overlay(frame, marker_detection, platform_detections, fps):
    if marker_detection is not None:
        name, marker_id, corners = marker_detection
        pts = corners.astype(int)
        cv2.polylines(frame, [pts], True, MARKER_BOX_COLOR, 2)
        label = f"ArUco id={marker_id}"
        tl = pts[0]
        cv2.putText(frame, label, (tl[0], max(0, tl[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, MARKER_BOX_COLOR, 1)

    for cls_id, conf, x1, y1, x2, y2 in platform_detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = CLASS_BOX_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls_id, 'unknown')} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return frame


def main():
    detector = TRTDetector(ENGINE_PATH)

    cap = open_camera(OVERHEAD_CAMERA_DEVICE)
    if cap is None:
        print("Error: could not open the overhead camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    dictionaries = build_aruco_dictionaries()


    dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for _ in range(3):
        inp, r, pad = preprocess(dummy)
        detector.infer(inp)
    print("[init] TRT engine warmed up")

    print("=" * 60)
    print(" TOPPLE/DROP TEST — ArUco marker + platform/ball/crop detection")
    print(" classes: ball, gcrop, platform, ycrop, ArUco marker")
    print(" Press 'q' to quit.")
    print("=" * 60)

    fps = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[warn] frame grab failed, stopping.")
                break
            t0 = time.time()

            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker_detection = detect_marker(gray, dictionaries)

            inp, r, pad = preprocess(frame)
            raw = detector.infer(inp)
            raw = raw.reshape(detector.output_shape)
            platform_detections = postprocess(raw, r, pad, frame.shape)

            fps = 1.0 / max(1e-6, time.time() - t0)
            overlay = draw_overlay(frame, marker_detection, platform_detections, fps)
            cv2.imshow("Topple/Drop Test", overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[info] stopped by user.")
                break
    except KeyboardInterrupt:
        print("\n[info] stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        del detector
        gc.collect()
        _CUDA_CONTEXT.pop()
        _CUDA_DEVICE.retain_primary_context().detach()


if __name__ == "__main__":
    main()
