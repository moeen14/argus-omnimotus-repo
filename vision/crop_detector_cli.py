import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import time
import sys


ENGINE_PATH = "crop_detector.engine"
IMG_SIZE    = 640
CONF_THRESH = 0.75
IOU_THRESH  = 0.75
ANGLE_GAUGE_MAX = 200
PRINT_EVERY = 1

CLASS_NAMES = {0: "gCrop", 1: "platform", 2: "ycrop", 3: "servo"}
CLASS_COLORS_TXT = {0: "\033[92m", 1: "\033[94m", 2: "\033[93m", 3: "\033[95m"}
RESET = "\033[0m"

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

    mask = confs > conf_thresh
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


def compute_platform_error(detections, frame_w):


    best = None
    for cls_id, conf, x1, y1, x2, y2 in detections:
        if cls_id != 1:
            continue
        if best is None or conf > best[0]:
            best = (conf, x1, y1, x2, y2)
    if best is None:
        return None
    _, x1, y1, x2, y2 = best
    platform_cx = (x1 + x2) / 2.0
    img_cx = frame_w / 2.0
    return platform_cx - img_cx, platform_cx


def draw_detections(frame, detections, fps):

    h, w = frame.shape[:2]


    for cls_id, conf, x1, y1, x2, y2 in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


        class_name = CLASS_NAMES.get(cls_id, "unknown")
        color_map = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255), 3: (255, 0, 255)}
        color = color_map.get(cls_id, (255, 255, 255))


        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


        label = f"{class_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    counts = {0: 0, 1: 0, 2: 0}
    for cls_id, conf, *_ in detections:
        counts[cls_id] = counts.get(cls_id, 0) + 1

    count_text = "  ".join([f"{CLASS_NAMES[cid]}: {counts[cid]}" for cid in sorted(CLASS_NAMES.keys())])
    text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, count_text, (w - text_size[0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


    img_cx = w // 2


    for y in range(0, h, 20):
        cv2.line(frame, (img_cx, y), (img_cx, min(y + 10, h)), (255, 255, 255), 1)
    cv2.putText(frame, "IMG CENTER", (img_cx - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    error_info = compute_platform_error(detections, w)
    if error_info is not None:
        error, platform_cx = error_info
        platform_cx_int = int(platform_cx)


        color = (0, 255, 0) if abs(error) < 50 else (0, 165, 255) if abs(error) < 100 else (0, 0, 255)
        for y in range(0, h, 20):
            cv2.line(frame, (platform_cx_int, y), (platform_cx_int, min(y + 10, h)), color, 2)
        cv2.putText(frame, "PLATFORM", (platform_cx_int - 45, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        err_text = f"Platform Error: {error:+6.1f}px"
        cv2.putText(frame, err_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(frame, "Platform Error: NO TARGET", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame


def live_detect():
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
    print(" CROP DETECTOR (YOLO11n TRT) - LIVE VIDEO")
    print(" classes: 0=gCrop (green) 1=platform (red) 2=ycrop (yellow) 3=servo (magenta)")
    print(" Press 'q' or Ctrl+C to quit.")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[warn] frame grab failed, stopping.")
                break
            t0 = time.time()

            inp, r, pad = preprocess(frame)
            raw = detector.infer(inp)
            raw = raw.reshape(detector.output_shape)
            detections = postprocess(raw, r, pad, frame.shape)

            fps = 1.0 / max(1e-6, time.time() - t0)


            frame_with_detections = draw_detections(frame, detections, fps)
            cv2.imshow("CROP DETECTOR (YOLO11n TRT)", frame_with_detections)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[info] stopped by user.")
                break

    except KeyboardInterrupt:
        print("\n[info] stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        del detector
        import gc
        gc.collect()
        _CUDA_CONTEXT.pop()
        _CUDA_DEVICE.retain_primary_context().detach()


if __name__ == "__main__":
    live_detect()
