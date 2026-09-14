import os
import sys
import time

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda


ENGINE_PATH = os.environ.get("ASABE_CROP_ENGINE", os.path.join(os.path.dirname(__file__), "crop_detector.engine"))
BOTTOM_CAM_DEVICE = "/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_9342D410-video-index0"
IMG_SIZE    = 640
CONF_THRESH = 0.30
IOU_THRESH  = 0.40

CLASS_NAMES = {0: "gCrop", 1: "platform", 2: "ycrop", 3: "servo"}
COLOR_MAP   = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255), 3: (255, 0, 255)}

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


def draw(frame, detections, fps):
    h, w = frame.shape[:2]
    for cls_id, conf, x1, y1, x2, y2 in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        area_px = (x2 - x1) * (y2 - y1)
        color = COLOR_MAP.get(cls_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls_id, '?')} conf={conf:.2f} area={area_px}px"
        lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lsz[1] - 4), (x1 + lsz[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    counts = {cid: sum(1 for d in detections if d[0] == cid) for cid in CLASS_NAMES}
    count_text = "  ".join(f"{CLASS_NAMES[c]}: {counts[c]}" for c in sorted(CLASS_NAMES))
    tsz, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, count_text, (w - tsz[0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


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

    print("=" * 60)
    print(" BOTTOM CAM  |  crop_detector.engine  |  area + conf viewer")
    print(" classes: 0=gCrop  1=platform  2=ycrop  3=servo")
    print(" 'q' quits. Ctrl+C also quits.")
    print("=" * 60)

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

            draw(frame, detections, fps)
            cv2.imshow("Bottom Cam — crop_detector area/conf", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        del detector
        import gc; gc.collect()
        _CTX.pop()
        _DEV.retain_primary_context().detach()

    return 0


if __name__ == "__main__":
    sys.exit(main())
