import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

ONNX_PATH = "crop_detector.onnx"
IMG_SIZE = 640
CONF_THRESH = 0.5
IOU_THRESH = 0.5
ID_COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255), 3: (255, 0, 255)}


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
    padded, r, pad = letterbox(frame)
    img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(img), r, pad


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

    mask = confs > CONF_THRESH
    if not np.any(mask):
        return []
    boxes_xywh, class_ids, confs = boxes_xywh[mask], class_ids[mask], confs[mask]

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
        keep = nms(boxes_xyxy[cls_mask], confs[cls_mask], IOU_THRESH)
        cls_boxes = boxes_xyxy[cls_mask]
        cls_confs = confs[cls_mask]
        for k in keep:
            x1, y1, x2, y2 = cls_boxes[k]
            results.append((int(cls_id), float(cls_confs[k]), float(x1), float(y1), float(x2), float(y2)))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=None, help="Camera index; auto-detects 0/1/2/3 if omitted.")
    args = parser.parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(ONNX_PATH, providers=providers)
    input_name = sess.get_inputs()[0].name
    print(f"[init] onnxruntime providers in use: {sess.get_providers()}")

    cam_indices = [args.cam] if args.cam is not None else [0, 1, 2, 3]
    cap = None
    for idx in cam_indices:
        c = cv2.VideoCapture(idx)
        if c.isOpened() and c.read()[0]:
            print(f"[init] camera on index {idx}")
            cap = c
            break
        c.release()
    if cap is None:
        print("Error: could not open any camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Point the camera at gcrop / ycrop / platform and read the id= label.")
    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[warn] frame grab failed")
                break
            t0 = time.time()
            inp, r, pad = preprocess(frame)
            raw = sess.run(None, {input_name: inp})[0]
            dets = postprocess(raw, r, pad, frame.shape)
            fps = 1.0 / max(1e-6, time.time() - t0)

            for cls_id, conf, x1, y1, x2, y2 in dets:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                color = ID_COLORS.get(cls_id, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"id={cls_id} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("verify crop_detector.onnx classes", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
