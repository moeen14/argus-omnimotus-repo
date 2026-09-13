import cv2
import numpy as np
import time

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


ENGINE_PATH = 'lanesegyv8.engine'
CONF_THRES = 0.25
IOU_THRES = 0.45
INPUT_SIZE = 640


CAM_INDEX = 0
CAP_W, CAP_H = 640, 480


class TRTSeg:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()


        self.inputs, self.outputs = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            size = int(trt.volume(shape))
            host = cuda.pagelocked_empty(size, dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            entry = {'name': name, 'shape': shape, 'dtype': dtype,
                     'host': host, 'dev': dev}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = entry
            else:
                self.outputs[name] = entry

        if len(self.inputs) != 1:
            raise RuntimeError(f"Expected 1 input, got {len(self.inputs)}")
        self.in_name = next(iter(self.inputs))
        self.in_shape = self.inputs[self.in_name]['shape']


        self.proto_name, self.det_name = None, None
        for name, e in self.outputs.items():
            if len(e['shape']) == 4:
                self.proto_name = name
            else:
                self.det_name = name
        if self.proto_name is None or self.det_name is None:
            raise RuntimeError(f"Could not identify seg outputs: "
                               f"{[(n, e['shape']) for n, e in self.outputs.items()]}")


        dummy = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        self.infer(dummy)

    @staticmethod
    def letterbox(img, new_size=640, color=(114, 114, 114)):
        h, w = img.shape[:2]
        r = min(new_size / w, new_size / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = (new_size - nw) / 2.0, (new_size - nh) / 2.0
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        return out, r, (left, top, nw, nh)

    def infer(self, frame):

        lb, r, pad = self.letterbox(frame, INPUT_SIZE)
        blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]
        blob = np.ascontiguousarray(blob)

        inp = self.inputs[self.in_name]
        np.copyto(inp['host'], blob.ravel().astype(inp['dtype'], copy=False))
        cuda.memcpy_htod_async(inp['dev'], inp['host'], self.stream)

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for e in self.outputs.values():
            cuda.memcpy_dtoh_async(e['host'], e['dev'], self.stream)
        self.stream.synchronize()

        det = self.outputs[self.det_name]['host'].reshape(
            self.outputs[self.det_name]['shape']).astype(np.float32)
        proto = self.outputs[self.proto_name]['host'].reshape(
            self.outputs[self.proto_name]['shape']).astype(np.float32)
        return det, proto, r, pad

    def get_binary_mask(self, frame):

        h, w = frame.shape[:2]
        det, proto, r, pad = self.infer(frame)
        left, top, nw, nh = pad

        proto = proto[0]
        nm, mh, mw = proto.shape
        pred = det[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.transpose(1, 0)
        nc = pred.shape[1] - 4 - nm

        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:4 + nc]
        coeffs = pred[:, 4 + nc:]
        conf = cls_scores.max(axis=1)

        keep = conf > CONF_THRES
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        if not np.any(keep):
            return binary_mask

        boxes_xywh = boxes_xywh[keep]
        coeffs = coeffs[keep]
        conf = conf[keep]


        xyxy = np.empty_like(boxes_xywh)
        xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        nms_boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
                     for b in xyxy]
        idxs = cv2.dnn.NMSBoxes(nms_boxes, conf.tolist(), CONF_THRES, IOU_THRES)
        if len(idxs) == 0:
            return binary_mask
        idxs = np.array(idxs).flatten()

        xyxy = xyxy[idxs]
        coeffs = coeffs[idxs]


        masks = coeffs @ proto.reshape(nm, -1)
        masks = 1.0 / (1.0 + np.exp(-masks))
        masks = masks.reshape(-1, mh, mw)


        sx, sy = mw / INPUT_SIZE, mh / INPUT_SIZE
        union = np.zeros((mh, mw), dtype=bool)
        for k in range(masks.shape[0]):
            x1 = int(np.clip(xyxy[k, 0] * sx, 0, mw))
            y1 = int(np.clip(xyxy[k, 1] * sy, 0, mh))
            x2 = int(np.clip(xyxy[k, 2] * sx, 0, mw))
            y2 = int(np.clip(xyxy[k, 3] * sy, 0, mh))
            m = masks[k] > 0.5
            cropped = np.zeros_like(m)
            cropped[y1:y2, x1:x2] = m[y1:y2, x1:x2]
            union |= cropped


        union_u8 = (union.astype(np.uint8) * 255)
        union_640 = cv2.resize(union_u8, (INPUT_SIZE, INPUT_SIZE),
                               interpolation=cv2.INTER_LINEAR)
        content = union_640[top:top + nh, left:left + nw]
        binary_mask = cv2.resize(content, (w, h), interpolation=cv2.INTER_NEAREST)
        binary_mask = (binary_mask > 127).astype(np.uint8) * 255
        return binary_mask


SRC_PTS = np.array([[73, 323], [467, 311], [348, 356], [97, 357]], dtype=np.float32)
SRC_CM = np.array([[0, 10], [30, 30], [16, -14.5], [6, -14.5]], dtype=np.float32)
PIXELS_PER_CM, ORIGIN_X_OFFSET, ORIGIN_Y_OFFSET, THRESHOLD_Y = 10.0, 150.0, 600.0, 290
FRAME_W, FRAME_H = 640, 480


DEFAULT_RAIL_WIDTH_BEV = 250
DEFAULT_SLOPE_DIFF = 0.0
WIDTH_EMA_ALPHA = 0.1

last_known_rail_width = DEFAULT_RAIL_WIDTH_BEV
last_known_slope_diff = DEFAULT_SLOPE_DIFF


def build_bev_transform(frame_w, frame_h):
    dst = np.zeros_like(SRC_CM)
    dst[:, 0] = SRC_CM[:, 0] * PIXELS_PER_CM + ORIGIN_X_OFFSET
    dst[:, 1] = ORIGIN_Y_OFFSET - SRC_CM[:, 1] * PIXELS_PER_CM
    H, _ = cv2.findHomography(SRC_PTS, dst)
    corners = np.float32([[0, THRESHOLD_Y], [frame_w, THRESHOLD_Y], [frame_w, frame_h], [0, frame_h]]).reshape(-1, 1, 2)
    wc = cv2.perspectiveTransform(corners, H)
    xmin, ymin = np.int32(wc.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(wc.max(axis=0).ravel() + 0.5)
    padding_x = 400
    xmin -= padding_x
    xmax += padding_x
    bev_w, bev_h = xmax - xmin, ymax - ymin
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
    return T.dot(H).astype(np.float32), int(bev_w), int(bev_h), xmin, ymin


H_IMG2BEV, BEV_W, BEV_H, BEV_XMIN, BEV_YMIN = build_bev_transform(FRAME_W, FRAME_H)
H_BEV2IMG = np.linalg.inv(H_IMG2BEV)


def widest_run_midpoint(row_bool):
    idx = np.where(row_bool)[0]
    if idx.size == 0:
        return None
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)
    best = max(runs, key=len)
    return (best[0] + best[-1]) // 2


def get_bev_skeleton(bev_mask):
    left_pts, right_pts = [], []
    h, w = bev_mask.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bev_mask, connectivity=8)
    valid_components = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 150:
            valid_components.append((i, centroids[i, 0]))
    valid_components.sort(key=lambda x: x[1])

    if len(valid_components) >= 2:
        left_id, right_id = valid_components[0][0], valid_components[-1][0]
        for y in range(h):
            l_mid = widest_run_midpoint(labels[y] == left_id)
            if l_mid is not None:
                left_pts.append([l_mid, y])
            r_mid = widest_run_midpoint(labels[y] == right_id)
            if r_mid is not None:
                right_pts.append([r_mid, y])
    elif len(valid_components) == 1:
        comp_id, comp_x = valid_components[0]
        bev_robot_x = ORIGIN_X_OFFSET - BEV_XMIN
        for y in range(h):
            mid = widest_run_midpoint(labels[y] == comp_id)
            if mid is not None:
                if comp_x < bev_robot_x:
                    left_pts.append([mid, y])
                else:
                    right_pts.append([mid, y])
    return np.array(left_pts), np.array(right_pts)


def fit_line_stable(pts):
    if pts is None or len(pts) < 5:
        return None, None
    [vx, vy, x0, y0] = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if abs(vy) < 1e-6:
        return None, None
    m = vx / vy
    b = x0 - m * y0
    return m, b


seg = TRTSeg(ENGINE_PATH)

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError(f"Could not open USB camera at index {CAM_INDEX}")

print("Live navigation viewer running. Press 'q' to quit.")
prev_t = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed, retrying...")
        continue
    if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    h, w = frame.shape[:2]
    img_center_x = w // 2


    binary_mask = seg.get_binary_mask(frame)

    bev_mask = cv2.warpPerspective(binary_mask, H_IMG2BEV, (BEV_W, BEV_H))
    left_bev_pts, right_bev_pts = get_bev_skeleton(bev_mask)


    display_1 = frame.copy()
    overlay = display_1.copy()
    overlay[binary_mask > 0] = (0, 200, 0)
    display_1 = cv2.addWeighted(overlay, 0.45, display_1, 0.55, 0)
    display_2 = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    display_3 = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)

    m1, b1 = fit_line_stable(left_bev_pts)
    m2, b2 = fit_line_stable(right_bev_pts)


    if m1 is not None and m2 is not None:
        current_width = b2 - b1
        current_slope_diff = m2 - m1
        if abs(current_width) > 50:
            last_known_rail_width = (WIDTH_EMA_ALPHA * current_width) + (1 - WIDTH_EMA_ALPHA) * last_known_rail_width
            last_known_slope_diff = (WIDTH_EMA_ALPHA * current_slope_diff) + (1 - WIDTH_EMA_ALPHA) * last_known_slope_diff
    elif m1 is not None:
        m2 = m1 + last_known_slope_diff
        b2 = b1 + last_known_rail_width
    elif m2 is not None:
        m1 = m2 - last_known_slope_diff
        b1 = b2 - last_known_rail_width

    if m1 is not None and m2 is not None:
        limit_pts_img = np.array([[[img_center_x, THRESHOLD_Y]], [[img_center_x, FRAME_H]]], dtype=np.float32)
        limit_pts_bev = cv2.perspectiveTransform(limit_pts_img, H_IMG2BEV)
        y_far_bev = limit_pts_bev[0][0][1]
        y_near_bev = limit_pts_bev[1][0][1]

        def project_line_safe(m, b, y_start, y_end, H_inv):
            ys = np.linspace(y_start, y_end, 20)
            xs = m * ys + b
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.float32)
            pts_back = cv2.perspectiveTransform(pts, H_inv)
            return pts_back.astype(np.int32)

        pl = project_line_safe(m1, b1, y_far_bev, y_near_bev, H_BEV2IMG)
        pr = project_line_safe(m2, b2, y_far_bev, y_near_bev, H_BEV2IMG)

        mc, bc = (m1 + m2) / 2.0, (b1 + b2) / 2.0
        pc = project_line_safe(mc, bc, y_far_bev, y_near_bev, H_BEV2IMG)

        cv2.polylines(display_1, [pl], False, (255, 0, 0), 3)
        cv2.polylines(display_1, [pr], False, (255, 255, 0), 3)
        cv2.polylines(display_1, [pc], False, (255, 255, 255), 4)

        cv2.polylines(display_2, [pc], False, (255, 0, 255), 3)

        def draw_bev(canvas, m, b, color):
            x_top = int(np.clip(m * 0 + b, -1000, 10000))
            x_bot = int(np.clip(m * BEV_H + b, -1000, 10000))
            cv2.line(canvas, (x_top, 0), (x_bot, BEV_H), color, 2)

        draw_bev(display_3, m1, b1, (255, 0, 0))
        draw_bev(display_3, m2, b2, (255, 255, 0))
        draw_bev(display_3, mc, bc, (255, 0, 255))


    target_h = 480

    def resize_to_h(img, target_h):
        h, w = img.shape[:2]
        target_w = int(w * (target_h / h))
        return cv2.resize(img, (target_w, target_h))

    p1 = resize_to_h(display_1, target_h)
    p2 = resize_to_h(display_2, target_h)
    p3 = resize_to_h(display_3, target_h)
    unified = np.hstack((p1, p2, p3))

    now = time.time()
    fps = 1.0 / max(now - prev_t, 1e-6)
    prev_t = now
    cv2.putText(unified, f"FPS: {fps:.1f}", (20, 30), 0, 0.7, (255, 255, 255), 2)
    cv2.imshow("Unified Navigation Viewer", unified)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
