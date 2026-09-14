import time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda


ENGINE_PATH = "lanesegyv8.engine"
IMG_SIZE    = 640
NUM_CLASSES = 1
NUM_MASKS   = 32
CONF_THRESH = 0.35
IOU_THRESH  = 0.45
MASK_THRESH = 0.5


SRC_PTS = np.array([[73, 323], [467, 311], [348, 356], [97, 357]], dtype=np.float32)
SRC_CM = np.array([[0, 10], [30, 30], [16, -14.5], [6, -14.5]], dtype=np.float32)
PIXELS_PER_CM, ORIGIN_X_OFFSET, ORIGIN_Y_OFFSET, THRESHOLD_Y = 10.0, 150.0, 600.0, 290


WIDTH_EMA_ALPHA = 0.1
STEER_EMA_ALPHA = 0.6
DEFAULT_RAIL_WIDTH_BEV = 250
MIN_COMPONENT_AREA = 150


LEFT_COLOR  = (255, 0, 0)
RIGHT_COLOR = (255, 255, 0)
VP_COLOR    = (0, 0, 255)
PATH_COLOR  = (0, 255, 0)
CENTER_COLOR = (255, 0, 255)

cuda.init()
_CUDA_DEVICE  = cuda.Device(0)
_CUDA_CONTEXT = _CUDA_DEVICE.retain_primary_context()
_CUDA_CONTEXT.push()


class TRTSegDetector:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
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
        in_size = int(np.prod(self.input_shape)) * np.dtype(np.float32).itemsize
        self.d_input = cuda.mem_alloc(in_size)
        self.context.set_tensor_address(self.input_name, int(self.d_input))

        self.out = {}
        for name in self.out_names:
            shp = tuple(self.engine.get_tensor_shape(name))
            h_buf = cuda.pagelocked_empty(shp, dtype=np.float32)
            d_buf = cuda.mem_alloc(int(np.prod(shp)) * np.dtype(np.float32).itemsize)
            self.context.set_tensor_address(name, int(d_buf))
            role = "proto" if len(shp) == 4 else "det"
            self.out[role] = dict(name=name, shape=shp, host=h_buf, dev=d_buf)

    def infer(self, input_array):
        input_array = np.ascontiguousarray(input_array, dtype=np.float32)
        cuda.memcpy_htod_async(self.d_input, input_array, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for role in ("det", "proto"):
            o = self.out[role]
            cuda.memcpy_dtoh_async(o["host"], o["dev"], self.stream)
        self.stream.synchronize()
        det = np.array(self.out["det"]["host"]).reshape(self.out["det"]["shape"])
        proto = np.array(self.out["proto"]["host"]).reshape(self.out["proto"]["shape"])
        return det, proto


def letterbox(frame, size=640):
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[top:top + nh, left:left + nw] = resized
    return padded, r, (left, top)

def preprocess(frame):
    padded, r, pad = letterbox(frame, IMG_SIZE)
    img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]
    return img, r, pad

def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0: return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        ww = np.maximum(0, xx2 - xx1); hh = np.maximum(0, yy2 - yy1)
        inter = ww * hh
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def seg_postprocess(det, proto, r, pad, orig_shape):
    H, W = orig_shape[:2]
    p = det[0].transpose(1, 0)
    boxes_xywh = p[:, :4]
    cls_scores = p[:, 4:4 + NUM_CLASSES]
    coeffs     = p[:, 4 + NUM_CLASSES:]
    conf = cls_scores.max(axis=1)
    keep = conf > CONF_THRESH
    if not np.any(keep): return []
    boxes_xywh, coeffs, conf = boxes_xywh[keep], coeffs[keep], conf[keep]
    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    idx = nms(boxes, conf, IOU_THRESH)
    if len(idx) == 0: return []
    boxes, coeffs = boxes[idx], coeffs[idx]
    protos = proto[0]
    pc, ph, pw = protos.shape
    protos_flat = protos.reshape(pc, ph * pw)
    left, top = pad
    masks = []
    for i in range(len(boxes)):
        m = sigmoid(coeffs[i] @ protos_flat).reshape(ph, pw)
        m640 = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        nh, nw = IMG_SIZE - 2 * top, IMG_SIZE - 2 * left
        m_unpad = m640[top:top + nh, left:left + nw] if (nh > 0 and nw > 0) else m640
        m_full = cv2.resize(m_unpad, (W, H), interpolation=cv2.INTER_LINEAR)
        masks.append(m_full > MASK_THRESH)
    return masks


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
    xmin -= padding_x; xmax += padding_x
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
    return T.dot(H).astype(np.float32), int(xmax - xmin), int(ymax - ymin), xmin

H_IMG2BEV, BEV_W, BEV_H, BEV_XMIN = build_bev_transform(640, 480)
H_BEV2IMG = np.linalg.inv(H_IMG2BEV)

def widest_run_midpoint(row_bool):
    idx = np.where(row_bool)[0]
    if idx.size == 0: return None
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)
    best = max(runs, key=len)
    return (best[0] + best[-1]) // 2

def get_bev_skeleton(bev_mask):
    left_pts, right_pts = [], []
    h, w = bev_mask.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bev_mask, connectivity=8)
    valid = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > MIN_COMPONENT_AREA:
            valid.append((i, centroids[i, 0]))
    valid.sort(key=lambda x: x[1])
    if len(valid) >= 2:
        left_id, right_id = valid[0][0], valid[-1][0]
        for y in range(h):
            lm = widest_run_midpoint(labels[y] == left_id)
            if lm is not None: left_pts.append([lm, y])
            rm = widest_run_midpoint(labels[y] == right_id)
            if rm is not None: right_pts.append([rm, y])
    elif len(valid) == 1:
        comp_id, comp_x = valid[0]
        bev_robot_x = ORIGIN_X_OFFSET - BEV_XMIN
        for y in range(h):
            m = widest_run_midpoint(labels[y] == comp_id)
            if m is not None:
                if comp_x < bev_robot_x: left_pts.append([m, y])
                else: right_pts.append([m, y])
    return np.array(left_pts), np.array(right_pts)

def fit_line_huber(pts):
    if pts is None or len(pts) < 5: return None, None
    [vx, vy, x0, y0] = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if abs(vy) < 1e-6: return None, None
    m = vx / vy
    b = x0 - m * y0
    return m, b

def project_path(m, b, H_inv, h_start=480, h_end=THRESHOLD_Y+10):
    ys = np.linspace(h_start, h_end, 20)
    xs = m * ys + b
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, H_inv).astype(np.int32)


def main():
    detector = TRTSegDetector(ENGINE_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    l_width, l_slope = DEFAULT_RAIL_WIDTH_BEV, 0.0
    s_steer = 0.0
    t_prev = time.time()

    print("[info] Rail Navigation (TensorRT + Advanced BEV) started.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            cx_img = w // 2


            inp, r, pad = preprocess(frame)
            det, proto = detector.infer(inp)
            masks = seg_postprocess(det, proto, r, pad, frame.shape)


            binary_mask = np.zeros((h, w), dtype=np.uint8)
            for m in masks:
                binary_mask[m] = 255

            bev_mask = cv2.warpPerspective(binary_mask, H_IMG2BEV, (BEV_W, BEV_H))
            l_pts, r_pts = get_bev_skeleton(bev_mask)
            m1, b1 = fit_line_huber(l_pts)
            m2, b2 = fit_line_huber(r_pts)


            if m1 is not None and m2 is not None:
                l_width = (WIDTH_EMA_ALPHA * (b2-b1)) + (1-WIDTH_EMA_ALPHA) * l_width
                l_slope = (WIDTH_EMA_ALPHA * (m2-m1)) + (1-WIDTH_EMA_ALPHA) * l_slope
            elif m1 is not None:
                m2, b2 = m1 + l_slope, b1 + l_width
            elif m2 is not None:
                m1, b1 = m2 - l_slope, b2 - l_width


            disp = frame.copy()


            for m in masks:
                disp[m] = (0.5 * np.array(MASK_COLOR if 'MASK_COLOR' in globals() else (0, 180, 0)) + 0.5 * disp[m]).astype(np.uint8)

            if m1 is not None and m2 is not None:

                path_l = project_path(m1, b1, H_BEV2IMG)
                path_r = project_path(m2, b2, H_BEV2IMG)
                cv2.polylines(disp, [path_l], False, LEFT_COLOR, 2)
                cv2.polylines(disp, [path_r], False, RIGHT_COLOR, 2)


                mc, bc = (m1+m2)/2.0, (b1+b2)/2.0
                path_c = project_path(mc, bc, H_BEV2IMG)
                cv2.polylines(disp, [path_c], False, PATH_COLOR, 4)


                heading_err_deg = np.degrees(np.arctan(mc))


                robot_x_bev = ORIGIN_X_OFFSET - BEV_XMIN
                lane_center_at_robot = mc * BEV_H + bc
                lateral_err_cm = (lane_center_at_robot - robot_x_bev) / PIXELS_PER_CM


                raw_steer = (heading_err_deg / 30.0) + (lateral_err_cm / 20.0)
                s_steer = (STEER_EMA_ALPHA * raw_steer) + (1 - STEER_EMA_ALPHA) * s_steer


                cv2.putText(disp, f"YAW ERR: {heading_err_deg:+.1f} deg", (30, 80), 0, 0.6, (0, 200, 255), 2)
                cv2.putText(disp, f"LAT ERR: {lateral_err_cm:+.1f} cm", (30, 105), 0, 0.6, (255, 0, 255), 2)


            t_now = time.time()
            fps = 1.0 / (t_now - t_prev); t_prev = t_now
            cv2.putText(disp, f"STEER: {s_steer:+.2f} | FPS: {fps:.1f}", (30, 50), 0, 1, PATH_COLOR, 2)
            cv2.imshow("RailNav TRT-BEV", disp)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release(); cv2.destroyAllWindows()
        _CUDA_CONTEXT.pop()

if __name__ == "__main__":
    main()
