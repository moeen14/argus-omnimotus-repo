import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import time
import sys
import gc
import threading
import queue


ENGINE_PATH = "highcam_rail.engine"
VIDEO_SOURCE = 2
IMG_SIZE    = 640
NUM_CLASSES = 1
NUM_MASKS   = 32
CONF_THRESH = 0.35
IOU_THRESH  = 0.45
MASK_THRESH = 0.5


SRC_PTS = np.array([[78, 291], [142, 56], [463, 57], [520, 290]], dtype=np.float32)
SRC_CM  = np.array([[0, 0], [0, 30.5], [30.5, 30.5], [30.5, 0]], dtype=np.float32)

PIXELS_PER_CM   = 3.0
ORIGIN_X_OFFSET = 250.0
ORIGIN_Y_OFFSET = 660.0
THRESHOLD_Y     = 80
FRAME_W, FRAME_H = 640, 480


DEFAULT_RAIL_WIDTH_BEV = 111
DEFAULT_SLOPE_DIFF     = 0.0
WIDTH_EMA_ALPHA        = 0.1
BEV_MIN_COMPONENT_AREA = 150


SMOOTHING_FACTOR = 0.7
LATERAL_KP       = 0.6
HEADING_KP       = 0.4
IMAGE_CENTER_X   = FRAME_W / 2.0


LEFT_COLOR   = (255, 0, 0)
RIGHT_COLOR  = (255, 255, 0)
CENTER_COLOR = (255, 255, 255)
MASK_COLOR   = np.array([0, 180, 0], dtype=np.float32)

cuda.init()
_CUDA_DEVICE  = cuda.Device(0)
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
    xmin -= 10; xmax += 10
    bev_w, bev_h = xmax - xmin, ymax - ymin
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
    return T.dot(H).astype(np.float32), int(bev_w), int(bev_h), xmin, ymin

H_IMG2BEV, BEV_W, BEV_H, BEV_XMIN, BEV_YMIN = build_bev_transform(FRAME_W, FRAME_H)
H_BEV2IMG  = np.linalg.inv(H_IMG2BEV)


_img_center_pt = np.array([[[FRAME_W / 2.0, FRAME_H]]], dtype=np.float32)
TRUE_BEV_CENTER_X = cv2.perspectiveTransform(_img_center_pt, H_IMG2BEV)[0, 0, 0]


class ThreadedCamera:
    def __init__(self, source, width=640, height=480):
        self.cap = cv2.VideoCapture(source)


        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[Camera] Started at {actual_w}x{actual_h}")

        self.q = queue.Queue(maxsize=3)
        self.running = True
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
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

    def read(self):
        if not self.running and self.q.empty():
            return False, None
        return True, self.q.get()

    def release(self):
        self.running = False
        self.t.join()
        self.cap.release()


class TRTSegDetector:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        self.input_name = None
        self.out_names  = []
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
            shp   = tuple(self.engine.get_tensor_shape(name))
            h_buf = cuda.pagelocked_empty(shp, dtype=np.float32)
            d_buf = cuda.mem_alloc(h_buf.nbytes)
            self.context.set_tensor_address(name, int(d_buf))
            role  = "proto" if len(shp) == 4 else "det"
            self.out[role] = dict(shape=shp, host=h_buf, dev=d_buf)

    def infer(self):

        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for o in self.out.values():
            cuda.memcpy_dtoh_async(o["host"], o["dev"], self.stream)
        self.stream.synchronize()
        det   = self.out["det"]["host"].reshape(self.out["det"]["shape"])
        proto = self.out["proto"]["host"].reshape(self.out["proto"]["shape"])
        return det, proto


def preprocess_into_buffer(frame, out_buffer, size=IMG_SIZE):

    h, w   = frame.shape[:2]
    r      = min(size / h, size / w)
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
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def seg_postprocess(det, proto, r, pad, orig_shape):
    H, W   = orig_shape[:2]
    p      = det[0].T
    conf   = p[:, 4:4 + NUM_CLASSES].max(axis=1)
    keep   = conf > CONF_THRESH
    if not np.any(keep):
        return np.zeros((H, W), dtype=np.uint8), []

    p_keep = p[keep]
    conf   = conf[keep]
    cx, cy, bw, bh = p_keep[:, 0], p_keep[:, 1], p_keep[:, 2], p_keep[:, 3]
    boxes  = np.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], axis=1)
    coeffs = p_keep[:, 4 + NUM_CLASSES:]

    idx    = nms(boxes, conf, IOU_THRESH)
    if not idx:
        return np.zeros((H, W), dtype=np.uint8), []
    boxes, coeffs = boxes[idx], coeffs[idx]

    protos     = proto[0]
    pc, ph, pw = protos.shape
    masks_160  = sigmoid(coeffs @ protos.reshape(pc, ph * pw)).reshape(-1, ph, pw)

    left, top  = pad
    binary_union = np.zeros((H, W), dtype=np.uint8)
    bool_masks   = []


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
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bev_mask, connectivity=8)

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
        s_rows   = rows[sort_idx]
        s_cols   = cols[sort_idx]

        boundaries = np.concatenate([[0],
                                     np.where(np.diff(s_rows))[0] + 1,
                                     [len(s_rows)]])
        edge_points = []
        for k in range(len(unique_rows)):
            seg = s_cols[boundaries[k]:boundaries[k+1]]
            if seg.size == 0: continue
            seg_sorted = np.sort(seg)
            gaps  = np.where(np.diff(seg_sorted) > 1)[0] + 1
            runs  = np.split(seg_sorted, gaps)
            best  = max(runs, key=len)


            if edge_type == 'inner_left':
                x = best[-1]
            elif edge_type == 'inner_right':
                x = best[0]
            else:
                x = int((best[0] + best[-1]) // 2)

            edge_points.append([int(x), int(unique_rows[k])])

        return np.array(edge_points, dtype=np.int32) if edge_points\
               else np.empty((0, 2), dtype=np.int32)

    left_pts  = np.empty((0, 2), dtype=np.int32)
    right_pts = np.empty((0, 2), dtype=np.int32)


    if len(valid) >= 2:
        left_pts  = extract_edge(valid[0][0], 'inner_left')
        right_pts = extract_edge(valid[-1][0], 'inner_right')
    elif len(valid) == 1:
        comp_id, comp_x = valid[0]

        if comp_x < TRUE_BEV_CENTER_X:
            left_pts  = extract_edge(comp_id, 'inner_left')
        else:
            right_pts = extract_edge(comp_id, 'inner_right')

    return left_pts, right_pts


def fit_line_stable(pts):
    if pts is None or len(pts) < 5:
        return None, None
    [vx, vy, x0, y0] = cv2.fitLine(
        pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if abs(vy) < 1e-6:
        return None, None
    m = vx / vy
    b = x0 - m * y0
    return float(m), float(b)


def project_line_safe(m, b, y_start, y_end, H_inv):
    ys  = np.linspace(y_start, y_end, 20)
    xs  = m * ys + b
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, H_inv).astype(np.int32)


def draw_bev_line(canvas, m, b, color):
    x_top = int(np.clip(m * 0     + b, -1000, 10000))
    x_bot = int(np.clip(m * BEV_H + b, -1000, 10000))
    cv2.line(canvas, (x_top, 0), (x_bot, BEV_H), color, 2)


def apply_smoothing(curr, prev, alpha):
    return curr if prev is None else alpha * prev + (1 - alpha) * curr


class RailNavProcessor:
    def __init__(self, engine_path, image_center_x, warmup_iters=5):
        self.engine_path = engine_path
        self.image_center_x = image_center_x

        print(f"Loading TRT Engine: {self.engine_path}")
        self.detector = TRTSegDetector(self.engine_path)

        self.last_known_rail_width = DEFAULT_RAIL_WIDTH_BEV
        self.last_known_slope_diff = DEFAULT_SLOPE_DIFF

        self.prev_steering = 0.0
        self.fps_s  = 0.0
        self._t_prev = None

        self._warmup(warmup_iters)
        print("Initialization complete. Ready to process feed.")

    def _warmup(self, iters):
        dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(max(1, iters)):
            preprocess_into_buffer(dummy, self.detector.h_input)
            self.detector.infer()
        print(f"GPU warm-up done ({iters} iters)")

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
        det, proto  = self.detector.infer()
        binary_mask, bool_masks = seg_postprocess(det, proto, r, pad, frame.shape)


        bev_mask = cv2.warpPerspective(binary_mask, H_IMG2BEV, (BEV_W, BEV_H))
        left_bev_pts, right_bev_pts = get_bev_skeleton_fast(bev_mask)

        m1, b1 = fit_line_stable(left_bev_pts)
        m2, b2 = fit_line_stable(right_bev_pts)


        if m1 is not None and m2 is not None:
            current_width      = b2 - b1
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

        print(f"Rail Width: {self.last_known_rail_width}")

        have_rails     = (m1 is not None and m2 is not None)
        heading_error  = 0.0
        lateral_error  = 0.0
        bottom_cx      = img_cx

        if have_rails:
            mc, bc = (m1 + m2) / 2.0, (b1 + b2) / 2.0

            heading_error = float(np.clip(-np.arctan(mc) / (np.pi / 2.0), -1.0, 1.0))

            y_near_bev = float(cv2.perspectiveTransform(
                np.array([[[img_cx, FRAME_H]]], np.float32), H_IMG2BEV)[0, 0, 1])

            cl_bev    = np.array([[[mc * y_near_bev + bc, y_near_bev]]], np.float32)
            bottom_cx = float(cv2.perspectiveTransform(cl_bev, H_BEV2IMG)[0, 0, 0])

            track_center_bev_x = mc * y_near_bev + bc


            raw_error_pixels = track_center_bev_x - TRUE_BEV_CENTER_X


            physical_tolerance = self.last_known_rail_width / 2.0
            lateral_error = float(np.clip(raw_error_pixels / physical_tolerance, -1.0, 1.0))


        total_error = HEADING_KP * heading_error + LATERAL_KP * lateral_error
        if have_rails:
            self.prev_steering = apply_smoothing(total_error, self.prev_steering, SMOOTHING_FACTOR)
        else:
            self.prev_steering = 0.0


        status_text = "TRACKING" if have_rails else "NO RAILS"
        print(f"[{status_text}] Steer: {self.prev_steering:+.2f} | "
              f"Heading: {heading_error:+.3f} | Lat: {lateral_error:+.3f} | "
              f"FPS: {self.fps_s:.1f}", end='\r')


        disp_1 = frame.copy()
        for m in bool_masks:
            disp_1[m] = (0.5 * MASK_COLOR + 0.5 * disp_1[m]).astype(np.uint8)
        disp_3 = cv2.cvtColor(bev_mask,    cv2.COLOR_GRAY2BGR)

        if have_rails:
            limit_img = np.array([[[img_cx, THRESHOLD_Y]],
                                   [[img_cx, FRAME_H]]], dtype=np.float32)
            limit_bev = cv2.perspectiveTransform(limit_img, H_IMG2BEV)
            y_far, y_near = limit_bev[0][0][1], limit_bev[1][0][1]

            cv2.polylines(disp_1, [project_line_safe(m1, b1, y_far, y_near, H_BEV2IMG)],
                          False, LEFT_COLOR,   3)
            cv2.polylines(disp_1, [project_line_safe(m2, b2, y_far, y_near, H_BEV2IMG)],
                          False, RIGHT_COLOR,  3)
            cv2.polylines(disp_1, [project_line_safe(mc, bc, y_far, y_near, H_BEV2IMG)],
                          False, CENTER_COLOR, 4)

            draw_bev_line(disp_3, m1, b1, LEFT_COLOR)
            draw_bev_line(disp_3, m2, b2, RIGHT_COLOR)
            draw_bev_line(disp_3, mc, bc, (255, 0, 255))

            cv2.line(disp_1, (int(self.image_center_x), THRESHOLD_Y),
                     (int(self.image_center_x), FRAME_H), (0, 0, 255), 1)
            cv2.circle(disp_1, (int(bottom_cx), FRAME_H - 5), 6, (0, 0, 255), -1)

        status = "Tracking Rails (BEV)" if have_rails else "NO RAILS DETECTED"
        target_h = 480

        def rh(img):
            ih, iw = img.shape[:2]
            return cv2.resize(img, (int(iw * target_h / ih), target_h))

        unified = np.hstack((rh(disp_1), rh(disp_3)))
        cv2.putText(unified, status,
                    (20, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(unified, f"e_h:{heading_error:+.3f}  e_l:{lateral_error:+.3f}",
                    (20, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)
        cv2.putText(unified, f"STEER:{self.prev_steering:+.2f}",
                    (20, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(unified, f"FPS:{self.fps_s:.1f}",
                    (unified.shape[1]-130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 2, cv2.LINE_AA)

        return unified


def main():
    processor = None
    cap = None
    try:
        processor = RailNavProcessor(
            engine_path=ENGINE_PATH,
            image_center_x=IMAGE_CENTER_X
        )

        print(f"Opening video source: {VIDEO_SOURCE}")
        cap = ThreadedCamera(VIDEO_SOURCE)

        print("Processing loop started. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video stream or waiting for frame...")
                continue

            unified_overlay = processor.process_frame(frame)
            cv2.imshow("Rail Nav BEV", unified_overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nPlayback interrupted by user.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by keyboard.")
    finally:

        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if processor:
            del processor.detector
        gc.collect()
        _CUDA_CONTEXT.pop()
        _CUDA_DEVICE.retain_primary_context().detach()
        print("\nCleanup complete.")

if __name__ == '__main__':
    main()
