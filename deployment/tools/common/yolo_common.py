import numpy as np


def letterbox(img, new_shape=640, color=(114, 114, 114)):

    import cv2
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape - nw) / 2, (new_shape - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def preprocess(path, imgsz=640, to_rgb=True):

    import cv2
    im0 = cv2.imread(path)
    h0, w0 = im0.shape[:2]
    im, r, (dw, dh) = letterbox(im0, imgsz)
    if to_rgb:
        im = im[:, :, ::-1]
    x = np.ascontiguousarray(im.transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
    return x, (r, dw, dh, w0, h0)


def preprocess_u8_nhwc(path, imgsz=640):

    import cv2
    im0 = cv2.imread(path)
    im, r, (dw, dh) = letterbox(im0, imgsz)
    im = im[:, :, ::-1]
    return np.ascontiguousarray(im)[None], (r, dw, dh, im0.shape[1], im0.shape[0])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _dfl(reg, reg_max=16):

    b = reg.reshape(*reg.shape[:-1], 4, reg_max)
    b = b - b.max(-1, keepdims=True)
    p = np.exp(b)
    p /= p.sum(-1, keepdims=True)
    return (p * np.arange(reg_max)).sum(-1)


def _make_anchors(feats_hw, strides):
    pts, st = [], []
    for (h, w), s in zip(feats_hw, strides):
        sx = np.arange(w) + 0.5
        sy = np.arange(h) + 0.5
        gy, gx = np.meshgrid(sy, sx, indexing="ij")
        pts.append(np.stack((gx.ravel(), gy.ravel()), -1))
        st.append(np.full((h * w, 1), s))
    return np.concatenate(pts), np.concatenate(st)


def decode_heads(outputs, nc, imgsz=640, reg_max=16, strides=(8, 16, 32)):


    arrs = list(outputs.values()) if isinstance(outputs, dict) else list(outputs)


    dec = [a for a in arrs if a.ndim == 3 and a.shape[1] in (4 + nc, 4 + nc + 32)]
    raw_box_head = any(a.ndim == 4 and a.shape[1] == 4 * reg_max for a in arrs)
    if len(dec) == 1 and not raw_box_head:
        p = dec[0][0]
        p = p.transpose(1, 0)
        xywh, rest = p[:, :4], p[:, 4:]
        cls = rest[:, :nc]
        mc = rest[:, nc:] if rest.shape[1] > nc else None
        xyxy = np.empty_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        sc = cls.max(1)
        cl = cls.argmax(1)
        proto = None
        for a in arrs:
            if a.ndim == 4 and a.shape[1] == 32:
                proto = a[0]
        return xyxy, sc, cl, mc, proto


    heads4d = [a for a in arrs if a.ndim == 4]
    proto = None
    box_feats, cls_feats, msk_feats = [], [], []
    for a in heads4d:
        c = a.shape[1]
        if c == 4 * reg_max:
            box_feats.append(a)
        elif c == nc:
            cls_feats.append(a)
        elif c == 32 and a.shape[2] <= 80:
            msk_feats.append(a)
        elif c == 32:
            proto = a[0]
    order = np.argsort([-f.shape[2] for f in box_feats])
    box_feats = [box_feats[i] for i in order]
    cls_feats = [cls_feats[i] for i in np.argsort([-f.shape[2] for f in cls_feats])]
    if msk_feats:
        msk_feats = [msk_feats[i] for i in np.argsort([-f.shape[2] for f in msk_feats])]

    feats_hw = [f.shape[2:] for f in box_feats]
    anchors, strd = _make_anchors(feats_hw, strides)

    box = np.concatenate([f[0].reshape(f.shape[1], -1).T for f in box_feats], 0)
    cls = np.concatenate([f[0].reshape(f.shape[1], -1).T for f in cls_feats], 0)


    dist = _dfl(box, reg_max)
    x1 = anchors[:, 0] - dist[:, 0]
    y1 = anchors[:, 1] - dist[:, 1]
    x2 = anchors[:, 0] + dist[:, 2]
    y2 = anchors[:, 1] + dist[:, 3]
    xyxy = np.stack([x1, y1, x2, y2], 1) * strd
    scores_all = _sigmoid(cls)
    sc = scores_all.max(1)
    cl = scores_all.argmax(1)
    mc = None
    if msk_feats:
        mc = np.concatenate([f[0].reshape(f.shape[1], -1).T for f in msk_feats], 0)
    return xyxy, sc, cl, mc, proto


def nms(boxes, scores, iou_thr=0.7):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


def postprocess(xyxy, sc, cl, meta, conf=0.001, iou=0.7, max_det=300):


    r, dw, dh, w0, h0 = meta
    m = sc > conf
    orig_idx = np.flatnonzero(m)
    xyxy, sc, cl = xyxy[m], sc[m], cl[m]
    if not len(sc):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0, int)
    keep_all = []
    for c in np.unique(cl):
        idx = np.where(cl == c)[0]
        k = nms(xyxy[idx], sc[idx], iou)
        keep_all.extend(idx[k])
    keep_all = np.array(keep_all, int)
    keep_all = keep_all[sc[keep_all].argsort()[::-1][:max_det]]
    b = xyxy[keep_all].copy()
    b[:, [0, 2]] -= dw
    b[:, [1, 3]] -= dh
    b /= r
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w0)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h0)
    return b, sc[keep_all], cl[keep_all], orig_idx[keep_all]


def box_iou(a, b):
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class MetricAccumulator:


    IOUV = np.linspace(0.5, 0.95, 10)

    def __init__(self, nc):
        self.nc = nc
        self.stats = []

    def add(self, det_boxes, det_scores, det_cls, gt_boxes, gt_cls, iou=None):
        nl = len(gt_cls)
        tcls = gt_cls.astype(int)
        if len(det_scores) == 0:
            if nl:
                self.stats.append((np.zeros((0, 10), bool), np.zeros(0),
                                   np.zeros(0), tcls))
            return
        correct = np.zeros((len(det_scores), 10), bool)
        if nl:
            if iou is None:
                iou = box_iou(det_boxes, gt_boxes)
            for k, thr in enumerate(self.IOUV):
                matched_gt, matched_dt = set(), set()
                order = det_scores.argsort()[::-1]
                for di in order:
                    ious = iou[di].copy()
                    for gj in np.argsort(ious)[::-1]:
                        if ious[gj] < thr:
                            break
                        if gj in matched_gt or det_cls[di] != tcls[gj]:
                            continue
                        matched_gt.add(gj)
                        matched_dt.add(di)
                        correct[di, k] = True
                        break
        self.stats.append((correct, det_scores, det_cls.astype(int), tcls))

    @staticmethod
    def _ap(recall, precision):
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        i = np.where(mrec[1:] != mrec[:-1])[0]
        return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    def compute(self, conf_thr=0.25):
        if not self.stats:
            return {}
        tp = np.concatenate([s[0] for s in self.stats], 0)
        conf = np.concatenate([s[1] for s in self.stats], 0)
        pcls = np.concatenate([s[2] for s in self.stats], 0)
        tcls = np.concatenate([s[3] for s in self.stats], 0)
        order = conf.argsort()[::-1]
        tp, conf, pcls = tp[order], conf[order], pcls[order]

        ap = np.zeros((self.nc, 10))
        p_at = np.zeros(self.nc)
        r_at = np.zeros(self.nc)
        for c in range(self.nc):
            i = pcls == c
            n_gt = (tcls == c).sum()
            n_p = i.sum()
            if n_p == 0 or n_gt == 0:
                continue
            fpc = (~tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
            recall = tpc / (n_gt + 1e-9)
            precision = tpc / (tpc + fpc)
            for k in range(10):
                ap[c, k] = self._ap(recall[:, k], precision[:, k])

            sel = conf[i] >= conf_thr
            tp50 = tp[i][:, 0]
            tp_n = tp50[sel].sum()
            fp_n = (~tp50[sel]).sum()
            p_at[c] = tp_n / (tp_n + fp_n + 1e-9)
            r_at[c] = tp_n / (n_gt + 1e-9)

        present = np.array([(tcls == c).sum() > 0 for c in range(self.nc)])
        map50 = ap[present, 0].mean() if present.any() else 0.0
        map5095 = ap[present].mean() if present.any() else 0.0
        p = p_at[present].mean() if present.any() else 0.0
        r = r_at[present].mean() if present.any() else 0.0
        f1 = 2 * p * r / (p + r + 1e-9)
        return {"mAP50": float(map50), "mAP50_95": float(map5095),
                "precision": float(p), "recall": float(r), "f1": float(f1),
                "conf_thr": conf_thr}


def decode_masks(coeffs, proto, boxes_orig, meta, imgsz=640):


    import cv2
    _, dw, dh, w0, h0 = meta
    c, mh, mw = proto.shape
    m = 1.0 / (1.0 + np.exp(-(coeffs @ proto.reshape(c, -1))))
    m = m.reshape(-1, mh, mw)
    out = np.zeros((len(m), h0, w0), dtype=bool)
    for i, (mi, b) in enumerate(zip(m, boxes_orig)):


        full = cv2.resize(mi, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        full = full[dh:imgsz - dh if dh else imgsz, dw:imgsz - dw if dw else imgsz]
        full = cv2.resize(full, (w0, h0), interpolation=cv2.INTER_LINEAR)
        keep = np.zeros((h0, w0), dtype=bool)
        x1, y1, x2, y2 = np.clip(b, 0, [w0, h0, w0, h0]).astype(int)
        keep[y1:y2, x1:x2] = True
        out[i] = (full > 0.5) & keep
    return out


def mask_iou(a, b):

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    af = a.reshape(len(a), -1).astype(np.float32)
    bf = b.reshape(len(b), -1).astype(np.float32)
    inter = af @ bf.T
    area_a, area_b = af.sum(1)[:, None], bf.sum(1)[None, :]
    return inter / (area_a + area_b - inter + 1e-9)


def load_gt_masks(label_path, w, h):


    import cv2
    masks, cls = [], []
    try:
        lines = open(label_path).read().strip().splitlines()
    except FileNotFoundError:
        return np.zeros((0, h, w), dtype=bool), np.zeros(0, int)
    for ln in lines:
        v = ln.split()
        if len(v) < 5:
            continue
        c = int(float(v[0]))
        nums = np.array(v[1:], float)
        m = np.zeros((h, w), dtype=np.uint8)
        if len(nums) == 4:
            cx, cy, bw, bh = nums
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(m, (x1, y1), (x2, y2), 1, -1)
        else:
            xs = (nums[0::2] * w).astype(np.int32)
            ys = (nums[1::2] * h).astype(np.int32)
            cv2.fillPoly(m, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], 1)
        masks.append(m.astype(bool))
        cls.append(c)
    if not masks:
        return np.zeros((0, h, w), dtype=bool), np.zeros(0, int)
    return np.array(masks), np.array(cls, int)


def load_gt(label_path, w, h):


    boxes, cls = [], []
    try:
        lines = open(label_path).read().strip().splitlines()
    except FileNotFoundError:
        return np.zeros((0, 4)), np.zeros(0, int)
    for ln in lines:
        v = ln.split()
        if len(v) < 5:
            continue
        c = int(float(v[0]))
        nums = np.array(v[1:], float)
        if len(nums) == 4:
            cx, cy, bw, bh = nums
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        else:
            xs = nums[0::2]
            ys = nums[1::2]
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        boxes.append([x1 * w, y1 * h, x2 * w, y2 * h])
        cls.append(c)
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0, int)
    return np.array(boxes), np.array(cls, int)


def onnx_params_flops(onnx_path, imgsz=640):


    import onnx
    from onnx import shape_inference
    m = shape_inference.infer_shapes(onnx.load(onnx_path))
    g = m.graph
    init = {i.name: i for i in g.initializer}
    params = sum(np.prod(i.dims) for i in g.initializer)

    vinfo = {vi.name: vi for vi in list(g.value_info) + list(g.output) + list(g.input)}

    def shape(name):
        vi = vinfo.get(name)
        if vi is None:
            return None
        return [d.dim_value if (d.dim_value > 0) else 1
                for d in vi.type.tensor_type.shape.dim]

    macs = 0
    for n in g.node:
        if n.op_type == "Conv":
            w = init.get(n.input[1])
            out = shape(n.output[0])
            if w is None or out is None:
                continue
            cout, cin_g = w.dims[0], w.dims[1]
            k = int(np.prod(w.dims[2:]))
            out_spatial = int(np.prod(out[2:]))
            macs += cout * cin_g * k * out_spatial
        elif n.op_type in ("Gemm", "MatMul"):
            a = shape(n.input[0])
            b = shape(n.input[1]) or (init[n.input[1]].dims if n.input[1] in init else None)
            if a and b:
                macs += int(np.prod(a)) * b[-1]
    return params / 1e6, 2 * macs / 1e9
