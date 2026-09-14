import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import PIL.Image
import time
import sys

cuda.init()
_CUDA_DEVICE = cuda.Device(0)
_CUDA_CONTEXT = _CUDA_DEVICE.retain_primary_context()
_CUDA_CONTEXT.push()


ENCODER_ENGINE_PATH = "resnet18_image_encoder.engine"
DECODER_ONNX_PATH   = "mobile_sam_mask_decoder.onnx"

EMA_ALPHA       = 0.25
EDGE_MARGIN_FR  = 0.03

MIN_WIDTH_FR    = 0.05
MIN_RAIL_PTS    = 8
ANGLE_GAUGE_MAX = 30
PRINT_EVERY     = 1


class TRTEncoder:


    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
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


class NanosamHybridPredictor:


    def __init__(self, encoder_engine_path, decoder_onnx_path):
        self.image_encoder = TRTEncoder(encoder_engine_path)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.mask_decoder = ort.InferenceSession(decoder_onnx_path, providers=providers)
        print(f"[init] Mask decoder providers: {self.mask_decoder.get_providers()}")

        self.image_encoder_size = 1024
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess_image(self, image, size=1024):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        image_pil = PIL.Image.fromarray(enhanced)

        image_mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
        image_std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]

        ar = image_pil.width / image_pil.height
        rw, rh = (size, int(size / ar)) if ar >= 1 else (int(size * ar), size)

        image_np = np.array(image_pil.resize((rw, rh))).copy()
        image_torch = torch.from_numpy(image_np).permute(2, 0, 1).float()
        image_norm = (image_torch - image_mean) / image_std

        tensor = torch.zeros((1, 3, size, size))
        tensor[0, :, :rh, :rw] = image_norm
        return tensor.numpy()

    def preprocess_points(self, points, image_size, size=1024):
        return points * (size / max(image_size))

    def set_image(self, image):
        self.image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inp = self.preprocess_image(image, self.image_encoder_size)
        feat = self.image_encoder.infer(inp)
        self.features = feat.reshape(self.image_encoder.output_shape).astype(np.float32)

    def predict(self, points, point_labels):
        points = self.preprocess_points(points, (self.image.height, self.image.width),
                                        self.image_encoder_size)
        outputs = self.mask_decoder.run(None, {
            "image_embeddings": self.features,
            "point_coords": np.array([points], dtype=np.float32),
            "point_labels": np.array([point_labels], dtype=np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.array([0], dtype=np.float32),
        })
        _, low_res_masks = outputs
        hi = self.upscale_mask(torch.from_numpy(low_res_masks),
                               (self.image.height, self.image.width))
        mask = hi.numpy()[0, 0] > 0

        m8 = mask.astype(np.uint8) * 255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
        if n > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest)
        return mask

    def upscale_mask(self, mask, image_shape, size=256):
        if image_shape[1] > image_shape[0]:
            lim_x, lim_y = size, int(size * image_shape[0] / image_shape[1])
        else:
            lim_x, lim_y = int(size * image_shape[1] / image_shape[0]), size
        return F.interpolate(mask[:, :, :lim_y, :lim_x], image_shape,
                             mode='bilinear', align_corners=False)


def find_corners(mask, min_area_frac=0.02):

    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    h, w = mask.shape
    if cv2.contourArea(c) < min_area_frac * h * w:
        return None
    pts = c.reshape(-1, 2)
    s, d = pts[:, 0] + pts[:, 1], pts[:, 0] - pts[:, 1]

    def P(i):
        return (int(pts[i, 0]), int(pts[i, 1]))
    return P(np.argmin(s)), P(np.argmax(d)), P(np.argmax(s)), P(np.argmin(d))


def ascii_gauge(value, vmax, width=21):

    v = int(np.clip(value / vmax, -1, 1) * (width // 2))
    bar = [' '] * width
    mid = width // 2
    bar[mid] = '|'
    pos = mid + v
    pos = max(0, min(width - 1, pos))
    bar[pos] = '#'
    return ''.join(bar)


def print_status(fps, status, angle_raw, angle_s, TL, TR):
    gauge = ascii_gauge(angle_s, ANGLE_GAUGE_MAX)
    if TL is not None:
        kp = f"TL=({TL[0]:4d},{TL[1]:4d}) TR=({TR[0]:4d},{TR[1]:4d})"
    else:
        kp = "TL=(  -- ,  --) TR=(  -- ,  --)"
    line = (f"\rFPS:{fps:5.1f} | {status:9s} | "
            f"angle: raw={angle_raw:+6.2f} smooth={angle_s:+6.2f} deg "
            f"[{gauge}] | {kp}")

    sys.stdout.write(line.ljust(120))
    sys.stdout.flush()


def live_nanosam():
    predictor = NanosamHybridPredictor(ENCODER_ENGINE_PATH, DECODER_ONNX_PATH)

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

    err_s = 0.0
    frame_idx = 0

    print("=" * 60)
    print(" NANOSAM LANE  (Jetson hybrid: TRT encoder + ORT-CUDA decoder)")
    print(" angle: tilt of the line joining the two far rail-top")
    print("        keypoints, vs horizontal. 0 = aligned.")
    print(" Ctrl+C to quit.")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[warn] frame grab failed, stopping.")
                break
            t0 = time.time()
            h, w = frame.shape[:2]
            cx_img = w // 2


            predictor.set_image(frame)
            mask = predictor.predict(np.array([[cx_img, h - 50]]), np.array([1]))


            corners = find_corners(mask)
            status = "NO LANE"
            angle_raw = 0.0
            TL = TR = None
            if corners is not None:
                TL, TR, _, _ = corners
                dx = float(TR[0] - TL[0])
                dy = float(TR[1] - TL[1])
                angle_raw = float(np.degrees(np.arctan2(dy, dx)))
                err_s = EMA_ALPHA * angle_raw + (1 - EMA_ALPHA) * err_s
                status = "OK"

            fps = 1.0 / max(1e-6, time.time() - t0)

            frame_idx += 1
            if frame_idx % PRINT_EVERY == 0:
                print_status(fps, status, angle_raw, err_s, TL, TR)

    except KeyboardInterrupt:
        print("\n[info] stopped by user.")
    finally:
        cap.release()


        del predictor
        import gc
        gc.collect()
        _CUDA_CONTEXT.pop()
        _CUDA_DEVICE.retain_primary_context().detach()


if __name__ == "__main__":
    live_nanosam()
