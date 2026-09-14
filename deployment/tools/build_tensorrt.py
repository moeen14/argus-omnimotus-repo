import argparse
import glob
import os
import sys
import time

import numpy as np


import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
from yolo_common import preprocess

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, files, cache, imgsz, batch=1):
        super().__init__()
        self.cuda = cuda
        self.files = files
        self.cache = cache
        self.imgsz = imgsz
        self.batch = batch
        self.idx = 0
        self.dev = cuda.mem_alloc(batch * 3 * imgsz * imgsz * 4)

    def get_batch_size(self):
        return self.batch

    def get_batch(self, names):
        if self.idx + self.batch > len(self.files):
            return None
        arrs = [preprocess(self.files[self.idx + i], self.imgsz)[0]
                for i in range(self.batch)]
        blob = np.ascontiguousarray(np.concatenate(arrs, 0), dtype=np.float32)
        self.cuda.memcpy_htod(self.dev, blob)
        self.idx += self.batch
        return [int(self.dev)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache):
            return open(self.cache, "rb").read()

    def write_calibration_cache(self, data):
        open(self.cache, "wb").write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="int8", choices=["int8", "fp16", "fp32"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workspace-mb", type=int, default=1024)
    ap.add_argument("--calib-limit", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    task = os.path.basename(os.path.dirname(args.onnx))
    name = os.path.splitext(os.path.basename(args.onnx))[0]
    stem = f"{task}__{name}_{args.precision}"
    engine_path = os.path.join(args.out, stem + ".engine")
    cache_path = os.path.join(args.out, stem + ".calib_cache")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit("[!] ONNX parse failed")

    cfg = builder.create_builder_config()
    try:
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
    except AttributeError:
        cfg.max_workspace_size = args.workspace_mb << 20

    if args.precision == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
    elif args.precision == "int8":
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.set_flag(trt.BuilderFlag.FP16)
        files = sorted(glob.glob(os.path.join(args.calib, "*")))
        files = [f for f in files
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))][:args.calib_limit]
        cfg.int8_calibrator = ImageCalibrator(files, cache_path, args.imgsz)
        print(f"[i] INT8 calibration with {len(files)} images")

    print(f"[i] building {stem} ... (minutes)")
    t0 = time.time()
    try:
        serialized = builder.build_serialized_network(network, cfg)
    except AttributeError:
        serialized = builder.build_engine(network, cfg).serialize()
    if serialized is None:
        raise SystemExit("[!] engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    nbytes = getattr(serialized, "nbytes", None) or len(serialized)
    print(f"[i] wrote {engine_path}  ({nbytes/1e6:.1f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
