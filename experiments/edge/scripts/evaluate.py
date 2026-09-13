import argparse
import atexit
import contextlib
import glob
import json
import os
import platform
import resource
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "common"))
from yolo_common import (MetricAccumulator, decode_heads, decode_masks, load_gt,
                         load_gt_masks, mask_iou, onnx_params_flops, postprocess,
                         preprocess, preprocess_u8_nhwc)

NC = {"crop_detector": 3, "highcam_platform": 4, "highcam_rail": 1}
SEG = {"highcam_rail": True}


class OnnxBackend:
    layout = "chw01"

    def __init__(self, path):
        import onnxruntime as ort
        prov = ort.get_available_providers()
        pref = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in prov]
        self.sess = ort.InferenceSession(path, providers=pref)
        self.iname = self.sess.get_inputs()[0].name
        print(f"[i] onnxruntime providers: {self.sess.get_providers()}")

    def infer(self, x):
        t = time.perf_counter()
        outs = self.sess.run(None, {self.iname: x})
        return outs, time.perf_counter() - t


class HailoBackend:
    layout = "nhwc_u8"

    def __init__(self, path):
        from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface,
                                    InferVStreams, InputVStreamParams, OutputVStreamParams,
                                    VDevice)
        self.HEF = HEF(path)
        self.vdev = VDevice()
        cfg = ConfigureParams.create_from_hef(self.HEF, interface=HailoStreamInterface.PCIe)
        self.ng = self.vdev.configure(self.HEF, cfg)[0]
        self.ngp = self.ng.create_params()
        self.ivp = InputVStreamParams.make(self.ng, quantized=True, format_type=FormatType.UINT8)
        self.ovp = OutputVStreamParams.make(self.ng, quantized=False, format_type=FormatType.FLOAT32)
        self.in_name = self.HEF.get_input_vstream_infos()[0].name


        self._stack = contextlib.ExitStack()
        self.pipe = self._stack.enter_context(InferVStreams(
            self.ng, self.ivp, self.ovp))
        self._stack.enter_context(self.ng.activate(self.ngp))
        atexit.register(self.close)

    def close(self):
        stack = getattr(self, "_stack", None)
        if stack is not None:
            self._stack = None
            stack.close()

    def infer(self, x):
        t = time.perf_counter()
        res = self.pipe.infer({self.in_name: x})
        dt = time.perf_counter() - t
        outs = []
        for v in res.values():
            a = np.asarray(v)
            if a.ndim == 4:
                a = a.transpose(0, 3, 1, 2)
            outs.append(a)
        return outs, dt


class TensorRTBackend:
    layout = "chw01"

    def __init__(self, path):
        import pycuda.autoinit
        import pycuda.driver as cuda
        import tensorrt as trt
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.host, self.dev, self.names = [], {}, {}, []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            self.names.append(n)
            shape = self.ctx.get_tensor_shape(n)
            dt = trt.nptype(self.engine.get_tensor_dtype(n))
            h = cuda.pagelocked_empty(trt.volume(shape), dt)
            d = cuda.mem_alloc(h.nbytes)
            self.host[n], self.dev[n] = h, d
            self.ctx.set_tensor_address(n, int(d))
        self.in_name = self.engine.get_tensor_name(0)

    def infer(self, x):
        c = self.cuda
        t = time.perf_counter()
        np.copyto(self.host[self.in_name], x.ravel())
        c.memcpy_htod_async(self.dev[self.in_name], self.host[self.in_name], self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        outs = []
        for n in self.names[1:]:
            c.memcpy_dtoh_async(self.host[n], self.dev[n], self.stream)
        self.stream.synchronize()
        dt = time.perf_counter() - t
        for n in self.names[1:]:
            shape = tuple(self.ctx.get_tensor_shape(n))
            outs.append(self.host[n].reshape(shape).copy())
        return outs, dt


class NcnnBackend:


    layout = "chw01"

    def __init__(self, path):
        import ncnn
        self.ncnn = ncnn
        assert path.endswith(".param"), "point --model at a .param file"
        binp = path[:-6] + ".bin"
        self.net = ncnn.Net()

        self.net.opt.use_vulkan_compute = False
        try:
            self.net.opt.num_threads = int(os.environ.get("NCNN_THREADS", "4"))
        except Exception:
            pass
        self.net.load_param(path)
        self.net.load_model(binp)
        self.inp = self.net.input_names()[0]
        self.outs = list(self.net.output_names())

    def infer(self, x):
        m = self.ncnn.Mat(np.ascontiguousarray(x[0]))
        t = time.perf_counter()
        ex = self.net.create_extractor()
        ex.input(self.inp, m)
        raw = []
        for o in self.outs:
            ret, r = ex.extract(o)
            raw.append(np.array(r))
        dt = time.perf_counter() - t
        return [a[None] for a in raw], dt


BACKENDS = {"onnx": OnnxBackend, "hailo": HailoBackend, "tensorrt": TensorRTBackend,
            "ncnn": NcnnBackend}


def pct(v, p):
    return statistics.quantiles(v, n=100)[p - 1] if len(v) > 2 else max(v)


def jetson_power_state():


    import shutil
    import subprocess
    mode = clocks = None
    try:
        if shutil.which("nvpmodel"):
            out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                                 timeout=5).stdout
            for ln in out.splitlines():
                if "NV Power Mode" in ln:
                    mode = ln.split(":")[-1].strip()
                    break
    except Exception:
        pass
    try:
        if shutil.which("jetson_clocks"):
            out = subprocess.run(["jetson_clocks", "--show"], capture_output=True,
                                 text=True, timeout=5).stdout
            gpu = [l for l in out.splitlines() if "GPU" in l and "MinFreq" in l]
            if gpu:
                import re as _re
                nums = _re.findall(r"(\w+)=(\d+)", gpu[0])
                d = {k: int(v) for k, v in nums}
                if "MinFreq" in d and "CurrentFreq" in d:
                    clocks = d["CurrentFreq"] >= d.get("MaxFreq", d["CurrentFreq"])
    except Exception:
        pass
    return mode, clocks


def _read_soc_power_mw_sysfs():


    import glob as _g
    for base in _g.glob("/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*/"):
        try:
            pfiles = _g.glob(base + "power*_input")
            if pfiles:
                tot = sum(int(open(f).read().strip()) for f in pfiles)
                if tot:
                    return tot / 1000.0 if tot > 100000 else tot
            currs = sorted(_g.glob(base + "curr*_input"))
            volts = sorted(_g.glob(base + "in[1-3]_input"))
            if currs and volts and len(currs) == len(volts):
                tot_mw = sum(int(open(cf).read().strip()) * int(open(vf).read().strip())
                            for cf, vf in zip(currs, volts)) / 1000.0
                if tot_mw:
                    return tot_mw
        except Exception:
            pass
    return None


class PowerSampler:


    def __init__(self, interval_s=0.2, ina260_bus=None, ina260_address=0x40):
        self.interval_s = interval_s
        self.ina260_bus = ina260_bus
        self.ina260_address = ina260_address
        self.samples = []
        self.source = None
        self._stop = None
        self._thread = None
        self._jetson = None
        self._ina260_fd = None

    def start(self):
        import threading
        if self.ina260_bus is not None:


            import fcntl
            self._ina260_fd = os.open(f"/dev/i2c-{self.ina260_bus}", os.O_RDWR)
            fcntl.ioctl(self._ina260_fd, 0x0703, self.ina260_address)
            self.source = f"ina260-i2c-{self.ina260_bus}-0x{self.ina260_address:02x}"


            if self._read_ina260_mw() is None:
                os.close(self._ina260_fd)
                self._ina260_fd = None
                raise RuntimeError(f"INA260 unavailable on /dev/i2c-{self.ina260_bus} "
                                   f"at 0x{self.ina260_address:02x}")
        else:
            try:
                from jtop import jtop
                self._jetson = jtop()
                self._jetson.start()
                self.source = "jtop"
            except Exception:
                self._jetson = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _read_ina260_mw(self):
        try:
            os.write(self._ina260_fd, b"\x03")
            raw = os.read(self._ina260_fd, 2)
            return int.from_bytes(raw, "big") * 10.0 if len(raw) == 2 else None
        except OSError:
            return None

    def _read_once(self):
        if self._ina260_fd is not None:
            return self._read_ina260_mw()
        if self._jetson is not None:
            try:
                return float(self._jetson.power["tot"]["power"])
            except Exception:
                return None
        mw = _read_soc_power_mw_sysfs()
        if mw is not None and self.source is None:
            self.source = "sysfs"
        return mw

    def _run(self):
        while not self._stop.is_set():
            mw = self._read_once()
            if mw:
                self.samples.append(mw)
            self._stop.wait(self.interval_s)

    def stop(self):
        if self._stop is not None:
            self._stop.set()
            self._thread.join(timeout=2)
        if self._jetson is not None:
            try:
                self._jetson.close()
            except Exception:
                pass
        if self._ina260_fd is not None:
            os.close(self._ina260_fd)
            self._ina260_fd = None
        return self.samples


def append_table_row(md_path, section_title, table_header, row_line):


    marker = f"## {section_title}"
    if not os.path.exists(md_path):
        open(md_path, "w").write(f"{marker}\n\n{table_header}{row_line}")
        return
    text = open(md_path).read()
    if marker not in text:
        sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        open(md_path, "a").write(f"{sep}{marker}\n\n{table_header}{row_line}")
        return
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.rstrip("\n") == marker)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    while end > start + 1 and lines[end - 1].strip() == "":
        end -= 1
    lines[end:end] = [row_line]
    open(md_path, "w").writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=BACKENDS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=NC)
    ap.add_argument("--device", required=True, help="free-form device tag for the report")
    ap.add_argument("--data", default=None, help="dataset dir (default datasets/<task>)")
    ap.add_argument("--onnx-ref", default=None,
                    help="ONNX used only for params/GFLOPs (default models/onnx/<task>/<name>.onnx)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1,
                    help="extra timing-only passes over the dataset (thermal / sustained FPS)")
    ap.add_argument("--power-mw", type=float, default=None,
                    help="measured board/SoC power in mW -> FPS/Watt (manual, e.g. from a USB "
                         "meter). Overrides the automatic mean sampled over the whole run "
                         "(PowerSampler, via jtop on Jetson) when given.")
    ap.add_argument("--ina260-bus", type=int, default=None,
                    help="sample an INA260 on this Linux I2C bus for the whole benchmark run")
    ap.add_argument("--ina260-address", type=lambda x: int(x, 0), default=0x40,
                    help="INA260 I2C address (default 0x40; accepts decimal or 0x notation)")
    ap.add_argument("--baseline", default=None,
                    help="path to a previous result .json (usually the onnx FP32 run) "
                         "-> accuracy deltas vs it")
    ap.add_argument("--note", default="", help="free-form note stored in the row")
    args = ap.parse_args()

    nc = NC[args.task]
    data = args.data or os.path.join(ROOT, "datasets", args.task)
    imgs = sorted(glob.glob(os.path.join(data, "images", "*")))
    imgs = [f for f in imgs if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if args.limit:
        imgs = imgs[:args.limit]
    assert imgs, f"no images in {data}/images"

    name = os.path.basename(args.model)
    for ext in (".onnx", ".hef", ".engine", ".param", ".bin",
                "_int8mixed", "_int8", "_fp16", "_fp32"):
        name = name.replace(ext, "")
    name = name.replace(f"{args.task}__", "")
    model_base = os.path.basename(args.model)
    precision = ("int8-mixed" if "int8mixed" in model_base else
                 next((p for p in ("int8", "fp16", "fp32") if p in model_base), None))
    onnx_ref = args.onnx_ref or os.path.join(ROOT, "models", "onnx", args.task, name + ".onnx")

    print(f"[i] backend={args.backend} model={args.model}")
    print(f"[i] task={args.task} nc={nc} images={len(imgs)} device={args.device}")

    be = BACKENDS[args.backend](args.model)
    acc = MetricAccumulator(nc)
    acc_mask = MetricAccumulator(nc) if SEG.get(args.task) else None
    mask_capable = None
    lat, pre_t, post_t = [], [], []


    x0, _ = (preprocess_u8_nhwc if be.layout == "nhwc_u8" else preprocess)(imgs[0], args.imgsz)
    for _ in range(args.warmup):
        be.infer(x0)

    power_sampler = PowerSampler(ina260_bus=args.ina260_bus,
                                 ina260_address=args.ina260_address).start()
    for rep in range(args.repeat):
        score = rep == 0
        for k, path in enumerate(imgs):
            t = time.perf_counter()
            if be.layout == "nhwc_u8":
                x, meta = preprocess_u8_nhwc(path, args.imgsz)
            else:
                x, meta = preprocess(path, args.imgsz)
            pre = (time.perf_counter() - t) * 1e3

            outs, dt = be.infer(x)
            lat.append(dt * 1e3)

            t = time.perf_counter()
            xyxy, sc, cl, mc, proto = decode_heads(outs, nc, args.imgsz)
            if acc_mask is not None and mask_capable is None:


                mask_capable = mc is not None and proto is not None
                print("[i] mask mAP: computed (mask coeffs + proto found in model output)"
                      if mask_capable else
                      "[!] mask mAP: NOT computed - this model/backend exposes no mask "
                      "coeffs/proto (only the box-around-polygon mAP above will be "
                      "reported); see decode_masks()/mask_iou()/load_gt_masks() in "
                      "scripts/common/yolo_common.py and scripts/testin_code_reference.txt")
            b, s, c, idx = postprocess(xyxy, sc, cl, meta, conf=0.001, iou=args.iou)
            post = (time.perf_counter() - t) * 1e3
            if score:
                pre_t.append(pre)
                post_t.append(post)
                lbl = os.path.join(data, "labels",
                                   os.path.splitext(os.path.basename(path))[0] + ".txt")
                gtb, gtc = load_gt(lbl, meta[3], meta[4])
                acc.add(b, s, c, gtb, gtc)
                if acc_mask is not None and mc is not None and proto is not None:
                    pm = (decode_masks(mc[idx], proto, b, meta, args.imgsz) if len(idx)
                          else np.zeros((0, meta[4], meta[3]), dtype=bool))
                    gtm, _ = load_gt_masks(lbl, meta[3], meta[4])
                    acc_mask.add(b, s, c, gtb, gtc, iou=mask_iou(pm, gtm))
            if score and (k + 1) % 25 == 0:
                print(f"    {k+1}/{len(imgs)}")
        if args.repeat > 1:
            print(f"    pass {rep+1}/{args.repeat}  "
                  f"mean {statistics.mean(lat[-len(imgs):]):.2f} ms")
    power_samples = power_sampler.stop()

    m = acc.compute(args.conf)
    mm = acc_mask.compute(args.conf) if acc_mask is not None else None
    params, gflops = onnx_params_flops(onnx_ref, args.imgsz) if os.path.exists(onnx_ref)\
        else (float("nan"), float("nan"))
    fps = 1000.0 / statistics.mean(lat)
    n_burst = max(5, len(lat) // 20)
    fps_burst = 1000.0 / statistics.mean(lat[:n_burst])
    fps_sustained = 1000.0 / statistics.mean(lat[-n_burst:])
    e2e_ms = statistics.mean(pre_t) + statistics.mean(lat[:len(pre_t)]) + statistics.mean(post_t)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    model_mb = os.path.getsize(args.model) / 1e6
    if args.model.endswith(".param"):
        _b = args.model[:-6] + ".bin"
        if os.path.exists(_b):
            model_mb += os.path.getsize(_b) / 1e6
    pmode, jclocks = jetson_power_state()
    power_mw_mean = statistics.mean(power_samples) if power_samples else None
    power_mw_min = min(power_samples) if power_samples else None
    power_mw_max = max(power_samples) if power_samples else None


    power_mw = args.power_mw or power_mw_mean
    fps_per_w = (fps / (power_mw / 1000.0)) if power_mw else None

    row = {
        "device": args.device, "backend": args.backend, "task": args.task,
        "model": name, "quant": precision or "native",
        "segmentation": bool(SEG.get(args.task)),
        "images": len(imgs), "conf": args.conf, "iou": args.iou,
        "mAP50": round(m["mAP50"], 4), "mAP50_95": round(m["mAP50_95"], 4),
        "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "fps": round(fps, 1),
        "lat_mean_ms": round(statistics.mean(lat), 2),
        "lat_median_ms": round(statistics.median(lat), 2),
        "lat_p90_ms": round(pct(lat, 90), 2),
        "lat_p99_ms": round(pct(lat, 99), 2),
        "preprocess_ms": round(statistics.mean(pre_t), 2),
        "postprocess_ms": round(statistics.mean(post_t), 2),
        "e2e_latency_ms": round(e2e_ms, 2),
        "e2e_fps": round(1000.0 / e2e_ms, 1),
        "fps_burst": round(fps_burst, 1),
        "fps_sustained": round(fps_sustained, 1),
        "params_M": round(params, 3), "gflops": round(gflops, 2),
        "model_file_MB": round(model_mb, 2),
        "peak_rss_MB": round(rss, 1),
        "power_mw": round(power_mw, 1) if power_mw else None,
        "power_mw_min": round(power_mw_min, 1) if power_mw_min else None,
        "power_mw_max": round(power_mw_max, 1) if power_mw_max else None,
        "power_samples_n": len(power_samples),
        "power_source": "manual" if args.power_mw else power_sampler.source,
        "fps_per_watt": round(fps_per_w, 2) if fps_per_w else None,
        "power_mode": pmode, "jetson_clocks": jclocks,
        "passes": args.repeat, "note": args.note,
        "host": platform.node(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if mm:


        row["mask_mAP50"] = round(mm["mAP50"], 4)
        row["mask_mAP50_95"] = round(mm["mAP50_95"], 4)
        row["mask_precision"] = round(mm["precision"], 4)
        row["mask_recall"] = round(mm["recall"], 4)
        row["mask_f1"] = round(mm["f1"], 4)

    try:
        base = json.load(open(args.baseline)) if args.baseline else None
    except (ValueError, OSError) as e:
        print(f"[!] --baseline ignored ({e})")
        base = None
    if base:
        row["baseline"] = os.path.basename(args.baseline)
        delta_keys = ["mAP50", "mAP50_95", "precision", "recall", "f1"]
        if mm:
            delta_keys += ["mask_mAP50", "mask_mAP50_95", "mask_precision",
                           "mask_recall", "mask_f1"]
        for key in delta_keys:
            if key in base and key in row:
                row[f"{key}_delta"] = round(row[key] - base[key], 4)
        if base.get("fps"):
            row["speedup_vs_baseline"] = round(fps / base["fps"], 2)

    print(json.dumps(row, indent=2))

    outdir = os.path.join(ROOT, "results", args.device)
    os.makedirs(outdir, exist_ok=True)
    stem = f"{args.task}__{name}__{args.backend}" + (f"_{precision}" if precision else "")
    json.dump(row, open(os.path.join(outdir, stem + ".json"), "w"), indent=2)

    md = os.path.join(ROOT, "docs", "RESULTS.md")
    g = lambda k: "" if row.get(k) is None else row.get(k)

    header = ("| device | backend | task | model | seg | mAP50 | mAP50-95 | quant "
              "| ΔmAP50-95 | P | R | F1 | FPS | FPS_sust | e2e_FPS | lat_mean | lat_p99 | pre | post "
              "| params(M) | GFLOPs | file(MB) | RSS(MB) | power(mW) | FPS/W | mode | date |\n"
              "|" + "---|" * 27 + "\n")
    line = (f"| {row['device']} | {row['backend']} | {row['task']} | {row['model']} "
            f"| {'Y' if row['segmentation'] else 'N'} | {row['mAP50']} | {row['mAP50_95']} "
            f"| {row['quant']} "
            f"| {g('mAP50_95_delta')} | {row['precision']} | {row['recall']} | {row['f1']} "
            f"| {row['fps']} | {row['fps_sustained']} | {row['e2e_fps']} "
            f"| {row['lat_mean_ms']} | {row['lat_p99_ms']} "
            f"| {row['preprocess_ms']} | {row['postprocess_ms']} | {row['params_M']} "
            f"| {row['gflops']} | {row['model_file_MB']} | {row['peak_rss_MB']} "
            f"| {g('power_mw')} | {g('fps_per_watt')} | {g('power_mode')} | {row['timestamp']} |\n")
    append_table_row(md, "Benchmark rows", header, line)
    print(f"[i] appended row to {md}")
    print(f"[i] wrote {os.path.join(outdir, stem + '.json')}")

    if mm:


        mask_header = (
            "| device | backend | task | model | quant | mask_mAP50 | mask_mAP50-95 "
            "| Δmask_mAP50-95 | mask_P | mask_R | mask_F1 | images | date |\n"
            "|" + "---|" * 13 + "\n")
        mline = (f"| {row['device']} | {row['backend']} | {row['task']} | {row['model']} "
                 f"| {row['quant']} | {row['mask_mAP50']} | {row['mask_mAP50_95']} "
                 f"| {g('mask_mAP50_95_delta')} | {row['mask_precision']} | {row['mask_recall']} "
                 f"| {row['mask_f1']} | {row['images']} | {row['timestamp']} |\n")
        append_table_row(md, "Segmentation mask metrics (real per-pixel mask IoU)",
                         mask_header, mline)
        print(f"[i] appended mask-metric row to {md}")


if __name__ == "__main__":
    main()
