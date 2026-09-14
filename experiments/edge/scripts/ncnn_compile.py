import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
from yolo_common import letterbox


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def make_calib_images(src_dir, dst_dir, imgsz, limit):
    import cv2
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(src_dir, "*"))
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))[:limit]
    listing = os.path.join(dst_dir, "filelist.txt")
    with open(listing, "w") as lf:
        for f in files:
            im = cv2.imread(f)
            im, _, _ = letterbox(im, imgsz)
            out = os.path.join(dst_dir, os.path.splitext(os.path.basename(f))[0] + ".jpg")
            cv2.imwrite(out, im)
            lf.write(os.path.abspath(out) + "\n")
    print(f"[i] {len(files)} letterboxed calib images -> {dst_dir}")
    return listing


def make_mixed_precision_table(param_path, table_path, out_path):


    layers, producer = [], {}
    with open(param_path) as f:
        lines = f.read().splitlines()[2:]
    for line in lines:
        v = line.split()
        if len(v) < 4:
            continue
        typ, name, nb, nt = v[0], v[1], int(v[2]), int(v[3])
        bottoms = v[4:4 + nb]
        tops = v[4 + nb:4 + nb + nt]
        params = v[4 + nb + nt:]
        rec = {"type": typ, "name": name, "bottoms": bottoms,
               "tops": tops, "params": params}
        layers.append(rec)
        for blob in tops:
            producer[blob] = rec

    keep_float = set()
    for layer in layers:
        if layer["type"] != "Reshape":
            continue
        attrs = dict(p.split("=", 1) for p in layer["params"] if "=" in p)
        cells = int(attrs.get("0", "0"))
        channels = int(attrs.get("1", "0"))
        if cells not in (400, 1600, 6400) or not (channels == 32 or 65 <= channels <= 160):
            continue
        parent = producer.get(layer["bottoms"][0])
        if not parent:
            continue
        parents = [parent]
        if parent["type"] == "Concat":
            parents = [producer.get(b) for b in parent["bottoms"]]
        keep_float.update(p["name"] for p in parents if p and p["type"] == "Convolution")


    for layer in layers:
        if layer["type"] == "Convolution" and "0=1" in layer["params"]:
            keep_float.add(layer["name"])

    with open(table_path) as src, open(out_path, "w") as dst:
        for line in src:
            key = line.split(None, 1)[0] if line.split() else ""
            layer_name = key.split("_param_", 1)[0]
            if layer_name not in keep_float:
                dst.write(line)
    return sorted(keep_float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", default=os.environ.get("NCNN_TOOLS", ""),
                    help="dir with ncnnoptimize / ncnn2table / ncnn2int8 (x86)")
    ap.add_argument("--pnnx", default=(shutil.which("pnnx")
                    or os.path.join(os.path.dirname(sys.executable), "pnnx")))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib-limit", type=int, default=200)
    ap.add_argument("--method", default="aciq", choices=["aciq", "kl", "eq"])
    args = ap.parse_args()

    def tool(name):
        p = os.path.join(args.tools, name) if args.tools else shutil.which(name)
        if not p or not os.path.exists(p):
            raise SystemExit(f"[!] {name} not found (pass --tools <ncnn bin dir>)")
        return p

    os.makedirs(args.out, exist_ok=True)
    task = os.path.basename(os.path.dirname(args.onnx))
    name = os.path.splitext(os.path.basename(args.onnx))[0]
    stem = f"{task}__{name}"
    t0 = time.time()

    with tempfile.TemporaryDirectory() as td:
        work = os.path.join(td, "m.onnx")
        shutil.copy(args.onnx, work)


        run([args.pnnx, work, f"inputshape=[1,3,{args.imgsz},{args.imgsz}]",
             "fp16=0", "optlevel=2",
             f"ncnnparam={td}/base.param", f"ncnnbin={td}/base.bin",
             f"pnnxparam={td}/x.pnnx.param", f"pnnxbin={td}/x.pnnx.bin",
             f"pnnxpy={td}/x_pnnx.py", f"pnnxonnx={td}/x.pnnx.onnx",
             f"ncnnpy={td}/x_ncnn.py"], cwd=td)
        base_p, base_b = f"{td}/base.param", f"{td}/base.bin"


        fp16_p = os.path.join(args.out, stem + "_fp16.param")
        fp16_b = os.path.join(args.out, stem + "_fp16.bin")
        run([tool("ncnnoptimize"), base_p, base_b, fp16_p, fp16_b, 1])


        fp32_p = os.path.join(args.out, stem + "_fp32.param")
        fp32_b = os.path.join(args.out, stem + "_fp32.bin")
        run([tool("ncnnoptimize"), base_p, base_b, fp32_p, fp32_b, 0])


        calib_dir = os.path.join(td, "calib")
        listing = make_calib_images(args.calib, calib_dir, args.imgsz, args.calib_limit)
        table = os.path.join(args.out, stem + ".table")
        run([tool("ncnn2table"), fp32_p, fp32_b, listing, table,
             f"shape=[{args.imgsz},{args.imgsz},3]", "pixel=RGB",
             "mean=[0.0,0.0,0.0]", "norm=[0.003921569,0.003921569,0.003921569]",
             f"method={args.method}", "thread=8"])


        int8_p = os.path.join(args.out, stem + "_int8.param")
        int8_b = os.path.join(args.out, stem + "_int8.bin")
        run([tool("ncnn2int8"), fp32_p, fp32_b, int8_p, int8_b, table])


        mixed_table = os.path.join(args.out, stem + "_int8mixed.table")
        float_layers = make_mixed_precision_table(fp32_p, table, mixed_table)
        mixed_p = os.path.join(args.out, stem + "_int8mixed.param")
        mixed_b = os.path.join(args.out, stem + "_int8mixed.bin")
        print(f"[i] mixed precision keeps float: {', '.join(float_layers)}")
        run([tool("ncnn2int8"), fp32_p, fp32_b, mixed_p, mixed_b, mixed_table])

    meta = {
        "stem": stem, "task": task, "model": name, "imgsz": args.imgsz,
        "input_layout": "NCHW float, RGB, /255 (mean 0, norm 1/255)",
        "calib_method": args.method,
        "files": {k: os.path.basename(v) for k, v in {
            "fp32": fp32_p, "fp16": fp16_p, "int8": int8_p,
            "int8mixed": mixed_p, "table": table,
            "mixed_table": mixed_table}.items()},
        "mixed_float_layers": float_layers,
        "sizes_MB": {p: round(os.path.getsize(os.path.join(args.out, p)) / 1e6, 2)
                     for p in (os.path.basename(int8_b), os.path.basename(mixed_b),
                               os.path.basename(fp16_b), os.path.basename(fp32_b))},
        "convert_seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(os.path.join(args.out, stem + ".meta.json"), "w"), indent=2)
    print(f"[i] {stem} done in {time.time()-t0:.0f}s")
    for suf in ("_fp32", "_fp16", "_int8", "_int8mixed"):
        print(f"    {stem}{suf}.param  "
              f"{os.path.getsize(os.path.join(args.out, stem + suf + '.bin'))/1e6:.2f} MB bin")


if __name__ == "__main__":
    main()
