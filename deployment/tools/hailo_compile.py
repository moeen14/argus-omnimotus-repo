import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
from yolo_common import preprocess_u8_nhwc


def find_endnodes(onnx_path):

    import onnx
    m = onnx.load(onnx_path)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    det = sorted(n.name for n in g.node
                 if re.search(r"/(cv2|cv3)\.\d+/.*\.2/Conv$", n.name))
    is_seg = any(o.name == "output1" for o in g.output)
    ends = list(det)
    if is_seg:
        ends += sorted(n.name for n in g.node
                       if re.search(r"/cv4\.\d+/.*\.2/Conv$", n.name))

        cur = prod["output1"]
        seen = set()
        while cur is not None and cur.op_type != "Conv" and cur.name not in seen:
            seen.add(cur.name)
            cur = prod.get(cur.input[0])
        if cur is not None:
            ends.append(cur.name)
    if not det:
        raise SystemExit(f"[!] could not locate cv2/cv3 head convs in {onnx_path}")
    return ends, is_seg


def build_calib(calib_dir, imgsz, limit):
    files = sorted(glob.glob(os.path.join(calib_dir, "*")))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"[!] no calib images in {calib_dir}")
    arr = np.concatenate([preprocess_u8_nhwc(f, imgsz)[0] for f in files], 0)
    print(f"[i] calib set: {arr.shape} uint8 from {len(files)} images")
    return arr.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True, help="directory of calibration images")
    ap.add_argument("--nc", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="hailo8")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib-limit", type=int, default=128)
    ap.add_argument("--optimization-level", type=int, default=2)
    args = ap.parse_args()

    from hailo_sdk_client import ClientRunner

    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.onnx))[0]
    task = os.path.basename(os.path.dirname(args.onnx))
    stem = f"{task}__{name}"
    har = os.path.join(args.out, stem + ".har")
    hef = os.path.join(args.out, stem + ".hef")
    meta_path = os.path.join(args.out, stem + ".meta.json")
    log = open(os.path.join(args.out, stem + ".compile.log"), "w")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg)
        log.write(msg + "\n")
        log.flush()

    t0 = time.time()
    ends, is_seg = find_endnodes(args.onnx)
    say(f"[i] {stem}  seg={is_seg}")
    say(f"[i] end nodes ({len(ends)}):")
    for e in ends:
        say("      ", e)

    runner = ClientRunner(hw_arch=args.arch)
    say("[i] translate_onnx_model ...")
    runner.translate_onnx_model(
        args.onnx, stem,
        start_node_names=["images"],
        end_node_names=ends,
        net_input_shapes={"images": [1, 3, args.imgsz, args.imgsz]},
    )


    alls = f'normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n'
    alls += f'model_optimization_flavor(optimization_level={args.optimization_level}, compression_level=0)\n'
    runner.load_model_script(alls)

    calib = build_calib(args.calib, args.imgsz, args.calib_limit)
    say("[i] optimize (INT8 PTQ) ...")
    runner.optimize(calib)
    runner.save_har(har)
    say(f"[i] saved {har}")

    say("[i] compile ...")
    hef_bytes = runner.compile()
    with open(hef, "wb") as f:
        f.write(hef_bytes)
    say(f"[i] saved {hef}  ({len(hef_bytes)/1e6:.2f} MB)")

    json.dump({
        "stem": stem, "task": task, "model": name, "arch": args.arch,
        "num_classes": args.nc, "imgsz": args.imgsz, "segmentation": is_seg,
        "end_nodes": ends, "calib_images": int(len(calib)),
        "input_layout": "NHWC uint8 RGB (net applies /255)",
        "compile_seconds": round(time.time() - t0, 1),
    }, open(meta_path, "w"), indent=2)
    say(f"[i] done in {time.time()-t0:.0f}s -> {meta_path}")


if __name__ == "__main__":
    main()
