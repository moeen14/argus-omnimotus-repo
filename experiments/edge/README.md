# Edge inference experiments

This is the separate multi-device benchmark harness imported from the paper workspace. Run it in a dedicated environment; its headless OpenCV and backend packages should not replace the robot dashboard environment.

The original evaluator declares `crop_detector: 3`, `highcam_platform: 4`, and `highcam_rail: 1` classes. These are benchmark-export definitions, not the four-class robot crop detector's schema. Use the corresponding benchmark dataset/model release and confirm class ordering before interpreting results.

```bash
python -m pip install -r experiments/edge/requirements.txt
python experiments/edge/scripts/evaluate.py --backend onnx --model /path/to/benchmark.onnx --onnx-ref /path/to/benchmark.onnx --data /path/to/dataset --task highcam_platform --device local-onnx
```

Datasets need the `images/` and `labels/` structure expected by `scripts/common/yolo_common.py`. Images, labels and model binaries will be distributed through [Hugging Face](../../docs/project-links.md). Raw results are written under this experiment's `results/` directory, with a generated summary under `docs/RESULTS.md`.

| Script | Purpose / runtime |
|---|---|
| `scripts/evaluate.py` | ONNX, TensorRT, Hailo and NCNN inference/evaluation |
| `scripts/build_tensorrt.py` | INT8 calibration/build on the target Jetson |
| `scripts/hailo_compile.py` | Hailo compilation with the vendor SDK and calibration images |
| `scripts/ncnn_compile.py` | NCNN conversion with pnnx and NCNN conversion tools |
| `scripts/common/yolo_common.py` | Shared preprocessing, decoding and metrics |

Install the selected backend separately on its supported platform. The benchmark builder is distinct from the runtime FP16 builder in `tools/models.py`. Historical shell wrappers tied to an individual's mount points and Conda environments are not release entry points; their Python builders/evaluator are provided here with explicit path arguments.

No training pipeline was present in the supplied robot/benchmark export. Do not describe these inference scripts as a reproducible training release until training code and dataset splits are supplied.
