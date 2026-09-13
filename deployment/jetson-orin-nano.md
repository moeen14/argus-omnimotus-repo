# Jetson Orin Nano

Install the board firmware and JetPack using [NVIDIA's Orin Nano setup guide](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/software_setup.html). For this source, use a JetPack 6 / TensorRT 10 environment compatible with the [recorded Jetson stack](../docs/jetson-setup.md), rather than assuming the newest major release preserves its APIs.

Follow the Python, serial and camera setup in that guide. The original robot ran on AGX Orin 64 GB; Orin Nano has not been physically validated in this packaging work. Build engines on the Nano and check memory use and control-loop timing before driving.

## Robot applications

```bash
python tools/models.py verify
python tools/models.py build --model crop_detector.onnx --model highcam_rail.onnx --model highcam_platform.onnx
python tools/run.py --dry-run dashboard
python tools/run.py dashboard
```

Supply the ONNX files first using the [model download guide](../models/README.md). Finish [bring-up and calibration](../docs/bring-up.md) before launching `navigate`, `field` or `lane`. Run only one serial-owning application at a time.

## Vision evaluation

Use the **benchmark** ONNX export and dataset, not the robot crop-model manifest. Create an FP16 engine on the Nano:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=/path/to/platform.onnx --saveEngine=/path/to/platform.engine --fp16
python deployment/launch_vision.py orin-nano --check
python deployment/launch_vision.py orin-nano --model /path/to/platform.engine --data /path/to/platform-dataset --task highcam_platform --onnx-ref /path/to/platform.onnx
```

The evaluator additionally uses `onnx` for model metadata; install it in the selected environment. Optional INT8 calibration is in `experiments/edge/scripts/build_tensorrt.py`. Existing engines from AGX Orin or another TensorRT release are not the portable release format.

[Deployment targets](README.md)
