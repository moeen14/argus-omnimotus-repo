# Raspberry Pi 5 + Hailo-8

Install the accelerator and the compatible Raspberry Pi OS/HailoRT stack using the [official Raspberry Pi AI software instructions](https://www.raspberrypi.com/documentation/computers/ai.html). Hailo-8/8L hardware uses `hailo-all`; the AI HAT+ 2 has a different accelerator/software stack and is not this target.

```bash
sudo apt update
sudo apt install hailo-all python3-venv
sudo reboot
```

After reboot, activate a Python 3.10–3.12 environment that can see the system HailoRT bindings. The imported evaluator uses `InferVStreams`; verify that API exists in the installed HailoRT version:

```bash
hailortcli fw-control identify
python3 -m venv --system-site-packages .venv-hailo
source .venv-hailo/bin/activate
python -m pip install -r deployment/requirements-hailo.txt
python deployment/launch_vision.py pi5-hailo8 --check
python deployment/launch_vision.py pi5-hailo8 --model /path/to/platform.hef --data /path/to/platform-dataset --task highcam_platform --onnx-ref /path/to/platform.onnx
```

Obtain a **Hailo-8** HEF from the matching [model release](../docs/project-links.md); Hailo-8L is a different compilation target. To compile your ONNX model, use the vendor-supported Hailo Dataflow Compiler environment on its build host:

```bash
python experiments/edge/scripts/hailo_compile.py --onnx /path/to/platform.onnx --calib /path/to/platform-dataset/images --nc 4 --arch hailo8 --out /path/to/hailo-output
```

The compiler needs its vendor SDK; installing HailoRT on the Pi alone does not supply it. The shared evaluator expects the exported YOLO output layout used by this project; arbitrary precompiled HEFs or embedded-NMS outputs may require decoder changes. This path has not been tested on physical hardware during repository preparation.

[Deployment targets](README.md)
