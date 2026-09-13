# Jetson setup

[Repository home](../README.md) · [Project resources](project-links.md)

## Reference environment

The exported robot used Jetson AGX Orin 64 GB, Ubuntu 22.04.5, L4T 36.4.7, Python 3.10.12, CUDA 12.6, cuDNN 9.3 and TensorRT 10.7.0.23. These are the recorded versions, not a claim that arbitrary JetPack releases are interchangeable.

Install a compatible NVIDIA JetPack/L4T image and its CUDA/TensorRT development packages using [NVIDIA's Jetson documentation](https://docs.nvidia.com/jetson/). Retain the system TensorRT Python bindings. Do not restore the original 1.4 GB virtual-environment archive or install the entire device package dump on another machine.

## Python environment

Run these commands on the Jetson from this repository's root:

```bash
sudo apt update
sudo apt install python3-venv python3-dev python3-tk python3-opencv build-essential v4l-utils
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pycuda==2026.1
python -c "import cv2, numpy, serial, tkinter, tensorrt, pycuda.driver; print(tensorrt.__version__); print(cv2.__version__)"
```

PyCUDA needs the installed CUDA headers/compiler; ensure `/usr/local/cuda/bin` is on PATH when building it. The reference environment had NumPy 1.26.4 and PySerial 3.5. OpenCV must have GUI support for the camera windows; do not replace it with a headless wheel in this environment. Utilities using ArUco also require `cv2.aruco`.

Only the optional CuPy vision utility needs `cupy-cuda12x==13.6.0`. NanoSAM and the ONNX class viewer need a Jetson-compatible ONNX Runtime installation; NanoSAM expects its CUDA execution provider. Use the NVIDIA/ONNX Runtime build appropriate to this JetPack version and verify providers with `onnxruntime.get_available_providers()`. These GPU packages are deliberately separate from the generic requirements file. Torch is not required to launch the packaged TensorRT autonomy task.

## Device configuration

```bash
ls -l /dev/serial/by-id/
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id/
cp config/robot.example.json config/robot.json
sudo usermod -aG dialout,video "$USER"
```

Log out and back in after changing groups. Edit the local JSON with stable device paths. The `/dev/video0` and `/dev/video2` values are examples; camera indexes vary. Board 01 is the sensor board; Board 02 is the actuator board. Both use 921600 baud. When USB serial identities are identical, assign stable udev aliases using each board's actual physical USB path, following [device naming](devices.md).

The launcher passes configured `ASABE_*` environment variables to each program and runs it with the model directory as its working directory. Explicit values in the JSON override shell values. An explicitly supplied engine environment variable overrides the launcher's model-directory default; use absolute engine paths. The complete source variable index is [environment-variables.json](environment-variables.json).

The dashboard and OpenCV displays require a graphical session. The remote keyboard client forwards keys to the running dashboard; it does not create a browser interface. Keep its listener on localhost and use an SSH tunnel for remote access.

Continue with [firmware](firmware.md), [models](../models/README.md), then [bring-up](bring-up.md).
