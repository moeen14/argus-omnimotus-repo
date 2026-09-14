# Vision utilities

Standalone camera tools are separate from the five robot applications. Launch these through `tools/run.py` so model filenames resolve in the configured model folder:

```bash
python tools/run.py vision/crop_detector_cli.py
python tools/run.py vision/platform_detector_cli.py
python tools/run.py vision/rail_detector.py
```

These use the Jetson TensorRT/PyCUDA environment. `rail_detector_cu.py` is the CuPy variant; `nanosam_cli.py` uses an encoder engine and ONNX decoder. Earlier lane utilities and the distinct crop detector variant remain available for reproduction.

[Model downloads and builds](../models/README.md) | [Pi 5 / Hailo-8 / Orin Nano deployment](../deployment/README.md) | [Main applications](../docs/programs.md)
