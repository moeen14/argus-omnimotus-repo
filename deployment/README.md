# Choose a deployment target

| Target | Vision backend | Start here |
|---|---|---|
| Raspberry Pi 5 CPU | NCNN `.param` + `.bin` | [Pi 5](raspberry-pi-5.md) |
| Raspberry Pi 5 + Hailo-8 | HailoRT `.hef` | [Pi 5 + Hailo-8](raspberry-pi-5-hailo8.md) |
| Jetson Orin Nano | TensorRT `.engine` | [Orin Nano](jetson-orin-nano.md) |

All three use `launch_vision.py` and the shared evaluator. It runs inference on labeled dataset images and writes metrics. Individual live-camera TensorRT utilities are in [vision](../vision/README.md).

The five robot applications use the [robot launcher](../docs/programs.md). Dashboard control uses serial, Tkinter and OpenCV; identification and autonomous applications use TensorRT/PyCUDA. The Pi and Hailo paths provide vision evaluation, while full autonomous control uses the Jetson implementation.

Weights and images are distributed through Hugging Face. Keep benchmark assets separate from runtime assets: the benchmark crop model has three classes, while the robot crop model has four.

[Repository home](../README.md)
