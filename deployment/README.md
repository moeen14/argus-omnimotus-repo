# Choose a deployment target

| Target | Vision backend | Start here |
|---|---|---|
| Raspberry Pi 5 CPU | NCNN `.param` + `.bin` | [Pi 5](raspberry-pi-5.md) |
| Raspberry Pi 5 + Hailo-8 | HailoRT `.hef` | [Pi 5 + Hailo-8](raspberry-pi-5-hailo8.md) |
| Jetson Orin Nano | TensorRT `.engine` | [Orin Nano](jetson-orin-nano.md) |

All three have a launch path through `launch_vision.py` and the shared evaluator. It runs inference on labeled dataset images and writes metrics; it is not a robot-motion launcher or a live-camera dashboard. Individual live-camera TensorRT utilities are in [vision](../vision/README.md).

The five robot applications use the [robot launcher](../docs/programs.md). Dashboard control uses serial, Tkinter and OpenCV; identification and autonomous applications currently require TensorRT/PyCUDA. Pi/Hailo vision support does not mean the full autonomous robot controller has been ported to those backends. Orin Nano requires target-device validation; the source robot used AGX Orin.

Weights and images belong on [Hugging Face](../docs/project-links.md). Keep a separate benchmark asset directory: its crop model has three classes, while the robot crop model has four. Do not interchange the exports. The deployment launchers never flash boards or start motors.

[Repository home](../README.md)
