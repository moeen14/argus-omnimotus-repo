# Argus Omnimotus

Argus Omnimotus is an open-source, modular agricultural robot for autonomy research and robotics education. The platform combines four-wheel independent swerve drive, distributed ESP32 control, multimodal sensing, interchangeable edge computers, and configurable manipulation in a sense-plan-act-verify architecture.

![Argus Omnimotus assembly views](hardware/Assembly_Drawing.jpg)

## Repository structure

```text
argus-omnimotus-repo/
|-- applications/   Five complete robot applications
|-- tuning/         Isolated hardware and controller tests
|-- firmware/       ESP32 sensor-board and motion-board firmware
|-- hardware/       CAD, assembly drawing, circuit diagram, BOM and pin map
|-- vision/         Standalone Jetson vision programs
|-- deployment/     Pi 5, Hailo-8 and Jetson deployment tools
|-- models/         Model manifest and Hugging Face download/build utility
|-- docs/           Setup, operation and protocol documentation
|-- tools/          Application, firmware and model launchers
`-- tests/          Hardware-independent repository tests
```

## Main applications

| Command | Program | Purpose |
|---|---|---|
| `dashboard` | `applications/operator_dashboard.py` | Operator dashboard, manual control and telemetry |
| `identify` | `applications/crop_identification.py` | Bottom-camera crop and platform identification |
| `navigate` | `applications/navigate_testbed.py` | Testbed navigation without crop actuation |
| `field` | `applications/field_tasks.py` | Full-field navigation and actuation |
| `lane` | `applications/single_lane_tasks.py` | Repeated navigation and tasks on one lane |

The complete application guide, including optional task verification, is in [docs/programs.md](docs/programs.md). Component-level programs are listed in [tuning/README.md](tuning/README.md).

## Build and run

1. Review the [hardware files](hardware/README.md) and assemble the platform.
2. Build and flash both ESP32 boards using [docs/firmware.md](docs/firmware.md).
3. Configure the Jetson using [docs/jetson-setup.md](docs/jetson-setup.md).
4. Obtain the ONNX weights using [models/README.md](models/README.md), then build TensorRT engines on the Jetson.
5. Copy and edit the robot configuration, complete [calibration](docs/bring-up.md), and launch an application.

```bash
cp robot.example.json robot.json
python tools/run.py --dry-run dashboard
python tools/run.py dashboard
```

The software defaults match the original platform. Confirm device paths, steering angles, wheel directions, camera geometry, controller gains, and mission settings before operating another build.

## Vision deployment

| Target | Backend | Guide |
|---|---|---|
| Raspberry Pi 5 | NCNN | [deployment/raspberry-pi-5.md](deployment/raspberry-pi-5.md) |
| Raspberry Pi 5 with Hailo-8 | HailoRT | [deployment/raspberry-pi-5-hailo8.md](deployment/raspberry-pi-5-hailo8.md) |
| Jetson Orin Nano | TensorRT | [deployment/jetson-orin-nano.md](deployment/jetson-orin-nano.md) |

These targets share the inference evaluator in `deployment/tools/`. The complete autonomous robot applications use the Jetson TensorRT/CUDA implementation. Download the [vision models](https://huggingface.co/moeen14/argus-omnimotus-vision-models) and [labeled evaluation dataset](https://huggingface.co/datasets/moeen14/argus-omnimotus-vision-dataset) from Hugging Face.

## Citation

Please cite the project using [CITATION.cff](CITATION.cff). Contributions are described in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Argus Omnimotus is released under the [MIT License](LICENSE).
