# Argus Omnimotus

Robot software and vision deployment for the modular agricultural swerve-drive platform.

**[Project homepage](docs/project-links.md#project-homepage) ? [CAD, wiring and BOM](docs/project-links.md#hardware) ? [Hugging Face models](docs/project-links.md#vision-models) ? [Datasets](docs/project-links.md#image-datasets) ? [Paper](docs/project-links.md#paper)**

## Five main applications

| Command | File in `applications/` | What it does |
|---|---|---|
| `dashboard` | `operator_dashboard.py` | Operator dashboard for manual control and telemetry |
| `identify` | `crop_identification.py` | Bottom-camera crop/platform identification |
| `navigate` | `navigate_testbed.py` | Navigate the testbed without crop actuation tasks |
| `field` | `field_tasks.py` | Navigate the whole field and perform actuation tasks, with optional ball-placement verification |
| `lane` | `single_lane_tasks.py` | Travel back and forth along one lane, repeatedly performing tasks |

```bash
cp config/robot.example.json config/robot.json
python tools/run.py --dry-run dashboard
python tools/run.py dashboard
```

Edit device paths in the local config before running. Start one application at a time; they share the robot's serial ports. See [application instructions](docs/programs.md) for all five commands and the verification flag.

## Rebuild and run the robot

1. Obtain the [hardware designs and BOM](docs/project-links.md#hardware); check the [firmware-derived wiring](docs/wiring.md).
2. Build/flash both [ESP32 DevKit firmwares](docs/firmware.md).
3. Set up the [Jetson environment](docs/jetson-setup.md), serial ports and cameras.
4. Download [vision models from Hugging Face](models/README.md) and build engines on the target Jetson.
5. Complete [bring-up and calibration](docs/bring-up.md), then choose a main application.

Controller settings match the original robot and must be calibrated for a new build. The supplied workspace did not contain complete CAD or a circuit schematic; the hardware repository identifies those missing build assets.

## Vision deployment targets

| Hardware | Backend | Guide |
|---|---|---|
| Raspberry Pi 5 CPU | NCNN | [Pi 5 deployment](deployment/raspberry-pi-5.md) |
| Raspberry Pi 5 + Hailo-8 | HailoRT | [Hailo-8 deployment](deployment/raspberry-pi-5-hailo8.md) |
| Jetson Orin Nano | TensorRT | [Orin Nano deployment](deployment/jetson-orin-nano.md) |

All three have vision inference/evaluation launch code. The full autonomous applications currently require Jetson TensorRT/CUDA; they have not been ported to Pi/Hailo. The exported robot used AGX Orin, and Orin Nano operation still requires physical validation. Weights and images stay on Hugging Face.

## Where things live

| Folder | Purpose |
|---|---|
| `applications/` | The five main robot programs |
| [tuning/](tuning/README.md) | Component calibration and isolated activity tests |
| [vision/](vision/README.md) | Standalone camera inference utilities |
| [deployment/](deployment/README.md) | Device-specific setup and vision launch commands |
| `firmware/` | Sensor-board and actuator-board ESP32 sketches |
| `models/` | Download/build tools guide and model hashes; no weights |
| `experiments/edge/` | Shared multi-device model builders and evaluator |
| [archive/](archive/README.md) | Earlier control variants retained for reproducibility |
| `config/`, `tools/`, `tests/` | Local configuration, launch utilities and automated checks |

## Validation and release

Both ESP32 sketches compile for DevKit. Hardware-independent checks cover Python syntax, launch configuration, model checksums and repository links; they do not validate physical motion. See [validation](docs/validation.md).

Please cite [CITATION.cff](CITATION.cff) when using the project in research. Read [CONTRIBUTING.md](CONTRIBUTING.md) before modifying control code. Public links and the project license still need the owner's release details: [release preparation](docs/release.md), [license status](LICENSE-STATUS.md).
