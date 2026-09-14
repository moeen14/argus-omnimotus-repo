# Main applications

Run commands from the repository root after completing [setup](jetson-setup.md), [model deployment](../models/README.md), and [calibration](bring-up.md). Edit `robot.json` first. Launcher options go before the application name.

| Application | Launch command | Original source |
|---|---|---|
| Operator dashboard | `python tools/run.py dashboard` | `robot/main.py` |
| Crop identification | `python tools/run.py identify` | `robot/identification.py` |
| Navigation without crop tasks | `python tools/run.py navigate` | `Tests/navigation_test.py` |
| Full-field navigation and tasks | `python tools/run.py field` | `Tests/main_task.py` |
| Repeated single-lane tasks | `python tools/run.py lane` | `Tests/lane_platform_alignment_test.py` |

The five primary programs are located directly in `applications/` with names that describe their roles.

## Optional field verification

```bash
python tools/run.py --verification on field
python tools/run.py --verification off field
python tools/run.py --dry-run --verification on field
```

Verification means the existing camera-based ball-placement check after dispensing, including its camera repositioning and retry behavior. It does not disable or enable every sensor feedback check in the controller. The original default remains **off**. When off, the verification TensorRT engine is not loaded; rail and crop engines are still required. The switch applies only to `field` and its compatibility alias `autonomy`.

You can also set `ASABE_ENABLE_BALL_VERIFY` to `1` or `0` in the config's environment dictionary. The command-line flag overrides that setting. `autonomy` remains an alias for `field` so older commands still work.

## Before motion

The dashboard is a local Tkinter interface; the identification and autonomous programs use camera windows and TensorRT. Applications can reposition mechanisms at startup. Run only one application owning the serial ports at a time. Navigation without crop actuation still steers wheels and positions cameras.

Use a dry run to inspect configuration without opening hardware. Test a short controlled sequence after calibration before a complete field run. Mission geometry and controller gains remain in the application source; the launcher does not make the original testbed parameters universal.

[Component tuning](../tuning/README.md) ? [Vision deployment](../deployment/README.md) ? [Repository home](../README.md)
