# Bring-up and calibration

1. Check the [firmware-derived wiring](wiring.md), board roles and power distribution against the actual robot. Verify the chosen servo supply, motor supply and common ground using the hardware design; GPIO tables alone are not a complete electrical schematic.
2. Flash each DevKit and verify serial identities at 921600 baud. With actuator power isolated, check Board 01's sensor telemetry and both camera feeds.
3. Configure stable serial/camera paths and matching model files. Preview the launch command with `python tools/run.py --dry-run dashboard`.
4. Raise the wheels clear of the ground and keep mechanisms clear before enabling actuator power. Dashboard and task initialization can reposition servos. Keep a physical power disconnect reachable; `X` stops drive motors but is not a complete mechanism emergency stop.
5. Verify wheel order **BR, FR, BL, FL** and individual motor signs. Wheel servo IDs are **1, 2, 4, 3** in that order. Calibrate Board 02's `SAFE_ARCS`, steering preset angles and motor sign arrays to the installed geometry before driving. Do not treat a timed-out servo move as successful arrival.
6. Calibrate camera intrinsics/BEV source points, ToF placement and offsets, wheel encoder directions and motor thresholds. The source settings are for the original platform. Use the individual [programs](programs.md), then validate a short manual movement and stop.
7. Validate crop/platform class ordering with the matched model before enabling closed-loop manipulation. Check single alignment and dispense cycles before running a complete course.
8. Set course, clearance and actuation constants in `applications/field_tasks.py` for your testbed. Launch `python tools/run.py autonomy` only after the component checks pass.

Run one process owning the serial ports at a time. Stop the dashboard before starting autonomy or a hardware test. The scripts do not implement shared serial arbitration.

Controller constants remain in their original modules to preserve the experimental implementation. The JSON config centralizes device names and model location; it does not yet expose every gain, geometry or mission parameter.

[Repository home](../README.md) · [Project resources](project-links.md)
