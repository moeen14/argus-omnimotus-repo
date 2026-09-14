# Component tuning and isolated tests

These programs calibrate or test individual activities. They are separate from the five [main applications](../docs/programs.md). They may move real hardware; do not run them through unit-test discovery.

Launch from the repository root using the shared configuration:

```bash
python tools/run.py tuning/motor_threshold_test.py
python tools/run.py tuning/servo_dashboard_test.py
python tools/run.py keyboard
```

Some utilities expose their own port/camera arguments or source constants; environment configuration applies only where supported. Inspect these before running a utility. Do not assume an older program?s `--help` is hardware-free.

| Utility | Purpose |
|---|---|
| `ball_dispense_test.py` | Ball dispense |
| `bottom_cam_crop_detector_test.py` | Bottom cam crop detector |
| `crop_detector_area_test.py` | Crop detector area |
| `dual_usb_camera_viewer.py` | Dual usb camera viewer |
| `esp_serial_sender.py` | Esp serial sender |
| `front_cam_TOF_alignment_test.py` | Front cam tof alignment |
| `interlane_forward_test.py` | Interlane forward |
| `lane_alignment_test.py` | Lane alignment |
| `motor_threshold_test.py` | Motor threshold |
| `overhead_cam_highcam_platform_test.py` | Overhead cam highcam platform |
| `platform_alignment_test.py` | Platform alignment |
| `remote_keyboard.py` | Remote keyboard |
| `servo_dashboard_test.py` | Servo dashboard |
| `teleop_image_capture.py` | Teleop image capture |
| `tof_rail_alignment.py` | Tof rail alignment |
| `tof_turnaround_tuner.py` | Tof turnaround tuner |
| `topple_drop_test.py` | Topple drop |
| `wheel_balancing_test.py` | Wheel balancing |

For whole-field navigation or repeated single-lane tasks, use `navigate`, `field` or `lane` instead of combining these tuning utilities.
