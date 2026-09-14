# Stable serial and camera names

Use `/dev/serial/by-id` and `/dev/v4l/by-id` paths in `config/robot.json` where available. Identify boards one at a time so sensor and actuator assignments are unambiguous.

If both ESP32 USB adapters report the same serial number, inspect physical connection attributes:

```bash
udevadm info --attribute-walk --name=/dev/ttyUSB0
udevadm info --attribute-walk --name=/dev/ttyUSB1
```

Create `/etc/udev/rules.d/99-argus-esp32.rules` using the actual `KERNELS` USB topology value associated with each adapter. The following is a template, not an installable rule until the placeholders are replaced:

```text
SUBSYSTEM=="tty", KERNELS=="SENSOR_USB_PATH", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="esp32_01"
SUBSYSTEM=="tty", KERNELS=="ACTUATOR_USB_PATH", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="esp32_02"
```

The VID/PID above are from the exported robot's CP210x adapters; check yours. Reload with `sudo udevadm control --reload-rules`, then reconnect the boards. Topology aliases change meaning if cables move between hub ports. Confirm identities before enabling actuator power.

[Jetson setup](jetson-setup.md) · [Wiring](wiring.md)
