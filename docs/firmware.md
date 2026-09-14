# ESP32 DevKit firmware

[Repository home](../README.md) · [Wiring reference](wiring.md)

The build target is the classic ESP32 DevKit (`esp32:esp32:esp32`, Arduino's ESP32 Dev Module profile). Confirm flash/board settings for your specific DevKit; ESP32-S3/C3 are different targets.

Install [Arduino CLI](https://arduino.github.io/arduino-cli/) and Python 3.10 or later. From the repository root:

```bash
python tools/firmware.py install
python tools/firmware.py compile
arduino-cli board list
python tools/firmware.py upload --board board01 --port /dev/ttyUSB0
python tools/firmware.py upload --board board02 --port /dev/ttyUSB1
```

Substitute the actual ports, such as `COM5` on Windows. Close serial monitors and control programs before uploading. Uploading compiles first and requires an explicit board and port; it never guesses which board to flash. Each sketch can also be opened in Arduino IDE from its same-named directory.

Core and library versions are defined in `tools/firmware.py`: ESP32 core 3.3.0; ESP32Servo 3.0.9; VL53L1X 1.3.1; Adafruit LSM6DS 4.7.4, LIS3MDL 1.2.4, Unified Sensor 1.1.15, SSD1306 2.5.15, GFX 1.12.6 and BusIO 1.17.4; SCServo 1.0.2. These versions compile with the packaged sketches. The original export did not contain its Arduino library lockfile, so this build establishes a reproducible compile configuration, not confirmation of identical original library binaries.

Board 02 uses the pin-based LEDC API introduced in [Arduino ESP32 3.x](https://docs.espressif.com/projects/arduino-esp32/en/latest/migration_guides/2.x_to_3.0.html). Core 2.x is unsuitable without porting those calls.

Board 01 identifies as `board01_final_v4`; Board 02 as `board02_final_v10`. Serial communication is newline-terminated ASCII at 921600 baud. See [protocol](protocol.md) and [calibration](bring-up.md).
