# Firmware-derived wiring reference

[Hardware and CAD](../hardware/README.md) · [Firmware setup](firmware.md)

This pin map is extracted from the packaged firmware. It is a signal reference, not a complete power, connector or protection schematic. All numbers below are ESP32 GPIO numbers, not DevKit header positions.

## Board 01: sensors

| Interface | GPIO / address |
|---|---|
| Shared I2C | SDA 21, SCL 22, 100 kHz |
| Encoder SPI | SCK 18, MISO 19, MOSI 23 |
| Encoder chip selects, BR / FR / BL / FL | 5 / 4 / 33 / 32 |
| ToF XSHUT, FRS / BRS / BLS / FLF / FLS / CAM / FRF | 13 / 14 / 16 / 17 / 25 / 26 / 27 |
| Corresponding ToF I2C addresses | 0x30 / 0x31 / 0x32 / 0x33 / 0x34 / 0x35 / 0x36 |
| Distal SG90 | 15; 50 Hz, 500–2500 μs; initial angle 180° |
| OLED | I2C, 0x3C, 128 × 64 |
| LSM6DSOX and LIS3MDL | Shared I2C bus |

`CAM` is the firmware's ToF channel label. It is not a camera data connection. Both USB cameras connect directly to the Jetson.

## Board 02: actuators

| Interface | GPIO |
|---|---|
| Bus-servo UART2 | RX 16, TX 17; connect through the appropriate bus-servo interface |
| L298N 1 IN1 / IN2 / IN3 / IN4 | 32 / 33 / 13 / 14 |
| L298N 1 ENA / ENB | 25 / 26 |
| L298N 2 IN1 / IN2 / IN3 / IN4 | 4 / 5 / 18 / 19 |
| L298N 2 ENA / ENB | 21 / 22 |
| RGB red / green / blue | 23 / 15 / 2 |
| Dispenser SG90 | 27 |
| Dispenser LDR | 35 |

Drive PWM is 1 kHz, 8-bit. Motors and serial motion commands use BR, FR, BL, FL order. Bus-servo IDs map as follows:

| ID | Mechanism |
|---|---|
| 1 / 2 / 3 / 4 | BR / FR / FL / BL wheel steering |
| 5 / 6 | Horizontal / vertical arm motion |
| 7 | Bottom camera |
| 8 / 9 | Overhead camera pan / height |

The original deployment note counted only IDs 5–9; the firmware controls **nine** bus servos. GPIO 15 appears on both boards for different functions; that is not a same-board conflict.

The manuscript's board ownership descriptions differ from the source for the wheel encoders and dispenser LDR. This release follows the firmware: **encoders on Board 01, LDR on Board 02**. Verify the physical harness and reconcile the final circuit drawing before publishing it as a build specification.
