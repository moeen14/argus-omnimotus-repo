# Serial protocol

[Firmware](firmware.md) · [Wiring](wiring.md)

Each board uses USB serial at 921600 baud. Send ASCII commands terminated by a newline. Responses, telemetry and asynchronous completion messages may interleave; clients must parse records rather than assume the next line answers the last command. The sketches remain the complete protocol specification.

## Board 01

Streams `RPM`, `TOF` and `IMU` records. Startup/status messages include firmware and sensor initialization state. OLED commands include `I`, `T`, `T0`, `G`, `B`, `R` and `Z`; `F` resets ToF initialization. `SG90,<angle>` positions the board's SG90. Use the dashboard's existing parser and message handling as the reference client.

## Board 02

| Command | Meaning |
|---|---|
| `P` | Request `POS` for servos 1–9 |
| `MODE,initial,BR,FR,BL,FL` | Apply explicit steering targets; mode may also be `mid` or `perpendicular` |
| `I`, `M`, `T` | Legacy initial, mid, perpendicular presets |
| `F<pwm>`, `B<pwm>` | Forward/reverse in initial mode |
| `L<pwm>`, `R<pwm>` | Lateral movement in perpendicular mode |
| `CW<pwm>`, `CCW<pwm>` | Rotation in mid mode |
| `Fbr,fr,bl,fl` and analogous motion commands | Individual wheel PWM values |
| `DR<delta>`, `DL<delta>`, `DF<delta>`, `DB<delta>` | Mode-dependent steering adjustment |
| `X` | Stop drive motors |
| `5 <angle>` through `9 <angle>` | Position a mechanism servo |
| `G`, `B`, `R` | RGB indication |
| `Z` | Startup RGB sequence |
| `D` | One dispenser cycle |
| `D0`, `RESET` | Reset dispenser memory / empty flags |
| `STATUS`, `L`, `A` | Dispenser state, LDR read, toggle LDR stream |
| `S<angle>`, `STEST` | Dispenser SG90 positioning / sweep |

Bare `B`, `R` and `L` have different meanings from movement commands with numeric payloads. Motion is gated by the current swerve mode. Steering command order is BR, FR, BL, FL, which differs from physical servo ID order 1, 2, 3, 4.

Servos 5–9 produce `ACK,SERVO,...` on command acceptance and unsolicited `DONE,<id>,<reason>` when the motion ends. A `timeout` reason is not proof that the target was reached. The firmware move timeout is 4000 ms. Preserve this distinction when extending plan–act–sense–verify behavior.
