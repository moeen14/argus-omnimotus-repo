# Argus Omnimotus hardware

Mechanical designs, electrical connections and component records for the agricultural swerve-drive robot. This folder is part of the [combined hardware and software repository](../README.md).

**[Project homepage](docs/project-links.md#project-homepage) · [Robot code and firmware](docs/project-links.md#code) · [Vision models](docs/project-links.md#vision-models) · [Image datasets](docs/project-links.md#image-datasets) · [Paper](docs/project-links.md#paper)**

| Folder | Contents and status |
|---|---|
| [cad](cad/README.md) | SolidWorks master multibody model and assembly drawing, with locations reserved for exchange and printable files |
| [circuits](circuits/README.md) | Firmware-derived signal map; complete schematic and harness drawings still need to be supplied |
| [bom](bom/README.md) | Source-derived component inventory with unverified purchasing details explicitly identified |

## Bill of materials at a glance

| Qty. | Component | Function / location |
|---:|---|---|
| 1 | NVIDIA Jetson AGX Orin 64 GB | High-level onboard computer |
| 2 | ESP32 development boards | Distributed sensing and actuator control |
| 4 | 12 V DC gearmotors | Wheel propulsion |
| 2 | L298N dual H-bridge modules | Drive-motor control |
| 4 | AS5048-series SPI magnetic encoders | Wheel-speed feedback |
| 7 | VL53L1X time-of-flight sensors | Ranging, alignment, and obstacle detection |
| 1 | LSM6DSOX IMU | Acceleration and angular-rate sensing |
| 1 | LIS3MDL magnetometer | Magnetic-heading sensing; may share the IMU breakout |
| 1 | SSD1306 128 x 64 I2C OLED | Local status display |
| 9 | STS-protocol serial-bus servos | Steering, manipulator, and camera mechanisms |
| 2 | SG90 micro servos | Distal flap and seed dispenser |
| 1 | Light-dependent resistor (LDR) | Dispenser sensing |
| 1 | RGB indicator | Local status indication |
| 2 | USB cameras | Lower and overhead vision |
| 2 | 5 mm steel rods | Reinforcement for the printed chassis |

The detailed, machine-readable inventory is in [`bom/components.csv`](bom/components.csv). This is a source-derived component inventory, not yet a complete purchasing BOM: exact motor, servo, camera, and breakout-board SKUs remain to be verified, and batteries, power distribution, wheels, bearings, belts, gears, shafts, fasteners, wiring, connectors, and protection hardware still need to be itemized.

## 3D-printed parts

The chassis is primarily made from 3D-printed PLA components and reinforced with two 5 mm steel rods. The editable printed geometry is included in the SolidWorks master multibody part under [`cad/native`](cad/native/Argus_Omnimotus_Master_Multibody.SLDPRT). Separate printable STL files, per-part quantities, print orientations, infill settings, and support requirements have not yet been exported or verified; consult the [CAD notes](cad/README.md) before fabrication.

Use these hardware files with the [bring-up and calibration guide](../docs/bring-up.md) and [ESP32 firmware setup](../docs/firmware.md) in this same repository. The available pin map does not specify complete power distribution, connector orientation or electrical protection.

Please credit the project in work using this platform. Citation details are in [CITATION.cff](../CITATION.cff). Hardware licensing remains to be selected; see [LICENSE-STATUS.md](LICENSE-STATUS.md).
