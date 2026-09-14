# Packaging validation

## Source inventory

The release contains 41 distinct imported Python modules and two firmware sketches, drawn from 49 source records. Identical copies map to a single destination in [source-manifest.json](source-manifest.json). Different dashboard/task/manuscript variants remain separate. The manifest stores SHA-256 hashes of original source files to make provenance traceable.

Python comments and documentation strings were removed with syntax-tree equivalence checks that ignore documentation strings. Firmware comment removal preserved executable statements; both resulting sketches compiled. Three camera utilities were then updated to accept `ASABE_CROP_ENGINE` or `ASABE_PLATFORM_ENGINE` while retaining their original fallback paths; these exceptions are recorded in the manifest. No controller gains, motion sequences or servo limits were changed during packaging.

The new launcher supplies device/model configuration without importing the hardware programs during a dry run. Model utilities verify the six exported ONNX files by size and SHA-256. Source-derived wiring records are provided in the root documentation and hardware folder.

The main programs now live in `applications/`; component tools are in `tuning/`, and older variants in `archive/`. The field application's existing ball-verification switch is configurable at launch, preserving the default of off, and its verification engine is loaded only when enabled. The interlane tuning utility imports the renamed navigation application. The crop identifier's fallback model location now points to `models/`. See the source manifest for these packaging changes.

Deployment launch commands are provided for Pi 5/NCNN, Pi 5/Hailo-8 and Orin Nano/TensorRT using the existing vision evaluator. Their command construction is tested without GPU/NPU imports. Installation and inference on these three devices have not been tested here; Pi/Hailo full robot autonomy is not implemented by this reorganization.

## Local verification

- Both sketches compiled for `esp32:esp32:esp32` with core 3.3.0 and the library versions in `tools/firmware.py`.
- Board 01: 382627 bytes flash and 23536 bytes global RAM.
- Board 02: 339683 bytes flash and 21680 bytes global RAM.
- Python sources pass syntax compilation; hardware-independent tests check source availability, launcher configuration/path handling, model checksum failures and resource/documentation links.
- Model assets from the original export were checked against the manifest; no model weights, engines, datasets, recorded video or environment archive are included in the code repository.

## Limits and discrepancies

Hardware execution, TensorRT engine building, camera inference and fresh Jetson installation were not tested in this Windows packaging environment. These require the target device and robot.

The exported dashboard is Tkinter, not a browser dashboard. Board 01 owns the encoders; Board 02 owns the LDR. This differs from parts of the manuscript prose. The old deployment note's five-servo count omitted the four steering servos; the firmware controls nine bus servos. The edge benchmark crop detector's three classes also differ from the runtime crop detector's four classes.

The hardware folder now includes the SolidWorks master multibody model and assembly drawing. Printable exports, an editable electrical schematic, a final purchasing BOM and a model-training pipeline remain incomplete. Generic Pi/Hailo/NCNN benchmark scripts do not imply that the Jetson autonomy application runs on those devices.

Historical device-specific shell wrappers, installed package dumps, development logs, model binaries, dataset images and recordings are excluded from this code release. The reusable benchmark builders and evaluator are included under `experiments/edge/scripts`.

Public URLs and project licenses still need the owner's release details; see [release preparation](release.md).
