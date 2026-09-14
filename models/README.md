# Vision assets

[Repository home](../README.md) · [Vision deployment](../deployment/README.md)

Model binaries are distributed through the [Argus Omnimotus model repository](https://huggingface.co/moeen14/argus-omnimotus-vision-models). The [labeled evaluation dataset](https://huggingface.co/datasets/moeen14/argus-omnimotus-vision-dataset) is available separately. This directory contains the SHA-256 and size manifest for the six ONNX files used by the project. Download the files into this directory before verification and engine building.

```bash
python -m pip install -r requirements-models.txt
python tools/models.py download --repo-id moeen14/argus-omnimotus-vision-models --revision main
python tools/models.py verify
python tools/models.py build
```

The download command checks every file against `manifest.json`. `build` runs `trtexec --fp16` on the target Jetson. Use `--directory` for another asset folder and set the same folder in `robot.json`. Repeat `--model FILENAME.onnx` to select individual models.

| ONNX file | Role |
|---|---|
| `crop_detector.onnx` | Bottom-camera crop/platform identification |
| `highcam_platform.onnx` | Overhead platform/ball verification |
| `highcam_rail.onnx` | Current rail segmentation and BEV lane following |
| `lanesegyv8.onnx` | Earlier lane segmentation utilities |
| `resnet18_image_encoder.onnx` | Optional NanoSAM image encoder |
| `mobile_sam_mask_decoder.onnx` | Optional NanoSAM decoder, kept as ONNX |

The field task needs rail and crop engines, plus the overhead platform engine when `--verification on` is selected. NanoSAM is optional. The decoder is excluded from TensorRT building because the existing NanoSAM script runs it with ONNX Runtime. TensorRT engines depend on the target environment; see [NVIDIA's compatibility documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html). Engine builds and inference must be verified on the target Jetson.

The runtime crop detector uses four classes: `gCrop`, `platform`, `ycrop`, `servo`. The overhead detector uses `ball`, `gcrop`, `platform`, `ycrop`. The separate edge benchmark export declares only three crop-detector classes. Do not substitute benchmark models for robot models based on matching filenames.

The Hugging Face model and dataset cards document class order, preprocessing, release contents, scope, attribution, and licensing. Model downloads use the [Hugging Face Hub API](https://huggingface.co/docs/huggingface_hub/guides/download).
