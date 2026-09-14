# Vision assets

[Project resources and Hugging Face links](../docs/project-links.md) · [Repository home](../README.md)

Model binaries and image datasets belong on Hugging Face. This directory contains the SHA-256 and size manifest for six ONNX files from the robot export. Place those exact files at the root of the eventual Hugging Face model repository, or copy them here locally before checking them.

```bash
python -m pip install -r requirements-models.txt
python tools/models.py download --repo-id OWNER/MODEL_REPOSITORY --revision COMMIT_OR_TAG
python tools/models.py verify
python tools/models.py build
```

Replace the repository and revision placeholders after the models are published. The download command checks every file against `manifest.json`. `build` runs `trtexec --fp16` on the target Jetson; it never downloads a prebuilt engine. Use `--directory` for another asset folder and set the same folder in `config/robot.json`. Use repeated `--model FILENAME.onnx` arguments to select individual models.

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

When publishing on Hugging Face, include model cards with architecture, source weights, class order, preprocessing, input/output shapes, training data, evaluation scope, license and this code repository/homepage links. For datasets, include the split definitions, label schema, provenance, license and a dataset card. Model download behavior follows [Hugging Face Hub's download API](https://huggingface.co/docs/huggingface_hub/guides/download).
