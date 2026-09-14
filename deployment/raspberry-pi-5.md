# Raspberry Pi 5: CPU vision

Use 64-bit Raspberry Pi OS with Python 3.10–3.12 (Bookworm/Python 3.11 is a suitable environment for these package pins). Create an environment separate from the robot dashboard:

```bash
sudo apt install python3-venv libgl1 libglib2.0-0
python3 -m venv .venv-vision
source .venv-vision/bin/activate
python -m pip install -r deployment/requirements-cpu.txt
python deployment/launch_vision.py pi5 --check
```

Obtain matching NCNN `.param` and `.bin` files plus labeled benchmark images from the project's Hugging Face release:

```bash
python deployment/launch_vision.py pi5 --model /path/to/platform.param --data /path/to/platform-dataset --task highcam_platform --onnx-ref /path/to/platform.onnx
```

For conversion, use `deployment/tools/ncnn_compile.py` on a build host with pnnx and the NCNN conversion tools:

```bash
python deployment/tools/ncnn_compile.py --onnx /path/to/platform.onnx --calib /path/to/platform-dataset/images --out /path/to/ncnn-output --tools /path/to/ncnn-tools --pnnx /path/to/pnnx
```

The converter creates the backend assets; it does not train the model. NCNN build instructions are maintained by [Tencent/NCNN](https://github.com/Tencent/ncnn/wiki/how-to-build).

[Deployment targets](README.md)
