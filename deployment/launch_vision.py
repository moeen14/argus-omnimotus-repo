import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    'pi5': ('ncnn', '.param', ['numpy', 'cv2', 'ncnn']),
    'pi5-hailo8': ('hailo', '.hef', ['numpy', 'cv2', 'hailo_platform']),
    'orin-nano': ('tensorrt', '.engine', ['numpy', 'cv2', 'tensorrt', 'pycuda.driver']),
}

def command_for(target, model, data, task, reference=None):
    backend, suffix, _ = TARGETS[target]
    if model.suffix != suffix:
        raise ValueError(f'{target} expects a {suffix} model.')
    command = [sys.executable, str(ROOT / 'deployment/tools/evaluate.py'),
               '--backend', backend, '--model', str(model.resolve()), '--data', str(data.resolve()),
               '--task', task, '--device', target]
    if reference:
        command += ['--onnx-ref', str(reference.resolve())]
    return command

def main():
    parser = argparse.ArgumentParser(description='Run the shared vision evaluator on Raspberry Pi 5, Hailo-8 or Jetson Orin Nano.')
    parser.add_argument('target', choices=TARGETS)
    parser.add_argument('--check', action='store_true', help='Check runtime imports without inference or robot motion')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--task', choices=['crop_detector', 'highcam_platform', 'highcam_rail'])
    parser.add_argument('--onnx-ref', type=Path)
    args = parser.parse_args()
    if args.check:
        imports = TARGETS[args.target][2]
        code = '; '.join('import ' + name for name in imports)
        if args.target == 'pi5-hailo8':
            code += '; from hailo_platform import InferVStreams, VDevice, HEF'
        result = subprocess.run([sys.executable, '-c', code], check=False)
        if result.returncode == 0:
            print('Runtime imports passed. Device access and model compatibility still require an inference run.')
        return result.returncode
    if args.model is None or args.data is None or args.task is None:
        parser.error('--model, --data and --task are required unless using --check.')
    try:
        command = command_for(args.target, args.model, args.data, args.task, args.onnx_ref)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({'command': command}, indent=2))
        return 0
    if not args.model.is_file() or not (args.data / 'images').is_dir() or not (args.data / 'labels').is_dir():
        parser.error('Supply an existing model and a dataset containing images/ and labels/.')
    if args.target == 'pi5' and not args.model.with_suffix('.bin').is_file():
        parser.error('NCNN requires the matching .bin beside the .param file.')
    return subprocess.call(command, cwd=ROOT)

if __name__ == '__main__':
    sys.exit(main())
