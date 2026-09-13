import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def verify(path, record):
    if path.stat().st_size != record['bytes']:
        raise ValueError(f'Size mismatch: {path.name}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != record['sha256']:
        raise ValueError(f'SHA-256 mismatch: {path.name}')

def main():
    parser = argparse.ArgumentParser(description='Download, verify, or build Argus model assets.')
    parser.add_argument('action', choices=['download', 'verify', 'build'])
    parser.add_argument('--repo-id')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--directory', type=Path, default=ROOT / 'models')
    parser.add_argument('--trtexec', default='/usr/src/tensorrt/bin/trtexec')
    parser.add_argument('--model', action='append', help='ONNX filename; repeat to select multiple models')
    args = parser.parse_args()
    records = json.loads((ROOT / 'models/manifest.json').read_text())['files']
    if args.model:
        unknown = set(args.model) - {item['filename'] for item in records}
        if unknown:
            parser.error(f'Unknown models: {sorted(unknown)}')
        records = [item for item in records if item['filename'] in args.model]
    if args.action == 'download' and not args.repo_id:
        parser.error('--repo-id OWNER/MODELS is required; the public repository is not configured yet.')
    args.directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = args.directory / record['filename']
        if args.action == 'download':
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=args.repo_id, filename=record['filename'], revision=args.revision, local_dir=args.directory)
        verify(path, record)
        print(f'Verified {path.name}')
        if args.action == 'build' and path.name != 'mobile_sam_mask_decoder.onnx':
            subprocess.run([args.trtexec, f'--onnx={path.resolve()}', f'--saveEngine={path.with_suffix(".engine").resolve()}', '--fp16'], check=True)

if __name__ == '__main__':
    main()
