import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {'dashboard': 'applications/operator_dashboard.py', 'autonomy': 'applications/field_tasks.py', 'identify': 'applications/crop_identification.py', 'keyboard': 'tuning/remote_keyboard.py'}
ALIASES.update({'navigate': 'applications/navigate_testbed.py', 'field': 'applications/field_tasks.py', 'lane': 'applications/single_lane_tasks.py'})

def prepare(script, config_path):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    target = (ROOT / ALIASES.get(script, script)).resolve()
    if not target.is_relative_to(ROOT) or target.suffix != '.py' or not target.is_file():
        raise ValueError('Choose an existing Python script inside this repository.')
    model_dir = Path(config['model_directory']).expanduser()
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    model_dir = model_dir.resolve()
    overrides = config.get('environment', {})
    if any(not key.startswith('ASABE_') or not isinstance(value, str) for key, value in overrides.items()):
        raise ValueError('Environment entries must be ASABE_ names with string values.')
    env = dict(os.environ)
    env.update(overrides)
    for key, filename in {'ASABE_LANE_ENGINE': 'highcam_rail.engine', 'ASABE_CROP_ENGINE': 'crop_detector.engine', 'ASABE_PLATFORM_ENGINE': 'highcam_platform.engine', 'ASABE_BALL_VERIFY_ENGINE': 'highcam_platform.engine'}.items():
        env.setdefault(key, str(model_dir / filename))
    return target, model_dir, env

def main():
    parser = argparse.ArgumentParser(description='Launch Argus software with local device and model configuration.')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/robot.json')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verification', choices=['on', 'off'], help='Enable or disable ball-placement verification for the field application')
    parser.add_argument('script', help='dashboard, identify, navigate, field, lane, keyboard, or repository-relative Python path')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        target, model_dir, env = prepare(args.script, args.config)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if args.verification is not None:
        if target != (ROOT / 'applications/field_tasks.py').resolve():
            parser.error('--verification is supported only by the field application.')
        env['ASABE_ENABLE_BALL_VERIFY'] = '1' if args.verification == 'on' else '0'
    command = [sys.executable, str(target), *args.arguments]
    if args.dry_run:
        print(json.dumps({'command': command, 'working_directory': str(model_dir), 'environment': {k: v for k, v in env.items() if k.startswith('ASABE_')}}, indent=2))
        return 0
    model_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.call(command, cwd=model_dir, env=env)

if __name__ == '__main__':
    sys.exit(main())
