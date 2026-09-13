import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
INDEX = 'https://espressif.github.io/arduino-esp32/package_esp32_index.json'
LIBRARIES = ['ESP32Servo@3.0.9', 'VL53L1X@1.3.1', 'Adafruit LSM6DS@4.7.4', 'Adafruit LIS3MDL@1.2.4', 'Adafruit Unified Sensor@1.1.15', 'Adafruit SSD1306@2.5.15', 'Adafruit GFX Library@1.12.6', 'Adafruit BusIO@1.17.4', 'SCServo@1.0.2']

def main():
    parser = argparse.ArgumentParser(description='Build or explicitly upload firmware for ESP32 DevKit.')
    parser.add_argument('action', choices=['install', 'compile', 'upload'])
    parser.add_argument('--board', choices=['board01', 'board02'])
    parser.add_argument('--port')
    parser.add_argument('--cli', default='arduino-cli')
    parser.add_argument('--fqbn', default='esp32:esp32:esp32')
    args = parser.parse_args()
    def run(*command):
        subprocess.run([args.cli, *command], check=True)
    if args.action == 'install':
        run('core', 'update-index', '--additional-urls', INDEX)
        run('core', 'install', 'esp32:esp32@3.3.0', '--additional-urls', INDEX)
        for library in LIBRARIES:
            run('lib', 'install', library)
        return
    if args.action == 'upload' and (not args.board or not args.port):
        parser.error('Uploading requires an explicit --board and --port.')
    for board in ([args.board] if args.board else ['board01', 'board02']):
        sketch = str(ROOT / 'firmware' / board)
        run('compile', '--fqbn', args.fqbn, sketch)
        if args.action == 'upload':
            run('upload', '--fqbn', args.fqbn, '--port', args.port, sketch)

if __name__ == '__main__':
    main()
