from __future__ import annotations

import argparse
import sys
import threading
import time

try:
    import serial
except ImportError:
    print("pyserial not found. Install it or activate nanosam-env.")
    sys.exit(1)

BAUD = 921600
DEFAULT_PORT = "/dev/esp32_02"


VALID_MOTIONS = {
    "i": {"f", "b"},
    "m": {"cw", "ccw"},
    "t": {"l", "r"},
}


MOTION_CMD = {
    "f":   "F{}",
    "b":   "B{}",
    "l":   "L{}",
    "r":   "R{}",
    "cw":  "CW{}",
    "ccw": "CCW{}",
}

MODE_CMD = {"i": "I", "m": "M", "t": "T"}


def serial_reader(ser: serial.Serial, stop_event: threading.Event) -> None:

    while not stop_event.is_set():
        try:
            line = ser.readline()
            if line:
                print(f"  [esp] {line.decode(errors='replace').strip()}")
        except Exception:
            break


def send(ser: serial.Serial, cmd: str) -> None:
    ser.write((cmd + "\n").encode())
    ser.flush()
    print(f"  >>> {cmd}")


def prompt(speed: int, duration_ms: int, mode: str) -> str:
    return f"[speed={speed} dur={duration_ms}ms mode={mode}] > "


def run_pulse(ser: serial.Serial, motion: str, speed: int, duration_ms: int, mode: str) -> None:
    allowed = VALID_MOTIONS.get(mode, set())
    if motion not in allowed:
        print(f"  ! '{motion}' not valid in mode '{mode}'. Allowed: {sorted(allowed)}")
        return

    cmd = MOTION_CMD[motion].format(speed)
    send(ser, cmd)
    time.sleep(duration_ms / 1000.0)
    send(ser, "X")


def main() -> None:
    parser = argparse.ArgumentParser(description="Motor threshold test")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Board 02 serial port")
    args = parser.parse_args()

    print(f"Connecting to {args.port} at {BAUD} baud...")
    try:
        ser = serial.Serial(args.port, BAUD, timeout=0.1)
    except serial.SerialException as e:
        print(f"Could not open {args.port}: {e}")
        sys.exit(1)

    time.sleep(0.5)
    ser.reset_input_buffer()

    stop_event = threading.Event()
    reader = threading.Thread(target=serial_reader, args=(ser, stop_event), daemon=True)
    reader.start()

    speed = 100
    duration_ms = 300
    mode = "i"

    print()
    print("Motor threshold test ready. Type 'help' to see commands.")
    print()

    try:
        while True:
            try:
                raw = input(prompt(speed, duration_ms, mode)).strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if not raw:
                continue

            if raw in ("q", "quit"):
                break

            if raw == "help":
                print(__doc__)
                continue

            if raw == "x":
                send(ser, "X")
                continue

            if raw in ("i", "m", "t"):
                mode = raw
                send(ser, MODE_CMD[mode])
                continue

            if raw.startswith("s") and raw[1:].isdigit():
                val = int(raw[1:])
                if 0 <= val <= 255:
                    speed = val
                    print(f"  speed set to {speed}")
                else:
                    print("  ! speed must be 0–255")
                continue

            if raw.startswith("g") and raw[1:].isdigit():
                val = int(raw[1:])
                if val > 0:
                    duration_ms = val
                    print(f"  duration set to {duration_ms} ms")
                else:
                    print("  ! duration must be > 0")
                continue

            if raw in MOTION_CMD:
                run_pulse(ser, raw, speed, duration_ms, mode)
                continue

            print(f"  ? unknown command '{raw}' — type 'help' for usage")

    finally:
        stop_event.set()
        send(ser, "X")
        time.sleep(0.1)
        ser.close()
        print("Closed.")


if __name__ == "__main__":
    main()
