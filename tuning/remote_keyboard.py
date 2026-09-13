from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import termios
import time
import tty
import select


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50552
HOLD_TIMEOUT_SEC = 0.16
KEY_POLL_SEC = 0.01
ESCAPE_TIMEOUT_SEC = 0.12

ARROW_MAP = {
    "A": "up",
    "B": "down",
    "C": "right",
    "D": "left",
}

TAP_KEYS = {
    "i", "m", "t",
    "q", "w", "e",
    "g", "b", "r", "c", "u", "p", "o", "y",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "x", " ",
}

HOLD_KEYS = {"up", "down", "left", "right"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send SSH keyboard events to system_analysis_dashboard.py."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Dashboard UDP host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Dashboard UDP port.")
    return parser.parse_args()


def send_event(sock: socket.socket, addr: tuple[str, int], event: str, key: str) -> None:
    payload = json.dumps({"event": event, "key": key}, separators=(",", ":")).encode("utf-8")
    sock.sendto(payload, addr)


def read_char(fd: int, timeout: float) -> str | None:
    if not select.select([fd], [], [], timeout)[0]:
        return None
    data = os.read(fd, 1)
    if not data:
        return None
    return data.decode("utf-8", errors="ignore")


def read_key(fd: int) -> str | None:
    char = read_char(fd, 0.0)
    if char is None:
        return None
    if char == "\x03":
        raise KeyboardInterrupt

    if char == "\x1b":


        introducer = read_char(fd, ESCAPE_TIMEOUT_SEC)
        if introducer not in ("[", "O"):
            return None

        sequence = ""
        deadline = time.monotonic() + ESCAPE_TIMEOUT_SEC
        while time.monotonic() < deadline:
            next_char = read_char(fd, 0.005)
            if next_char is None:
                continue
            sequence += next_char
            if sequence[-1].isalpha() or sequence[-1] == "~":
                break

        if not sequence:
            return None
        return ARROW_MAP.get(sequence[-1])

    if char == "\r":
        return "\n"
    if char:
        return char.lower()
    return None


def print_help(host: str, port: int) -> None:
    print(f"Sending keys to dashboard UDP {host}:{port}")
    print("Controls:")
    print("  I/M/T        swerve modes")
    print("  Arrows       drive / steer / rotate")
    print("  Space or X   stop motors")
    print("  Q/W/E        overhead camera left / forward / right")
    print("  1-9,0        servo presets from dashboard")
    print("  G/B/R/C/U/P/O/Y same as dashboard")
    print("  Ctrl-C       quit, sending motor stop")
    print()


def main() -> int:
    args = parse_args()
    addr = (args.host, args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    active_until: dict[str, float] = {}
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    print_help(args.host, args.port)
    try:
        tty.setraw(fd)
        while True:
            now = time.monotonic()
            readable, _, _ = select.select([fd], [], [], KEY_POLL_SEC)
            if readable:
                key = read_key(fd)
                if key in HOLD_KEYS:
                    if key not in active_until:
                        send_event(sock, addr, "down", key)
                    active_until[key] = now + HOLD_TIMEOUT_SEC
                    print(f"\rhold {key:<5}", end="", flush=True)
                elif key in TAP_KEYS:
                    send_event(sock, addr, "tap", key)
                    label = "space" if key == " " else key
                    print(f"\rtap  {label:<5}", end="", flush=True)

            now = time.monotonic()
            expired = [key for key, deadline in active_until.items() if deadline <= now]
            for key in expired:
                send_event(sock, addr, "up", key)
                active_until.pop(key, None)
                print(f"\rup   {key:<5}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for key in list(active_until):
            send_event(sock, addr, "up", key)
        send_event(sock, addr, "tap", "x")
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        sock.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
