import sys
import glob
import threading
import argparse
import time
import termios
import tty

try:
    import serial
except ImportError:
    print("pyserial not found. Install it with:  pip install pyserial")
    sys.exit(1)

BAUD = 921600


SHORTCUTS = {
    'm': 'M',
    't': 'T',
    'i': 'I',
    's': 'S0',
    'p': 'PING',
}

SHORTCUT_HELP = [
    ("m", "M             swerve MID"),
    ("t", "T             swerve PERPENDICULAR"),
    ("i", "I             swerve INITIAL  /  board01 team name"),
    ("s", "S0            stop motors"),
    ("p", "PING          board status"),
    ("q", "              quit"),
    ("",  "(anything else + Enter = sent as-is)"),
]


def find_port():
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if candidates:
        return sorted(candidates)[0]
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="ESP32 serial command sender")
    parser.add_argument("port", nargs="?", help="Serial port (auto-detected if omitted)")
    parser.add_argument("--board", choices=["1", "2"], default=None,
                        help="Label only — does not change behaviour")
    return parser.parse_args()


def print_help(port):
    print("\n" + "─" * 48)
    print(f" Connected: {port}  @  {BAUD} baud")
    print("─" * 48)
    for key, desc in SHORTCUT_HELP:
        if key:
            print(f"  [{key}]  {desc}")
        else:
            print(f"       {desc}")
    print("─" * 48 + "\n")


def reader_thread(ser, stop_event):

    while not stop_event.is_set():
        try:
            line = ser.readline()
            if line:
                print("\r" + line.decode("utf-8", errors="replace").rstrip() + "\n> ", end="", flush=True)
        except Exception:
            break


def send(ser, cmd):
    ser.write((cmd + "\n").encode())
    ser.flush()
    print(f"\r→ {cmd}\n> ", end="", flush=True)


def getch():

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main():
    args = parse_args()
    port = args.port or find_port()

    if port is None:
        print("No serial port found. Plug in the ESP32 or pass the port as argument.")
        sys.exit(1)

    try:
        ser = serial.Serial(port, BAUD, timeout=0.1)
    except serial.SerialException as e:
        print(f"Could not open {port}: {e}")
        sys.exit(1)

    time.sleep(0.2)
    ser.reset_input_buffer()

    print_help(port)

    stop_event = threading.Event()
    t = threading.Thread(target=reader_thread, args=(ser, stop_event), daemon=True)
    t.start()

    print("> ", end="", flush=True)

    try:
        while True:
            ch = getch()


            if ch in ('q', 'Q', '\x03'):
                print("\nBye.")
                break


            if ch in SHORTCUTS:
                send(ser, SHORTCUTS[ch])
                continue


            if ch in ('\r', '\n'):
                print("\n> ", end="", flush=True)
                continue


            buf = ch
            sys.stdout.write(ch)
            sys.stdout.flush()
            while True:
                c = getch()
                if c in ('\r', '\n'):
                    break
                if c in ('\x7f', '\x08'):
                    if buf:
                        buf = buf[:-1]
                        sys.stdout.write('\b \b')
                        sys.stdout.flush()
                else:
                    buf += c
                    sys.stdout.write(c)
                    sys.stdout.flush()
            if buf.strip():
                send(ser, buf.strip().upper())
            else:
                print("\n> ", end="", flush=True)

    finally:
        stop_event.set()
        ser.close()


if __name__ == "__main__":
    main()
