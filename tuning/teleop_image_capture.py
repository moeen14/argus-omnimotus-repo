from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import serial
except ImportError:
    serial = None


WINDOW_NAME = "ASABE Teleop + Image Capture"
CAPTURE_DIR = Path(__file__).resolve().parent / "Jun26capturesLane"
DEFAULT_CAMERAS = [0, 2]
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30

DEFAULT_PORT = "/dev/esp32_02"
BAUD = 921600

DRIVE_RPM = 80
HOLD_TIMEOUT_SEC = 0.2

SERIAL_SETTLE_SEC = 2.0
STARTUP_SERVO_POSITIONS = [
    (8, 266-90),
    (9, 116.5),
]
DRIVE_MODE_COMMAND = "T"

ARM_VERTICAL_SERVO_ID = 6
SERVO_INPUT_BAR_HEIGHT = 50
SERVO_INPUT_TOGGLE_KEY = ord('s')
SERVO_INPUT_CHARS = set("0123456789.-")


BACKSPACE_KEY_CODES = {8, 127, 65288}
ENTER_KEY_CODES = {13, 10}
ESC_KEY_CODE = 27


POSITION_QUERY_COMMAND = "P"
POSITION_FEEDBACK_TIMEOUT_SEC = 0.5


ARROW_KEY_CODES = {
    "up": {82, 65362, 2490368, 63232},
    "down": {84, 65364, 2621440, 63233},
    "left": {81, 65361, 2424832, 63234},
    "right": {83, 65363, 2555904, 63235},
}


QT_EXTENDED_KEY_BIT = 0x100000


DRIVE_COMMANDS = {
    "up": (f"R{DRIVE_RPM}", f"L{DRIVE_RPM}"),
    "down": (f"R-{DRIVE_RPM}", f"L-{DRIVE_RPM}"),
    "left": (f"R{DRIVE_RPM}", f"L-{DRIVE_RPM}"),
    "right": (f"R-{DRIVE_RPM}", f"L{DRIVE_RPM}"),
}
STOP_COMMAND = "X"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-camera teleop with image capture.")
    parser.add_argument("--cams", nargs=2, type=str, default=DEFAULT_CAMERAS,
                         metavar=("CAM_A", "CAM_B"), help="Two camera indices/devices, e.g. 0 2.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--port", default=DEFAULT_PORT, help="Motor controller serial port.")
    parser.add_argument("--baud", type=int, default=BAUD)
    parser.add_argument("--no-serial", action="store_true", help="Skip robot connection, camera/capture only.")
    return parser.parse_args()


def camera_source(value):
    if isinstance(value, int):
        return value
    return int(value) if value.isdigit() else value


class Robot:
    def __init__(self, port: str, baud: int, enabled: bool) -> None:
        self.ser = None
        self.enabled = enabled
        self.last_command = None
        if not enabled:
            print("Serial disabled (--no-serial): commands will only be printed.")
            return
        if serial is None:
            print("pyserial not installed; running without robot control. pip install pyserial")
            self.enabled = False
            return
        try:
            self.ser = serial.Serial(port, baud, timeout=0.05)
        except Exception as exc:
            print(f"Could not open {port}: {exc}. Running without robot control.")
            self.enabled = False
            return

        time.sleep(SERIAL_SETTLE_SEC)
        self.ser.reset_input_buffer()
        for servo_id, angle in STARTUP_SERVO_POSITIONS:
            self.send(f"{servo_id} {angle}")
        self.confirm_servo_positions(STARTUP_SERVO_POSITIONS)
        self.send(DRIVE_MODE_COMMAND)

    def send(self, command: str) -> None:
        if command == self.last_command:
            return
        self.last_command = command
        if self.enabled and self.ser is not None:
            self.ser.write((command + "\n").encode("ascii"))
            self.ser.flush()
        print(f"  >>> {command}")

    def confirm_servo_positions(self, targets: list[tuple[int, float]]) -> None:


        if not self.enabled or self.ser is None:
            return
        self.ser.reset_input_buffer()
        self.ser.write((POSITION_QUERY_COMMAND + "\n").encode("ascii"))
        self.ser.flush()
        print(f"  >>> {POSITION_QUERY_COMMAND}")

        deadline = time.time() + POSITION_FEEDBACK_TIMEOUT_SEC
        line = ""
        while time.time() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            decoded = raw.decode(errors="ignore").strip()
            if decoded.startswith("POS,"):
                line = decoded
                break
        if not line:
            print("  !! no position feedback (POS,...) within "
                  f"{POSITION_FEEDBACK_TIMEOUT_SEC:.1f}s — could not confirm servo positions.")
            return

        values = line.split(",")[1:]
        for servo_id, target_angle in targets:
            idx = servo_id - 1
            if idx >= len(values):
                continue
            reported = values[idx]
            try:
                actual = float(reported)
                print(f"  servo {servo_id} feedback: {actual:.1f} deg "
                      f"(target {target_angle}, delta {actual - target_angle:+.1f} deg)")
            except ValueError:
                print(f"  servo {servo_id} feedback: {reported!r} (target {target_angle}, unreadable)")

    def close(self) -> None:
        self.send(STOP_COMMAND)
        if self.ser is not None:
            self.ser.close()


def open_capture(source, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(camera_source(source))
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def blank_frame(width: int, height: int):
    frame = cv2.UMat(height, width, cv2.CV_8UC3).get()
    frame[:] = (30, 30, 30)
    cv2.putText(frame, "NO FRAME", (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return frame


def resize_for_display(frame, height: int):
    if frame.shape[0] == height:
        return frame
    scale = height / frame.shape[0]
    width = max(1, int(frame.shape[1] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_label(frame, label: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def save_capture(frames, sources, index: int) -> list[Path]:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    for cam_idx, frame in enumerate(frames):
        path = CAPTURE_DIR / f"capture_{index:04d}_cam{cam_idx}_{stamp}.jpg"
        cv2.imwrite(str(path), frame)
        saved.append(path)
    return saved


def direction_from_key(key: int) -> str | None:
    normalized = key & ~QT_EXTENDED_KEY_BIT if key & QT_EXTENDED_KEY_BIT else key
    for direction, codes in ARROW_KEY_CODES.items():
        if key in codes or normalized in codes:
            return direction
    return None


def draw_servo_input_bar(width: int, active: bool, buffer: str):


    bar = cv2.UMat(SERVO_INPUT_BAR_HEIGHT, width, cv2.CV_8UC3).get()
    if active:
        bar[:] = (40, 90, 90)
        text = f"Servo {ARM_VERTICAL_SERVO_ID} angle: {buffer}_"
        hint = "Enter=send   Esc=cancel   Backspace=delete"
        text_color = (0, 255, 255)
    else:
        bar[:] = (35, 35, 35)
        text = f"Servo {ARM_VERTICAL_SERVO_ID}: press 's' to set angle"
        hint = ""
        text_color = (200, 200, 200)
    cv2.putText(bar, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    if hint:
        cv2.putText(bar, hint, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return bar


def main() -> int:
    args = parse_args()

    if cv2 is None:
        print("OpenCV is required. Install it with: python3 -m pip install opencv-python")
        return 2

    sources = list(args.cams)
    caps = []
    for source in sources:
        cap = open_capture(source, args.width, args.height, args.fps)
        if cap is None:
            print(f"Warning: could not open camera {source}")
        caps.append(cap)

    if all(cap is None for cap in caps):
        print("No cameras could be opened.")
        return 1

    robot = Robot(args.port, args.baud, enabled=not args.no_serial)

    print()
    print(f"Cameras: {sources}")
    print("Controls: arrows drive (R70/L70 @ constant speed), c capture, q/Esc quit")
    print(f"Press 's' (in the video window) to set servo {ARM_VERTICAL_SERVO_ID}'s angle.")

    image_count = 0
    current_direction = None
    last_arrow_time = 0.0
    servo_input_active = False
    servo_input_buffer = ""

    try:
        while True:
            frames = []
            for idx, cap in enumerate(caps):
                if cap is None:
                    frames.append(blank_frame(args.width, args.height))
                    continue
                ok, frame = cap.read()
                frames.append(frame if ok else blank_frame(args.width, args.height))

            display_h = min(frame.shape[0] for frame in frames)
            shown = [resize_for_display(frame, display_h) for frame in frames]
            for idx, frame in enumerate(shown):
                draw_label(frame, f"cam{idx}  {sources[idx]}")
            combined = cv2.hconcat(shown) if len(shown) > 1 else shown[0]

            footer = (
                f"images={image_count} | cmd={robot.last_command or 'stopped'} | "
                "arrows drive | c capture | q quit"
            )
            cv2.putText(combined, footer, (10, combined.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            servo_bar = draw_servo_input_bar(combined.shape[1], servo_input_active, servo_input_buffer)
            combined = cv2.vconcat([combined, servo_bar])

            cv2.imshow(WINDOW_NAME, combined)
            key = cv2.waitKeyEx(20)
            key_masked = key & 0xFF if key != -1 else -1

            if servo_input_active:


                if key in ENTER_KEY_CODES or key_masked in ENTER_KEY_CODES:
                    if servo_input_buffer:
                        try:
                            angle = float(servo_input_buffer)
                            robot.send(f"{ARM_VERTICAL_SERVO_ID} {angle}")
                        except ValueError:
                            print(f"  !! '{servo_input_buffer}' isn't a valid angle — not sent.")
                    servo_input_active = False
                    servo_input_buffer = ""
                elif key_masked == ESC_KEY_CODE:
                    servo_input_active = False
                    servo_input_buffer = ""
                elif key in BACKSPACE_KEY_CODES or key_masked in BACKSPACE_KEY_CODES:
                    servo_input_buffer = servo_input_buffer[:-1]
                elif 0 <= key_masked < 128 and chr(key_masked) in SERVO_INPUT_CHARS:
                    servo_input_buffer += chr(key_masked)
                continue

            if key_masked == SERVO_INPUT_TOGGLE_KEY:
                servo_input_active = True
                servo_input_buffer = ""
                current_direction = None
                robot.send(STOP_COMMAND)
                continue

            now = time.monotonic()
            direction = direction_from_key(key)
            if direction is not None:
                current_direction = direction
                last_arrow_time = now
                right_cmd, left_cmd = DRIVE_COMMANDS[direction]
                robot.send(right_cmd)
                robot.send(left_cmd)
            elif current_direction is not None and (now - last_arrow_time) > HOLD_TIMEOUT_SEC:
                current_direction = None
                robot.send(STOP_COMMAND)
            elif key != -1 and key_masked not in (ord('c'), ord('q'), ESC_KEY_CODE, SERVO_INPUT_TOGGLE_KEY):


                print(f"  [debug] unrecognized key: raw={key} masked={key_masked} "
                      f"(if this was an arrow key, report this number)")

            if key_masked in (ESC_KEY_CODE, ord('q')):
                break
            if key_masked == ord('c'):
                image_count += 1
                saved = save_capture(frames, sources, image_count)
                print(f"  captured #{image_count}: {[p.name for p in saved]}")
    finally:
        robot.close()
        for cap in caps:
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()

    print(f"Total images captured: {image_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
