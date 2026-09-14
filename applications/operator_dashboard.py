import csv
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
import queue
import re
import socket
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont

try:
    import serial
except ImportError:
    serial = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


ESP1_PORT = os.environ.get("ASABE_ESP1_PORT", "/dev/esp32_01")
ESP2_PORT = os.environ.get("ASABE_ESP2_PORT", "/dev/esp32_02")
BOTTOM_CAMERA_DEVICE = os.environ.get(
    "ASABE_BOTTOM_CAMERA_DEVICE",
    "/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_9342D410-video-index0",
)
OVERHEAD_CAMERA_DEVICE = os.environ.get(
    "ASABE_OVERHEAD_CAMERA_DEVICE",
    "/dev/v4l/by-id/usb-046d_0825_3D19ADE0-video-index0",
)
CAMERA_DEVICE = os.environ.get("ASABE_CAMERA_DEVICE", BOTTOM_CAMERA_DEVICE)
CAMERA_DEVICES_ENV = os.environ.get("ASABE_CAMERA_DEVICES", "")
CAMERA_LABELS = ("Bottom", "Overhead")
REMOTE_KEY_HOST = os.environ.get("ASABE_REMOTE_KEY_HOST", "127.0.0.1")
REMOTE_KEY_PORT = int(os.environ.get("ASABE_REMOTE_KEY_PORT", "50552"))
ENABLE_LOCAL_KEYS = os.environ.get("ASABE_ENABLE_LOCAL_KEYS", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
BAUD_RATE = 921600
SERIAL_TIMEOUT = 0.08
UI_INTERVAL_MS = 25
COMMAND_INTERVAL_MS = 40
SERVO_POLL_MS = 1000
MOTOR_PWM = 175
DELTA_STEP = 2
DELTA_MAX = 30
DELTA_DECAY = 2
IMU_CALIBRATION_MIN_SAMPLES = 300
IMU_STILL_GYRO_MAX_DPS = 2.0
RPM_REASONABLE_MAX = 160
RPM_MAX_STEP = 35
RPM_SMOOTH_ALPHA = 0.28
TOF_STALE_SECONDS = 1.5
WHEEL_LABELS = ("BR", "FR", "BL", "FL")
WHEEL_SERVO_IDS = {"BR": 1, "FR": 2, "BL": 4, "FL": 3}
T_LEFT_FRONT_EXTRA_PWM_OFFSET = 6
T_RIGHT_BACK_EXTRA_PWM_OFFSET = 8
I_FORWARD_LEFT_EXTRA_PWM_OFFSET = -5
I_BACKWARD_RIGHT_EXTRA_PWM_OFFSET = -10
SWERVE_MODE_TARGETS = {
    "initial": {"BR": 37, "FR": 6, "BL": 145, "FL": 221},
    "mid": {"BR": 306, "FR": 96, "BL": 235, "FL": 131},
    "perpendicular": {"BR": 220, "FR": 180, "BL": 325, "FL": 42},
}
BASE_BALANCED_PWM_TARGETS = {
    "forward": {"BR": 175, "FR": 178, "BL": 174, "FL": 172},
    "backward": {"BR": 175, "FR": 176, "BL": 175, "FL": 174},
    "cw": {"BR": 179, "FR": 179, "BL": 172, "FL": 171},
    "ccw": {"BR": 176, "FR": 181, "BL": 172, "FL": 172},
    "left": {"BR": 174, "FR": 178, "BL": 171, "FL": 177},
    "right": {"BR": 176, "FR": 175, "BL": 176, "FL": 173},
}


def build_swerve_mode_command(mode_name):
    targets = SWERVE_MODE_TARGETS[mode_name]
    return "MODE,{},{:.2f},{:.2f},{:.2f},{:.2f}".format(
        mode_name,
        *[targets[label] for label in WHEEL_LABELS],
    )


def build_motor_command(prefix, movement):
    pwms = dict(BASE_BALANCED_PWM_TARGETS[movement])
    if movement == "forward":
        pwms["BL"] += I_FORWARD_LEFT_EXTRA_PWM_OFFSET
        pwms["FL"] += I_FORWARD_LEFT_EXTRA_PWM_OFFSET
    elif movement == "backward":
        pwms["BR"] += I_BACKWARD_RIGHT_EXTRA_PWM_OFFSET
        pwms["FR"] += I_BACKWARD_RIGHT_EXTRA_PWM_OFFSET
    elif movement == "left":
        pwms["FR"] += T_LEFT_FRONT_EXTRA_PWM_OFFSET
        pwms["FL"] += T_LEFT_FRONT_EXTRA_PWM_OFFSET
    elif movement == "right":
        pwms["BR"] += T_RIGHT_BACK_EXTRA_PWM_OFFSET
        pwms["BL"] += T_RIGHT_BACK_EXTRA_PWM_OFFSET
    values = [pwms[label] for label in WHEEL_LABELS]
    return f"{prefix}{values[0]},{values[1]},{values[2]},{values[3]}"

STARTUP_COMMANDS = [
    (build_swerve_mode_command("initial"), "swerve initial"),
    ("5 322", "arm horizontal initial/middle"),
    ("6 50", "arm vertical high/initial"),
    ("7 51", "camera front/initial"),
    ("8 266", "overhead camera forward"),
    ("52", "SG90 dispenser LDR spot"),
]

SERVO_KEY_COMMANDS = {
    "1": ("7 51", 7, 51, "camera front"),
    "2": ("7 141", 7, 141, "camera left"),
    "3": ("7 321", 7, 321, "camera right"),
    "4": ("7 231", 7, 231, "camera back"),
    "5": ("5 322", 5, 322, "arm horizontal middle"),
    "6": ("5 295", 5, 295, "arm horizontal left"),
    "7": ("5 345", 5, 345, "arm horizontal right"),
    "8": ("6 50", 6, 50, "arm vertical high"),
    "9": ("6 300", 6, 300, "arm vertical low"),
    "0": ("8 266", 8, 266, "overhead camera forward"),
    "q": ("8 179", 8, 179, "overhead camera left"),
    "w": ("8 266", 8, 266, "overhead camera forward"),
    "e": ("8 358", 8, 358, "overhead camera right"),
}

MAROON = "#7A1730"
MAROON_DARK = "#4b0d1d"
GREEN = "#1fa34a"
YELLOW = "#d39b00"
RED = "#d84a4a"
BLUE = "#3a6ea5"
BG = "#f0f2f5"
CARD = "#ffffff"
CARD_BORDER = "#dde3eb"
TEXT = "#1a1a2e"
MUTED = "#8898aa"
BLACK_PANEL = "#0d1117"
TOF_LABELS = ["FRS", "BRS", "BLS", "FLF", "FLS", "CAM", "FRF"]
SERVO_NAMES = {
    1: "BR swerve",
    2: "FR swerve",
    3: "FL swerve",
    4: "BL swerve",
    5: "Arm horiz",
    6: "Arm vert",
    7: "Cam pan",
    8: "Overhead cam",
}


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def natural_video_key(path):
    name = Path(path).name
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return prefix, int(digits or 0)


def video_devices():
    return [str(path) for path in sorted(Path("/dev").glob("video*"), key=natural_video_key)]


def camera_source(value):
    value = str(value).strip()
    return int(value) if value.isdigit() else value


def configured_camera_devices():
    if CAMERA_DEVICES_ENV.strip():
        return [item.strip() for item in CAMERA_DEVICES_ENV.split(",") if item.strip()]
    stable_devices = [BOTTOM_CAMERA_DEVICE, OVERHEAD_CAMERA_DEVICE]
    if all(Path(device).exists() for device in stable_devices):
        return stable_devices
    if CAMERA_DEVICE.strip():
        return [CAMERA_DEVICE.strip()]
    return []


def enable_high_dpi():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


class StudyLogger:
    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(__file__).resolve().parent / "logs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.files = {}
        self.writers = {}
        self.events_fp = (self.run_dir / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        self.video_path = self.run_dir / "camera_feed.mp4"
        self._open_csv("serial_raw", ["ts_iso", "t_unix", "source", "line"])
        self._open_csv("commands", ["ts_iso", "t_unix", "source", "command", "sent", "mode", "pwm", "green", "blue", "red", "total"])
        self._open_csv("notes", ["ts_iso", "t_unix", "note"])
        self._open_csv("rpm", ["ts_iso", "t_unix", "avg", "BR", "FR", "BL", "FL", "raw"])
        self._open_csv("tof", ["ts_iso", "t_unix", *TOF_LABELS, "raw"])
        self._open_csv("imu", [
            "ts_iso", "t_unix",
            "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz",
            "roll", "pitch", "yaw", "acc_mag",
            "gx_corrected", "gy_corrected", "gz_corrected",
            "yaw_mag_zeroed", "yaw_gyro_integrated",
            "calibration_samples", "raw",
        ])
        self._open_csv("servos", ["ts_iso", "t_unix", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "raw"])
        self._open_csv("colors", ["ts_iso", "t_unix", "green", "blue", "red", "total", "raw"])
        self.log_event("run_start", {
            "run_id": self.run_id,
            "esp1_port": ESP1_PORT,
            "esp2_port": ESP2_PORT,
            "bottom_camera_device": BOTTOM_CAMERA_DEVICE,
            "overhead_camera_device": OVERHEAD_CAMERA_DEVICE,
            "camera_devices": configured_camera_devices(),
            "baud": BAUD_RATE,
            "archive_reference": {
                "board01": "/dev/esp32_01",
                "board02": "/dev/esp32_02",
                "bottom_camera": BOTTOM_CAMERA_DEVICE,
                "overhead_camera": OVERHEAD_CAMERA_DEVICE,
            },
        })

    def _stamp(self, ts=None):
        t = time.time() if ts is None else ts
        return datetime.fromtimestamp(t, timezone.utc).isoformat(), t

    def _open_csv(self, name, fieldnames):
        fp = (self.run_dir / f"{name}.csv").open("a", newline="", encoding="utf-8", buffering=1)
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        self.files[name] = fp
        self.writers[name] = writer

    def log_event(self, kind, payload=None, ts=None):
        ts_iso, t_unix = self._stamp(ts)
        record = {"ts_iso": ts_iso, "t_unix": t_unix, "type": kind, "payload": payload or {}}
        self.events_fp.write(json.dumps(record, sort_keys=True) + "\n")

    def write_csv(self, name, row, ts=None):
        ts_iso, t_unix = self._stamp(ts)
        full = {"ts_iso": ts_iso, "t_unix": t_unix}
        full.update(row)
        self.writers[name].writerow(full)

    def log_serial(self, source, line, ts=None):
        self.write_csv("serial_raw", {"source": source, "line": line}, ts)
        self.log_event("serial_line", {"source": source, "line": line}, ts)

    def log_command(self, command, source="ui", sent=False, ts=None, meta=None):
        meta = meta or {}
        row = {
            "source": source,
            "command": command,
            "sent": int(bool(sent)),
            "mode": meta.get("mode", ""),
            "pwm": meta.get("pwm", ""),
            "green": meta.get("green", ""),
            "blue": meta.get("blue", ""),
            "red": meta.get("red", ""),
            "total": meta.get("total", ""),
        }
        self.write_csv("commands", row, ts)
        self.log_event("command", {"source": source, "command": command, "sent": bool(sent), **meta}, ts)

    def log_note(self, note, ts=None):
        self.write_csv("notes", {"note": note}, ts)
        self.log_event("operator_note", {"note": note}, ts)

    def close(self):
        self.log_event("run_stop")
        self.events_fp.close()
        for fp in self.files.values():
            fp.close()


class SerialWorker:
    def __init__(self, name, port, inbox, status_cb, logger):
        self.name = name
        self.port = port
        self.inbox = inbox
        self.status_cb = status_cb
        self.logger = logger
        self.ser = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def start(self):
        if serial is None:
            self.status_cb(self.name, "pyserial missing")
            return
        threading.Thread(target=self.run, daemon=True).start()

    def stop(self):
        self.stop_event.set()
        with self.lock:
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.ser = None

    def send(self, command):
        if not command:
            return False
        with self.lock:
            if not self.ser or not self.ser.is_open:
                self.status_cb(self.name, "offline")
                self.logger.log_command(command, source=self.name, sent=False)
                return False
            try:
                self.ser.write((command.strip() + "\n").encode("ascii"))
                self.logger.log_command(command, source=self.name, sent=True)
                return True
            except Exception as exc:
                self.status_cb(self.name, f"write error: {exc}")
                self.logger.log_command(command, source=self.name, sent=False)
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
                return False

    def run(self):
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    self.ser = serial.Serial(self.port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
                time.sleep(2)
                with self.lock:
                    if self.ser:
                        self.ser.reset_input_buffer()
                self.status_cb(self.name, "live")
                self.send("PING")
                if self.name == "board01":
                    self.status_cb(self.name, "resetting tof")
                    self.send("F")
                    time.sleep(3)
                    self.status_cb(self.name, "live")

                while not self.stop_event.is_set():
                    with self.lock:
                        ser = self.ser
                    if not ser or not ser.is_open:
                        break
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode(errors="ignore").strip()
                    if line:
                        ts = time.time()
                        self.logger.log_serial(self.name, line, ts)
                        self.inbox.put(("serial", self.name, line, ts))
            except Exception as exc:
                self.status_cb(self.name, "reconnecting")
                self.inbox.put(("serial_error", self.name, str(exc), time.time()))
                self.logger.log_event("serial_error", {"source": self.name, "port": self.port, "error": str(exc)})
                time.sleep(1)


class CameraWorker:
    def __init__(self, inbox, logger):
        self.inbox = inbox
        self.logger = logger
        self.stop_event = threading.Event()
        self.thread = None
        self.writer = None
        self.writer_lock = threading.RLock()
        self.writer_size = None
        self.writer_failed = False

    def start(self):
        if cv2 is None or Image is None or ImageTk is None:
            self.inbox.put(("camera_status", "camera libs missing", time.time()))
            self.logger.log_event("camera_status", {"status": "camera libs missing"})
            return
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.close_writer()

    def close_writer(self):
        with self.writer_lock:
            writer = self.writer
            self.writer = None
            self.writer_size = None
        if writer:
            writer.release()

    def open_capture(self, source):
        cap = cv2.VideoCapture(camera_source(source))
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, _frame = cap.read()
        if not ok:
            cap.release()
            return None
        return cap

    def open_cameras(self):
        sources = []
        caps = []
        tried = set()

        for source in configured_camera_devices() + video_devices():
            if source in tried:
                continue
            tried.add(source)
            cap = self.open_capture(source)
            if cap is None:
                continue
            sources.append(source)
            caps.append(cap)
            if len(caps) == 2:
                break

        return sources, caps

    def combined_for_recording(self, frames):
        if len(frames) != 2:
            return frames[0] if frames else None
        target_h = min(frame.shape[0] for frame in frames)
        resized = []
        for frame in frames:
            if frame.shape[0] != target_h:
                scale = target_h / frame.shape[0]
                target_w = max(1, int(frame.shape[1] * scale))
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            resized.append(frame)
        return cv2.hconcat(resized)

    def ensure_writer(self, frame):
        if self.writer is not None or self.writer_failed:
            return
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.logger.video_path), fourcc, 30.0, (w, h))
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            self.writer_failed = True
            self.inbox.put(("camera_status", "camera recorder failed", time.time()))
            self.logger.log_event("camera_recording_error", {
                "path": str(self.logger.video_path),
                "width": w,
                "height": h,
                "fps": 30,
                "codec": "mp4v",
            })
            return
        self.writer_size = (w, h)
        self.logger.log_event("camera_recording_start", {
            "path": str(self.logger.video_path),
            "width": w,
            "height": h,
            "fps": 30,
            "codec": "mp4v",
        })

    def run(self):
        while not self.stop_event.is_set():
            caps = []
            try:
                sources, caps = self.open_cameras()
                if len(caps) < 2:
                    for cap in caps:
                        cap.release()
                    self.inbox.put(("camera_status", f"{len(caps)}/2 cameras found", time.time()))
                    self.logger.log_event("camera_status", {
                        "status": "dual camera not ready",
                        "sources": sources,
                        "configured": configured_camera_devices(),
                    })
                    time.sleep(1.5)
                    continue

                status = f"dual cameras live: {sources[0]}, {sources[1]}"
                self.inbox.put(("camera_status", status, time.time()))
                self.logger.log_event("camera_status", {"status": "dual cameras live", "sources": sources})
                while not self.stop_event.is_set():
                    frames = []
                    missing = []
                    for index, cap in enumerate(caps):
                        ok, frame = cap.read()
                        if ok:
                            frames.append(frame)
                        else:
                            missing.append(sources[index])

                    if missing or len(frames) != 2:
                        self.inbox.put(("camera_status", "camera frame missing", time.time()))
                        self.logger.log_event("camera_status", {
                            "status": "camera frame missing",
                            "missing": missing,
                        })
                        break

                    combined = self.combined_for_recording(frames)
                    with self.writer_lock:
                        self.ensure_writer(combined)
                        h, w = combined.shape[:2]
                        if self.writer and self.writer_size != (w, h):
                            self.logger.log_event("camera_recording_frame_size_mismatch", {
                                "expected": self.writer_size,
                                "actual": (w, h),
                            })
                            self.close_writer()
                            self.ensure_writer(combined)
                        if self.writer and self.writer_size == (w, h):
                            self.writer.write(combined)
                    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
                    self.inbox.put(("camera_frame", {"frames": rgb_frames, "sources": sources}, time.time()))
                    time.sleep(0.033)
            except Exception as exc:
                self.inbox.put(("camera_status", f"camera error: {exc}", time.time()))
                self.logger.log_event("camera_error", {"error": str(exc), "configured": configured_camera_devices()})
                time.sleep(1.5)
            finally:
                for cap in caps:
                    cap.release()
        self.close_writer()


class RemoteKeyServer:
    def __init__(self, inbox, logger, host=REMOTE_KEY_HOST, port=REMOTE_KEY_PORT):
        self.inbox = inbox
        self.logger = logger
        self.host = host
        self.port = port
        self.sock = None
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.host, self.port))
            self.sock.settimeout(0.2)
            self.logger.log_event("remote_key_server_start", {
                "host": self.host,
                "port": self.port,
            })
        except OSError as exc:
            self.inbox.put(("remote_status", f"remote key bind failed: {exc}", time.time()))
            self.logger.log_event("remote_key_server_error", {"error": str(exc)})
            return

        self.inbox.put(("remote_status", f"remote keys {self.host}:{self.port}", time.time()))
        while not self.stop_event.is_set():
            try:
                data, addr = self.sock.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            event = str(payload.get("event", "")).lower()
            key = str(payload.get("key", "")).lower()
            if event not in {"down", "up", "tap"} or not key:
                continue

            ts = time.time()
            self.logger.log_event("remote_key", {
                "event": event,
                "key": key,
                "addr": addr[0],
            }, ts)
            self.inbox.put(("remote_key", event, key, ts))


class ImuFusion:
    def __init__(self):
        self.last_t = None
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw_gyro_integrated = 0.0
        self.acc = [0.0, 0.0, 9.81]
        self.mag = [1.0, 0.0, 0.0]
        self.calibration_samples = []
        self.bias = {"gx": 0.0, "gy": 0.0, "gz": 0.0}
        self.yaw_reference = None

    @staticmethod
    def wrap(angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def update(self, values, now):
        ax, ay, az, gx, gy, gz, mx, my, mz = values
        dt = 0.02 if self.last_t is None else clamp(now - self.last_t, 0.005, 0.15)
        self.last_t = now

        alpha = 0.16
        self.acc = [
            self.acc[0] * (1 - alpha) + ax * alpha,
            self.acc[1] * (1 - alpha) + ay * alpha,
            self.acc[2] * (1 - alpha) + az * alpha,
        ]
        self.mag = [
            self.mag[0] * (1 - alpha) + mx * alpha,
            self.mag[1] * (1 - alpha) + my * alpha,
            self.mag[2] * (1 - alpha) + mz * alpha,
        ]

        mag_yaw = self.wrap(math.degrees(math.atan2(self.mag[1], self.mag[0])))
        if (
            len(self.calibration_samples) < IMU_CALIBRATION_MIN_SAMPLES
            and abs(gx) < IMU_STILL_GYRO_MAX_DPS
            and abs(gy) < IMU_STILL_GYRO_MAX_DPS
            and abs(gz) < IMU_STILL_GYRO_MAX_DPS
        ):
            self.calibration_samples.append({"gx": gx, "gy": gy, "gz": gz, "yaw": mag_yaw})
            for axis in ("gx", "gy", "gz"):
                self.bias[axis] = statistics.fmean(sample[axis] for sample in self.calibration_samples)
            self.yaw_reference = self.circular_mean(sample["yaw"] for sample in self.calibration_samples)

        gx_corrected = gx - self.bias["gx"]
        gy_corrected = gy - self.bias["gy"]
        gz_corrected = gz - self.bias["gz"]

        ax, ay, az = self.acc
        acc_roll = math.degrees(math.atan2(ay, az))
        acc_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))

        comp = 0.96
        self.roll = comp * (self.roll + gx_corrected * dt) + (1 - comp) * acc_roll
        self.pitch = comp * (self.pitch + gy_corrected * dt) + (1 - comp) * acc_pitch
        self.yaw_gyro_integrated = self.wrap(self.yaw_gyro_integrated + gz_corrected * dt)
        yaw_reference = mag_yaw if self.yaw_reference is None else self.yaw_reference
        yaw_mag_zeroed = self.wrap(mag_yaw - yaw_reference)

        return {
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw_gyro_integrated,
            "acc_mag": math.sqrt(ax * ax + ay * ay + az * az),
            "gx_corrected": gx_corrected,
            "gy_corrected": gy_corrected,
            "gz_corrected": gz_corrected,
            "yaw_mag_zeroed": yaw_mag_zeroed,
            "yaw_gyro_integrated": self.yaw_gyro_integrated,
            "calibration_samples": len(self.calibration_samples),
        }

    @classmethod
    def circular_mean(cls, angles):
        angles = list(angles)
        if not angles:
            return 0.0
        sin_sum = sum(math.sin(math.radians(angle)) for angle in angles)
        cos_sum = sum(math.cos(math.radians(angle)) for angle in angles)
        return cls.wrap(math.degrees(math.atan2(sin_sum, cos_sum)))


class ASABEDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Maroon Robotics Team - ASABE System Analysis")
        self.root.configure(bg=BG)
        self.configure_display_scaling()
        self.ui_font = self.best_ui_font()
        self.font_cache = {}
        self.logger = StudyLogger()
        self.maximize()

        self.inbox = queue.Queue()
        self.serial_status = {"board01": "starting", "board02": "starting"}
        self.camera_status = "camera idle"
        self.serial_line_age = 999
        self.last_serial_t = 0.0
        self.last_command = "--"
        self.drive_mode = "Drive"
        self.swerve_mode = "unknown"
        self.swerve_mode_key = "unknown"
        self.manual_status = "idle"
        self.pressed = set()
        self.last_motor_sent = None
        self.last_delta_sent = None
        self.delta = 0
        self.delta_dir = None

        self.system_status = {
            "ESP01": "off",
            "ESP02": "off",
            "CAM": "off",
            "ENCODER": "off",
            "SERVO": "off",
            "TOF": "off",
        }
        self.prox = {label: None for label in TOF_LABELS}
        self.last_tof_t = 0.0
        self.tof_rx_count = 0
        self.actuators = {sid: None for sid in SERVO_NAMES}
        self.actuator_changed = {sid: 0.0 for sid in SERVO_NAMES}
        self.crop_green = 0
        self.crop_blue = 0
        self.crop_red = 0
        self.velocity = 0
        self.rpm_filtered = 0.0
        self.rpm_ready = False
        self.rpm_rejected = 0
        self.rpm_status = "waiting"
        self.command_pwm = 0
        self.heading = 0.0
        self.imu = {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "acc_mag": 0.0,
            "gx_corrected": 0.0,
            "gy_corrected": 0.0,
            "gz_corrected": 0.0,
            "yaw_mag_zeroed": 0.0,
            "yaw_gyro_integrated": 0.0,
            "calibration_samples": 0,
        }
        self.has_imu_data = False
        self.imu_filter = ImuFusion()
        self.latest_camera_frame = None
        self.latest_camera_frames = []
        self.latest_camera_sources = []
        self.latest_camera_photo = None
        self.latest_camera_photos = []
        self.detect_button_box = None
        self.detect_flash_until = 0.0
        self.note_status = ""
        self.note_active = False
        self.remote_status = "remote keys starting"

        self.board01 = SerialWorker("board01", ESP1_PORT, self.inbox, self.set_serial_status, self.logger)
        self.board02 = SerialWorker("board02", ESP2_PORT, self.inbox, self.set_serial_status, self.logger)
        self.serial = self.board02
        self.camera = CameraWorker(self.inbox, self.logger)
        self.remote_keys = RemoteKeyServer(self.inbox, self.logger)

        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.note_var = tk.StringVar()
        self.note_entry = tk.Entry(self.root, textvariable=self.note_var, relief="solid", bd=1)
        self.note_entry.bind("<FocusIn>", self.on_note_focus_in)
        self.note_entry.bind("<FocusOut>", self.on_note_focus_out)
        self.note_entry.bind("<Return>", self.on_note_enter)

        self.active_task = None
        self.task_buttons = []
        for i in range(4):
            btn = tk.Button(
                self.root,
                text=f"Task 0{i + 1}",
                relief="flat",
                cursor="hand2",
                command=lambda idx=i: self.select_task(idx),
            )
            self.task_buttons.append(btn)

        self.bind_keys()
        self.board01.start()
        self.board02.start()
        self.camera.start()
        self.remote_keys.start()
        self.root.after(150, self.maximize)
        self.root.after(UI_INTERVAL_MS, self.update_loop)
        self.root.after(COMMAND_INTERVAL_MS, self.command_loop)
        self.root.after(SERVO_POLL_MS, self.poll_servo_positions)
        self.root.after(3000, self.initialize_actuators)

    def maximize(self):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.minsize(960, 640)
        self.root.update_idletasks()
        try:
            self.root.state("zoomed")
            return
        except tk.TclError:
            pass

        try:
            self.root.attributes("-zoomed", True)
            return
        except tk.TclError:
            pass

        self.root.geometry(f"{max(960, sw - 80)}x{max(640, sh - 110)}+24+24")

    def configure_display_scaling(self):
        try:
            dpi = self.root.winfo_fpixels("1i")
            scaling = max(1.0, min(2.0, dpi / 72.0))
            self.root.tk.call("tk", "scaling", scaling)
        except tk.TclError:
            pass

    def best_ui_font(self):
        try:
            available = set(tkfont.families(self.root))
        except tk.TclError:
            available = set()
        if sys.platform == "win32":
            candidates = ("Aptos", "Segoe UI Variable", "Segoe UI", "Arial", "Helvetica")
        else:
            candidates = ("Inter", "Noto Sans", "DejaVu Sans", "Ubuntu", "Cantarell", "Arial", "Helvetica")
        for family in candidates:
            if family in available:
                return family
        return "TkDefaultFont"

    def bind_keys(self):
        if not ENABLE_LOCAL_KEYS:
            self.logger.log_event("dashboard_local_keys_disabled", {
                "enable_with": "ASABE_ENABLE_LOCAL_KEYS=1",
            })
            return
        self.root.bind_all("<KeyPress>", self.on_key_press)
        self.root.bind_all("<KeyRelease>", self.on_key_release)
        self.root.focus_set()

    def set_serial_status(self, source, status):
        self.inbox.put(("serial_status", source, status, time.time()))

    def send_command(self, command, source="ui"):
        self.last_command = command
        sent = self.board02.send(command)
        if source != "board02":
            self.logger.log_command(command, source=source, sent=sent, meta={
                "mode": self.swerve_mode,
                "pwm": self.command_pwm,
                "green": self.crop_green,
                "blue": self.crop_blue,
                "red": self.crop_red,
                "total": self.crop_green + self.crop_blue + self.crop_red,
            })

    def send_b01(self, command, source="ui"):
        self.last_command = command
        sent = self.board01.send(command)
        if source != "board01":
            self.logger.log_command(command, source=source, sent=sent, meta={
                "mode": self.swerve_mode,
                "pwm": self.command_pwm,
                "green": self.crop_green,
                "blue": self.crop_blue,
                "red": self.crop_red,
                "total": self.crop_green + self.crop_blue + self.crop_red,
            })

    def send_servo_command(self, command, sid, value, label, source="ui"):
        self.set_actuator(sid, value)
        self.manual_status = label
        self.send_command(command, source=source)

    def initialize_actuators(self):
        self.logger.log_event("startup_initial_positions", {"commands": STARTUP_COMMANDS})
        self.set_swerve_mode("i")
        for command, label in STARTUP_COMMANDS[1:]:
            if command.startswith("5 "):
                self.send_servo_command(command, 5, 322, label, source="startup")
            elif command.startswith("6 "):
                self.send_servo_command(command, 6, 50, label, source="startup")
            elif command.startswith("7 "):
                self.send_servo_command(command, 7, 51, label, source="startup")
            elif command.startswith("8 "):
                self.send_servo_command(command, 8, 266, label, source="startup")
            elif command == "52":
                self.manual_status = label
                self.send_command(command, source="startup")

    def set_swerve_mode(self, mode_key):
        mode_data = {
            "i": ("Drive", "initial"),
            "m": ("Mid", "mid"),
            "t": ("Perp", "perpendicular"),
        }[mode_key]
        self.drive_mode, self.swerve_mode = mode_data
        command = build_swerve_mode_command(self.swerve_mode)
        targets = SWERVE_MODE_TARGETS[self.swerve_mode]
        self.swerve_mode_key = mode_key
        self.delta = 0
        self.delta_dir = None
        self.last_motor_sent = None
        self.last_delta_sent = None
        for label in WHEEL_LABELS:
            self.set_actuator(WHEEL_SERVO_IDS[label], targets[label])
        self.send_command(command)
        self.send_b01({"i": "I", "m": "M", "t": "T"}[mode_key])

    def normalize_key(self, key):
        key = str(key).lower()
        if key == "space":
            key = " "
        return key

    def handle_control_press(self, key):
        if self.note_active:
            return
        key = self.normalize_key(key)
        self.pressed.add(key)
        handled = True

        if key == "i":
            self.set_swerve_mode("i")
        elif key == "m":
            self.set_swerve_mode("m")
        elif key == "t":
            self.set_swerve_mode("t")
        elif key in SERVO_KEY_COMMANDS:
            command, sid, value, label = SERVO_KEY_COMMANDS[key]
            self.send_servo_command(command, sid, value, label)
        elif key in (" ", "x"):
            self.stop_motors()
        elif key == "g":
            self.crop_green += 1
            self.send_command("G")
            self.send_b01("G")
        elif key == "b":
            self.crop_blue += 1
            self.send_command("B")
            self.send_b01("B")
        elif key == "r":
            self.crop_red += 1
            self.send_command("R")
            self.send_b01("R")
        elif key == "c":
            self.crop_green = self.crop_blue = self.crop_red = 0
            self.send_command("T0")
            self.send_b01("T0")
        elif key == "u":
            self.send_command("D")
        elif key == "p":
            self.send_command("P")
        elif key == "o":
            self.send_command("PING")
        elif key == "y":
            self.send_command("SYS")
        else:
            handled = False

        self.highlighted_key = key
        if handled:
            self.draw()

    def on_key_press(self, event):
        self.handle_control_press(event.keysym)

    def on_canvas_click(self, event):
        if not self.detect_button_box:
            return
        x1, y1, x2, y2 = self.detect_button_box
        if x1 <= event.x <= x2 and y1 <= event.y <= y2:
            self.run_detect()

    def run_detect(self):
        self.last_command = "DETECT"
        self.manual_status = "detect"
        self.detect_flash_until = time.time() + 0.45
        self.logger.log_event("detect_clicked", {"source": "ui"})

        self.draw()

    def on_note_focus_in(self, _event):
        self.note_active = True
        self.pressed.clear()
        self.stop_motors()
        self.delta = 0
        self.delta_dir = None
        self.last_delta_sent = None
        self.note_status = "typing note: robot keys paused"
        self.logger.log_event("note_entry_focus", {"keyboard_controls": "paused"})
        self.draw()

    def on_note_focus_out(self, _event):
        self.note_active = False
        self.pressed.clear()
        self.note_status = "robot keys active"
        self.logger.log_event("note_entry_blur", {"keyboard_controls": "active"})
        self.draw()

    def on_note_enter(self, _event):
        note = self.note_var.get().strip()
        if not note:
            self.root.focus_set()
            return
        self.logger.log_note(note)
        self.note_status = f"logged {datetime.now().strftime('%H:%M:%S')}"
        self.note_var.set("")
        self.root.focus_set()
        self.draw()

    def handle_control_release(self, key):
        if self.note_active:
            return
        key = self.normalize_key(key)
        self.pressed.discard(key)
        if key in ("up", "down") or (self.swerve_mode_key == "m" and key in ("left", "right")):
            self.stop_motors()

    def on_key_release(self, event):
        self.handle_control_release(event.keysym)

    def stop_motors(self):
        self.command_pwm = 0
        self.last_motor_sent = None
        self.manual_status = "idle"
        self.send_command("X")

    def apply_control_outputs(self):
        if self.note_active:
            if self.last_motor_sent is not None:
                self.stop_motors()
            self.delta = 0
            self.delta_dir = None
            self.last_delta_sent = None
            return

        motor = self.wanted_motor_command()
        if motor and motor != self.last_motor_sent:
            self.send_command(motor)
            self.last_motor_sent = motor
            nums = re.findall(r"\d+", motor)
            self.command_pwm = int(nums[0]) if nums else MOTOR_PWM
            self.manual_status = motor
        elif motor is None and self.last_motor_sent is not None:
            self.stop_motors()

        self.update_delta()

    def command_loop(self):
        self.apply_control_outputs()
        self.root.after(COMMAND_INTERVAL_MS, self.command_loop)

    def poll_servo_positions(self):
        self.board02.send("P")
        self.root.after(SERVO_POLL_MS, self.poll_servo_positions)

    def wanted_motor_command(self):
        forward = "up" in self.pressed
        backward = "down" in self.pressed
        left = "left" in self.pressed
        right = "right" in self.pressed

        if forward:
            if self.swerve_mode_key == "i":
                return build_motor_command("F", "forward")
            if self.swerve_mode_key == "t":
                return build_motor_command("L", "left")
        if backward:
            if self.swerve_mode_key == "i":
                return build_motor_command("B", "backward")
            if self.swerve_mode_key == "t":
                return build_motor_command("R", "right")
        if self.swerve_mode_key == "m":
            if left:
                return build_motor_command("CCW", "ccw")
            if right:
                return build_motor_command("CW", "cw")
        return None

    def update_delta(self):
        if self.swerve_mode_key == "m":
            self.delta = 0
            self.delta_dir = None
            self.last_delta_sent = None
            return

        desired_dir = None
        if "left" in self.pressed:
            desired_dir = "left"
        elif "right" in self.pressed:
            desired_dir = "right"

        desired_dir = self.effective_delta_direction(desired_dir)

        if desired_dir:
            if self.delta_dir != desired_dir:
                self.delta = 0
                self.last_delta_sent = None
            self.delta_dir = desired_dir
            self.delta = min(DELTA_MAX, self.delta + DELTA_STEP)
        else:
            self.delta = max(0, self.delta - DELTA_DECAY)
            if self.delta == 0:
                self.delta_dir = None

        if self.delta_dir is None and self.delta == 0:
            command = self.delta_command("right", 0)
        else:
            command = self.delta_command(self.delta_dir, self.delta)

        if command and command != self.last_delta_sent:
            self.send_command(command)
            self.last_delta_sent = command

    def delta_command(self, direction, amount):
        if self.swerve_mode_key == "i":
            return f"DL{amount}" if direction == "left" else f"DR{amount}"
        if self.swerve_mode_key == "t":
            return f"DB{amount}" if direction == "left" else f"DF{amount}"
        return None

    def effective_delta_direction(self, direction):
        if direction is None:
            return None
        if "up" not in self.pressed:
            return direction
        return "right" if direction == "left" else "left"

    def update_loop(self):
        self.process_inbox()
        if self.last_serial_t:
            self.serial_line_age = time.time() - self.last_serial_t
        if self.last_tof_t:
            self.system_status["TOF"] = "ok" if time.time() - self.last_tof_t <= TOF_STALE_SECONDS else "warn"
        self.draw()
        self.root.after(UI_INTERVAL_MS, self.update_loop)

    def process_inbox(self):
        for _ in range(300):
            try:
                item = self.inbox.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "serial_status":
                _, source, payload, ts = item
                self.serial_status[source] = payload
                key = "ESP01" if source == "board01" else "ESP02"
                self.system_status[key] = "ok" if payload == "live" else "warn"
            elif kind == "camera_status":
                _, payload, ts = item
                self.camera_status = payload
                self.system_status["CAM"] = "ok" if "live" in payload else "warn"
            elif kind == "camera_frame":
                _, payload, ts = item
                if isinstance(payload, dict):
                    self.latest_camera_frames = payload.get("frames", [])
                    self.latest_camera_sources = payload.get("sources", [])
                    self.latest_camera_frame = self.latest_camera_frames[0] if self.latest_camera_frames else None
                else:
                    self.latest_camera_frame = payload
                    self.latest_camera_frames = [payload]
                    self.latest_camera_sources = [CAMERA_DEVICE]
            elif kind == "serial":
                _, source, payload, ts = item
                self.last_serial_t = ts
                self.parse_serial_line(payload, ts, source)
            elif kind == "serial_error":
                _, source, payload, ts = item
                self.system_status["ESP01" if source == "board01" else "ESP02"] = "warn"
            elif kind == "remote_status":
                _, payload, ts = item
                self.remote_status = payload
            elif kind == "remote_key":
                _, event, key, ts = item
                if event in ("down", "tap"):
                    self.handle_control_press(key)
                if event in ("up", "tap"):
                    self.handle_control_release(key)
                self.apply_control_outputs()

    def parse_serial_line(self, line, ts, source):
        if line.startswith("RPM,"):
            self.handle_rpm(self.safe_float(line.split(",", 1)[1], self.velocity))
            self.logger.write_csv("rpm", {"avg": self.safe_float(line.split(",", 1)[1], ""), "BR": "", "FR": "", "BL": "", "FL": "", "raw": line}, ts)
        elif line.startswith("RPM ["):
            self.parse_wheel_rpm(line, ts)
        elif line.startswith("TOF,["):
            self.parse_tof(line, ts)
            self.logger.write_csv("tof", {**{label: self.prox.get(label, "") for label in TOF_LABELS}, "raw": line}, ts)
            self.system_status["TOF"] = "ok"
        elif line.startswith("IMU,"):
            self.parse_imu(line, ts, log_raw=True)
        elif line.startswith("POS,"):
            self.parse_pos(line)
            servo_row = {f"s{i}": self.actuators.get(i, "") for i in range(1, 9)}
            servo_row["raw"] = line
            self.logger.write_csv("servos", servo_row, ts)
            self.system_status["SERVO"] = "ok"
        elif line.startswith("ACK,MODE,"):
            parts = line.split(",")
            if len(parts) >= 3:
                self.drive_mode = {"initial": "Drive", "mid": "Mid", "perpendicular": "Perp"}.get(parts[2], self.drive_mode)
                self.swerve_mode = parts[2]
                self.swerve_mode_key = {"initial": "i", "mid": "m", "perpendicular": "t"}.get(parts[2], self.swerve_mode_key)
        elif line.startswith("ACK,COLOR,"):
            self.parse_color(line)
            total = self.crop_green + self.crop_blue + self.crop_red
            self.logger.write_csv("colors", {
                "green": self.crop_green,
                "blue": self.crop_blue,
                "red": self.crop_red,
                "total": total,
                "raw": line,
            }, ts)
        elif line.startswith("ACK,DISPENSER_SERVO,"):
            pass
        elif line.startswith("PONG,") or line.startswith("SYS,"):
            if "BOARD01_FINAL" in line or source == "board01":
                self.system_status["ESP01"] = "ok"
            if "board02_final" in line or source == "board02":
                self.system_status["ESP02"] = "ok"
        elif line.startswith("WARN,"):
            self.system_status["SERVO"] = "warn"
        elif line.startswith("ERR,"):
            self.system_status["ESP02"] = "warn"

    def parse_tof(self, line, ts=None):
        body = line[line.find("[") + 1:line.rfind("]")]
        updated = False
        for item in body.split(","):
            if ":" not in item:
                continue
            label, value = item.split(":", 1)
            label = label.strip()
            value = value.strip()
            if label in self.prox:
                try:
                    numeric = float(value)
                    self.prox[label] = int(numeric) if numeric.is_integer() else numeric
                except ValueError:
                    self.prox[label] = value
                updated = True
        if updated:
            self.last_tof_t = ts or time.time()
            self.tof_rx_count += 1

    def handle_rpm(self, raw_rpm):
        raw_rpm = abs(float(raw_rpm))

        if raw_rpm > RPM_REASONABLE_MAX:
            self.rpm_rejected += 1
            self.rpm_status = f"reject high {raw_rpm:.0f}"
            self.system_status["ENCODER"] = "warn"
            return

        if self.rpm_ready and abs(raw_rpm - self.rpm_filtered) > RPM_MAX_STEP:
            self.rpm_rejected += 1
            self.rpm_status = f"reject jump {raw_rpm:.0f}"
            self.system_status["ENCODER"] = "warn"
            return

        if not self.rpm_ready:
            self.rpm_filtered = raw_rpm
            self.rpm_ready = True
        else:
            self.rpm_filtered = (
                self.rpm_filtered * (1.0 - RPM_SMOOTH_ALPHA)
                + raw_rpm * RPM_SMOOTH_ALPHA
            )

        self.velocity = int(round(self.rpm_filtered))
        self.rpm_status = "filtered"
        self.system_status["ENCODER"] = "ok"

    def parse_wheel_rpm(self, line, ts):
        values = {label: "" for label in ("BR", "FR", "BL", "FL")}
        for label, value in re.findall(r"(BR|FR|BL|FL):\s*(-?\d+(?:\.\d+)?)", line):
            values[label] = value
        numeric = [abs(float(value)) for value in values.values() if value != ""]
        avg = sum(numeric) / len(numeric) if numeric else 0.0
        if numeric:
            self.handle_rpm(avg)
        self.logger.write_csv("rpm", {"avg": f"{avg:.2f}" if numeric else "", **values, "raw": line}, ts)

    def parse_imu(self, line, ts, log_raw=False):
        parts = line.split(",")
        if len(parts) != 10 or parts[1] == "ERR":
            return
        values = [self.safe_float(v, 0.0) for v in parts[1:]]
        self.imu = self.imu_filter.update(values, ts)
        self.heading = self.imu["yaw"]
        self.has_imu_data = True
        if log_raw:
            keys = ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]
            row = dict(zip(keys, values))
            row.update({
                "roll": self.imu["roll"],
                "pitch": self.imu["pitch"],
                "yaw": self.imu["yaw"],
                "acc_mag": self.imu["acc_mag"],
                "gx_corrected": self.imu["gx_corrected"],
                "gy_corrected": self.imu["gy_corrected"],
                "gz_corrected": self.imu["gz_corrected"],
                "yaw_mag_zeroed": self.imu["yaw_mag_zeroed"],
                "yaw_gyro_integrated": self.imu["yaw_gyro_integrated"],
                "calibration_samples": self.imu["calibration_samples"],
                "raw": line,
            })
            self.logger.write_csv("imu", row, ts)

    def parse_pos(self, line):
        for sid, value in enumerate(line.split(",")[1:], start=1):
            if value != "ERR":
                self.set_actuator(sid, self.safe_float(value, 0))

    def parse_color(self, line):
        parts = line.split(",")
        if len(parts) >= 7:
            self.crop_green = int(self.safe_float(parts[3], self.crop_green))
            self.crop_blue = int(self.safe_float(parts[4], self.crop_blue))
            self.crop_red = int(self.safe_float(parts[5], self.crop_red))

    def set_actuator(self, sid, value):
        if sid in self.actuators:
            self.actuators[sid] = value
            self.actuator_changed[sid] = time.time()

    def camera_pan_label(self):
        value = self.actuators.get(7)
        if value is None:
            return "--"
        if abs(value - 51.0) <= 2.0:
            return "FRONT"
        if abs(value - 141.0) <= 2.0:
            return "LEFT"
        if abs(value - 321.0) <= 2.0:
            return "RIGHT"
        return "--"

    def overhead_cam_label(self):
        value = self.actuators.get(8)
        if value is None:
            return "--"
        if abs(value - 266.0) <= 2.0:
            return "FRONT"
        if abs(value - 179.0) <= 2.0:
            return "LEFT"
        if abs(value - 358.0) <= 2.0:
            return "RIGHT"
        return "--"

    def select_task(self, index):
        self.active_task = None if self.active_task == index else index
        self.draw()

    @staticmethod
    def safe_float(value, fallback):
        try:
            return float(value)
        except Exception:
            return fallback


    def scale(self):
        W = max(1, self.canvas.winfo_width())
        H = max(1, self.canvas.winfo_height())
        sx = W / 1320.0
        sy = H / 900.0
        sf = math.sqrt(sx * sy)
        return sx, sy, max(0.64, min(1.50, sf))

    def sx(self):
        return self.scale()[2]

    def xywh(self, x, y, w, h):
        sx, sy, _ = self.scale()
        return x * sx, y * sy, w * sx, h * sy

    def font(self, size, weight="normal", family=None):
        point_size = max(7, int(round(size * self.sx())))
        key = (family or self.ui_font, point_size, weight)
        if key not in self.font_cache:
            self.font_cache[key] = tkfont.Font(
                root=self.root,
                family=key[0],
                size=point_size,
                weight=weight,
            )
        return self.font_cache[key]

    def draw(self):
        c = self.canvas
        c.delete("all")
        c.create_rectangle(0, 0, c.winfo_width(), c.winfo_height(), fill=BG, outline="")
        self.draw_header()
        self.draw_camera()
        self.draw_system()
        self.draw_drive_mode()
        self.draw_proximity()
        self.draw_velocity()
        self.draw_actuators()
        self.draw_crop_count()
        self.draw_imu_status()
        self.draw_footer()
        self.place_note_entry()
        self.place_task_buttons()

    def text(self, x, y, text, fill=TEXT, size=12, weight="normal", anchor="nw", family=None, **kwargs):
        x, y, _, _ = self.xywh(x, y, 0, 0)
        return self.canvas.create_text(x, y, text=text, fill=fill, font=self.font(size, weight, family), anchor=anchor, **kwargs)

    def rect(self, x, y, w, h, fill, outline="", width=1):
        x, y, w, h = self.xywh(x, y, w, h)
        return self.canvas.create_rectangle(x, y, x + w, y + h, fill=fill, outline=outline, width=max(1, int(width * self.sx())))

    def oval(self, x, y, w, h, fill="", outline="", width=1):
        x, y, w, h = self.xywh(x, y, w, h)
        return self.canvas.create_oval(x, y, x + w, y + h, fill=fill, outline=outline, width=max(1, int(width * self.sx())))

    def line(self, x1, y1, x2, y2, fill, width=1, dash=None):
        x1, y1, _, _ = self.xywh(x1, y1, 0, 0)
        x2, y2, _, _ = self.xywh(x2, y2, 0, 0)
        return self.canvas.create_line(x1, y1, x2, y2, fill=fill, width=max(1, int(width * self.sx())), dash=dash)

    def card(self, x, y, w, h, title):
        self.rect(x, y, w, h, CARD, CARD_BORDER)
        self.rect(x, y, w, 3, MAROON, "")
        self.rect(x + 20, y + 15, 3, 16, MAROON, "")
        self.text(x + 56, y + 15, title, MAROON, 15, "bold")

    def draw_header(self):
        self.text(40, 16, "MAROON Robotic Team", MAROON, 28, "bold", anchor="nw")
        self.text(40, 55, "ASABE Robotics Student Design Competition", MUTED, 8, "bold", anchor="nw")

    def draw_camera(self):
        self.rect(28, 90, 752, 560, BLACK_PANEL, BLACK_PANEL)

        frames = self.latest_camera_frames[:2]
        self.latest_camera_photos = []
        if frames and Image is not None and ImageTk is not None:
            boxes = [(50, 145, 350, 330), (410, 145, 350, 330)]
            for index, frame in enumerate(frames):
                bx, by, bw, bh = boxes[index]
                self.rect(bx, by, bw, bh, "#0d1117", "#2a2d35")
                x, y, w, h = self.xywh(bx + 8, by + 34, bw - 16, bh - 48)
                img = Image.fromarray(frame)
                img.thumbnail((int(w), int(h)))
                photo = ImageTk.PhotoImage(img)
                self.latest_camera_photos.append(photo)
                self.canvas.create_image(x + w / 2, y + h / 2, image=photo)
                source = self.latest_camera_sources[index] if index < len(self.latest_camera_sources) else "unknown"
                label = CAMERA_LABELS[index] if index < len(CAMERA_LABELS) else f"CAM {chr(65 + index)}"
                self.text(bx + 14, by + 10, label, "#f1f1ed", 12, "bold")
        else:
            self.text(404, 350, "awaiting feed", "#8898aa", 14, anchor="center")

        self.text(64, 480, self.camera_pan_label(), "#f1f1ed", 14, "bold")
        self.text(760, 480, self.overhead_cam_label(), "#f1f1ed", 14, "bold", anchor="ne")

    def status_pill(self, x, y, text, status):
        color = {"ok": GREEN, "warn": YELLOW, "bad": RED, "off": MUTED}.get(status, MUTED)
        bg = {"ok": "#eaf7ee", "warn": "#fef9e7", "bad": "#fdecea", "off": "#f0f2f5"}.get(status, "#f0f2f5")
        border = {"ok": "#a8d5b5", "warn": "#f0d060", "bad": "#e8a0a0", "off": "#dde3eb"}.get(status, "#dde3eb")
        self.rect(x, y, 92, 29, bg, border)
        self.oval(x + 14, y + 10, 8, 8, color, color)
        self.text(x + 30, y + 7, text, "#333333", 9)

    def draw_system(self):
        self.card(805, 90, 488, 132, "SYSTEM")
        positions = [(839, 143), (951, 143), (1063, 143), (839, 184), (951, 184), (1063, 184)]
        for name, (x, y) in zip(["ESP01", "ESP02", "CAM", "ENCODER", "SERVO", "TOF"], positions):
            self.status_pill(x, y, name, self.system_status.get(name, "off"))

    def mode_indicator(self, x, y, label, active):
        bg = MAROON if active else "#f0f2f5"
        fg = "white" if active else "#5a6678"
        border = MAROON_DARK if active else "#dde3eb"
        dot = "#ffffff" if active else "#aab4c0"
        self.rect(x, y, 94, 31, bg, border)
        self.oval(x + 16, y + 11, 8, 8, dot, dot)
        self.text(x + 35, y + 7, label, fg, 9)

    def draw_drive_mode(self):
        self.card(805, 238, 488, 150, "DRIVE MODE")
        for label, x in [("Drive", 839), ("Mid", 955), ("Perp", 1071)]:
            self.mode_indicator(x, 300, label, self.drive_mode == label)
        self.text(839, 355, "LAST COMMAND", MUTED, 10)
        self.text(1055, 349, self.last_command, MAROON, 16, "bold")
        self.text(839, 374, "arrows drive/steer  space stop  u dispense", "#8f8a82", 7)
        self.text(839, 386, "1-4 cam front/left/right/back  5-9 arm", "#8f8a82", 7)

    def draw_proximity(self):
        self.card(28, 675, 300, 185, "PROXIMITY")
        cx, cy = 178, 786
        tof_age = time.time() - self.last_tof_t if self.last_tof_t else None
        tof_note = "TOF --" if tof_age is None else f"TOF {tof_age:.1f}s #{self.tof_rx_count}"
        tof_color = MUTED if tof_age is None or tof_age <= TOF_STALE_SECONDS else RED
        self.text(294, 694, tof_note, tof_color, 7, "bold", anchor="e")
        self.oval(cx - 92, cy - 46, 184, 92, "", "#dde3eb")
        self.line(cx - 72, cy, cx + 72, cy, "#dde3eb", dash=(3, 4))
        self.line(cx, cy - 42, cx, cy + 42, "#dde3eb", dash=(3, 4))
        self.rect(cx - 22, cy - 27, 44, 56, "#f0f2f5", "#dde3eb")
        self.polygon([(cx - 28, cy - 27), (cx + 28, cy - 27), (cx + 22, cy - 44), (cx - 22, cy - 44)], MAROON, MAROON_DARK)
        for wx, wy in [(cx - 36, cy - 24), (cx + 36, cy - 24), (cx - 36, cy + 27), (cx + 36, cy + 27)]:
            self.rect(wx - 6, wy - 10, 12, 20, "#191916", "#191916")
        self.text(cx, 722, "FRONT", MUTED, 8, anchor="center")
        self.text(cx, 854, "BACK", MUTED, 8, anchor="center")
        cam_val = self.prox.get("CAM")
        self.text(cx, cy - 5, "CAM", "#777", 7, anchor="center")
        self.text(cx, cy + 15, "N/A" if cam_val is None else str(cam_val), self.prox_color(cam_val), 9, "bold", anchor="center")
        points = {
            "FLF": (cx - 48, cy - 52), "FRF": (cx + 48, cy - 52),
            "FLS": (cx - 96, cy - 11), "FRS": (cx + 96, cy - 11),
            "BLS": (cx - 96, cy + 37), "BRS": (cx + 96, cy + 37),
        }
        for name, (x, y) in points.items():
            val = self.prox.get(name)
            self.text(x, y - 10, name, MUTED, 7, anchor="center")
            self.text(x, y + 1, "N/A" if val is None else str(val), self.prox_color(val), 8, "bold", anchor="center")

    def polygon(self, points, fill, outline):
        scaled = []
        for x, y in points:
            sx, sy, _, _ = self.xywh(x, y, 0, 0)
            scaled.extend([sx, sy])
        self.canvas.create_polygon(scaled, fill=fill, outline=outline)

    def prox_color(self, value):
        if not isinstance(value, int):
            return MUTED
        if value < 100:
            return RED
        if value < 200:
            return YELLOW
        return GREEN

    def draw_velocity(self):
        self.card(350, 675, 290, 185, "VELOCITY")
        self.text(378, 733, "N/A" if not self.rpm_ready else str(self.velocity), TEXT, 38, "bold")
        self.text(378, 795, "RPM", MUTED, 10)
        self.line(378, 820, 612, 820, "#dde3eb")
        self.text(378, 829, "intent", MUTED, 10)
        self.text(500, 829, f"PWM {self.command_pwm}", "#777", 10)
        self.text(378, 846, f"{self.rpm_status}  rejected:{self.rpm_rejected}", MUTED, 8)

    def draw_actuators(self):
        self.card(805, 405, 488, 258, "ACTUATORS")
        self.text(1248, 420, "poll 1 Hz", MUTED, 9, "bold", anchor="ne")
        ranked = sorted(self.actuators, key=lambda sid: self.actuator_changed[sid], reverse=True)
        for row, sid in enumerate(ranked[:8]):
            y = 456 + row * 25
            val = self.actuators[sid]
            age = time.time() - self.actuator_changed[sid]
            color = MAROON if age < 2 else TEXT
            self.text(839, y, SERVO_NAMES[sid], "#555", 11)
            self.text(1190, y, "--" if val is None else f"{val:.1f}", color, 11, "bold", anchor="ne")
            self.line(839, y + 22, 1248, y + 22, "#eef1f5")

    def draw_crop_count(self):
        self.card(662, 675, 310, 185, "CROP COUNT")
        total = self.crop_green + self.crop_blue + self.crop_red
        self.text(817, 710, str(total), MAROON, 48, "bold", anchor="n")
        self.text(817, 768, "TOTAL", MUTED, 10, anchor="n")
        for label, color, value, x in [
            ("GRN", GREEN, self.crop_green, 694),
            ("BLU", BLUE, self.crop_blue, 786),
            ("RED", RED, self.crop_red, 878),
        ]:
            self.rect(x, 794, 62, 31, "#f1efe8", "#d4cec4")
            self.text(x + 10, 799, str(value), color, 13, "bold")
            self.text(x + 34, 804, label, color, 8)

        bx, by, bw, bh = 719, 836, 198, 17
        clicked = time.time() < self.detect_flash_until
        btn_fill = GREEN if clicked else MAROON
        btn_border = "#137832" if clicked else MAROON_DARK
        btn_text = "DETECTED" if clicked else "DETECT"
        self.rect(bx, by, bw, bh, btn_fill, btn_border)
        self.text(bx + bw / 2, by + 2, btn_text, "white", 10, "bold", anchor="n")
        sx, sy, sw, sh = self.xywh(bx, by, bw, bh)
        self.detect_button_box = (sx, sy, sx + sw, sy + sh)

    def draw_imu_status(self):
        self.card(995, 675, 298, 185, "IMU / CONTROL")
        rows = [
            ("ROLL", self.imu["roll"], "deg"),
            ("PITCH", self.imu["pitch"], "deg"),
            ("YAW", self.imu["yaw"], "deg"),
            ("ACC", self.imu["acc_mag"], "m/s2"),
        ]
        for i, (label, value, unit) in enumerate(rows):
            y = 724 + i * 24
            self.text(1025, y, label, MUTED, 10, "bold")
            val_text = "    N/A" if not self.has_imu_data else f"{value:7.2f}"
            self.text(1114, y, val_text, TEXT, 11, "bold")
            self.text(1212, y, unit, MUTED, 9)

    def draw_footer(self):
        self.text(28, 872, f"analysis run: {self.logger.run_id}", "#8898aa", 10)
        self.text(780, 872, self.remote_status, "#8898aa", 10)
        self.text(1010, 872, self.note_status or "operator note: press Enter to log", "#8898aa", 10)

    def place_note_entry(self):
        x, y, w, h = self.xywh(350, 868, 610, 24)
        self.note_entry.configure(font=self.font(10), fg=TEXT, bg="#ffffff", insertbackground=MAROON)
        self.note_entry.place(x=x, y=y, width=w, height=h)

    def place_task_buttons(self):
        for i, btn in enumerate(self.task_buttons):
            x, y, w, h = self.xywh(935 + i * 90, 22, 80, 24)
            selected = self.active_task == i
            btn.configure(
                font=self.font(8, "bold"),
                bg=MAROON if selected else "#f0f2f5",
                fg="white" if selected else "#5a6678",
                activebackground=MAROON_DARK if selected else "#dde3eb",
                activeforeground="white" if selected else TEXT,
            )
            btn.place(x=x, y=y, width=w, height=h)

    def close(self):
        try:
            self.board02.send("X")
        except Exception:
            pass
        self.board01.stop()
        self.board02.stop()
        self.camera.stop()
        self.remote_keys.stop()
        self.logger.close()
        self.root.destroy()


if __name__ == "__main__":
    enable_high_dpi()
    root = tk.Tk()
    app = ASABEDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
