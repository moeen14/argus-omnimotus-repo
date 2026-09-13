from __future__ import annotations

import argparse
import math
import os
import select
import statistics
import sys
import termios
import threading
import time
import tty

try:
    import serial
except ImportError:
    serial = None

ESP1_PORT = os.environ.get("ASABE_ESP1_PORT", "/dev/esp32_01")
ESP2_PORT = os.environ.get("ASABE_ESP2_PORT", "/dev/esp32_02")
BAUD = 921600

SPEED = 120
BURST_PULSE_MS = 30
LINEAR_RESEND_S = 0.05
TOF_THRESHOLD_MM = 120.0

TOF_LABELS = ["FRS", "BRS", "BLS", "FLF", "FLS", "CAM", "FRF"]
SIDE_TOF_LABELS = {"R": ("BRS", "FRS"), "L": ("BLS", "FLS")}


ROTATE_TARGET_DEG = 180.0
ROTATE_TOLERANCE_DEG = 5.0
ROTATE_SPEED = 110
ROTATE_MIN_PULSE_MS = 6
ROTATE_MAX_PULSE_MS = 20
ROTATE_ERROR_SATURATION_DEG = 15.0

IMU_CALIBRATION_MIN_SAMPLES = 300
IMU_STILL_GYRO_MAX_DPS = 2.0

_SERIAL_LOCK = threading.Lock()


def send(ser, cmd):
    with _SERIAL_LOCK:
        ser.write((cmd + "\n").encode())
        ser.flush()


TOF_RESET_RETRY_DELAY_S = 2.0


def reset_esp1_tofs(ser1):


    attempt = 0
    while True:
        attempt += 1
        print(f"[init] Resetting Board 01 ToF sensors (attempt {attempt})...")
        send(ser1, "F")

        deadline = time.time() + 6.0
        ok = total = None
        while time.time() < deadline:
            raw = ser1.readline()
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if not line:
                continue
            if line.startswith("ACK,TOF_RESET,DONE"):
                parts = line.split(",")[-1].split("/")
                if len(parts) == 2:
                    try:
                        ok, total = int(parts[0]), int(parts[1])
                    except ValueError:
                        ok = total = None
                break

        if ok is not None and ok == total:
            print(f"[init] Board 01 ToF reset OK — {ok}/{total} sensors reporting. Safe to proceed.")
            return

        seen = f"{ok}/{total}" if ok is not None else "no ACK reply seen"
        print(f"[init] Board 01 ToF reset INCOMPLETE ({seen}) — the robot will NOT start until "
              f"every sensor resets successfully. Retrying in {TOF_RESET_RETRY_DELAY_S:.1f}s...")
        time.sleep(TOF_RESET_RETRY_DELAY_S)


def drive_burst(ser, direction, speed=SPEED, pulse_ms=BURST_PULSE_MS):
    send(ser, f"{direction}{speed}")
    time.sleep(pulse_ms / 1000.0)
    send(ser, "X")


def compute_pulse_ms(error_mag, deadband, saturation, min_pulse_ms, max_pulse_ms):
    mag = min(abs(error_mag), saturation)
    if mag <= deadband:
        return 0.0
    span = saturation - deadband
    frac = (mag - deadband) / span
    return min_pulse_ms + frac * (max_pulse_ms - min_pulse_ms)


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
        dt = 0.02 if self.last_t is None else max(0.005, min(0.15, now - self.last_t))
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

        return {"yaw": self.yaw_gyro_integrated, "roll": self.roll, "pitch": self.pitch}

    @classmethod
    def circular_mean(cls, angles):
        angles = list(angles)
        if not angles:
            return 0.0
        sin_sum = sum(math.sin(math.radians(angle)) for angle in angles)
        cos_sum = sum(math.cos(math.radians(angle)) for angle in angles)
        return cls.wrap(math.degrees(math.atan2(sin_sum, cos_sum)))


class Telemetry:


    def __init__(self):
        self.tof = {label: None for label in TOF_LABELS}
        self.heading = None
        self._imu_fusion = ImuFusion()

    def update_imu(self, values, now):
        self.heading = self._imu_fusion.update(values, now)["yaw"]


def esp1_reader(ser, stop_event, telemetry):


    while not stop_event.is_set():
        try:
            raw = ser.readline()
        except Exception:
            break
        if not raw:
            continue
        line = raw.decode(errors="ignore").strip()
        if not line:
            continue
        if line.startswith("TOF,["):
            body = line[line.find("[") + 1:line.rfind("]")]
            for item in body.split(","):
                if ":" not in item:
                    continue
                label, value = item.split(":", 1)
                label = label.strip()
                if label in telemetry.tof:
                    try:
                        telemetry.tof[label] = float(value.strip())
                    except ValueError:
                        pass
        elif line.startswith("IMU,"):
            parts = line.split(",")
            if len(parts) == 10 and parts[1] != "ERR":
                try:
                    values = [float(v) for v in parts[1:]]
                except ValueError:
                    continue
                telemetry.update_imu(values, time.time())


def serial_reader(ser, stop_event):

    while not stop_event.is_set():
        try:
            ser.readline()
        except Exception:
            break


def read_key(fd):
    if not select.select([fd], [], [], 0)[0]:
        return None
    data = os.read(fd, 1)
    if not data:
        return None
    return data.decode(errors="ignore").lower()


def parse_args():
    parser = argparse.ArgumentParser(description="ToF drive + 180-rotate tuning harness.")
    parser.add_argument("--speed", type=int, default=SPEED)
    parser.add_argument("--threshold", type=float, default=TOF_THRESHOLD_MM)
    parser.add_argument("--esp1", default=ESP1_PORT)
    parser.add_argument("--esp2", default=ESP2_PORT)
    return parser.parse_args()


def main():
    args = parse_args()
    if serial is None:
        print("pyserial not found. Install it or activate nanosam-env.")
        return 1

    speed = args.speed
    threshold = args.threshold

    print(f"Connecting to ESP1 (sensors) {args.esp1} and ESP2 (actuators) {args.esp2} at {BAUD} baud...")
    ser1 = serial.Serial(args.esp1, BAUD, timeout=0.1)
    ser2 = serial.Serial(args.esp2, BAUD, timeout=0.1)
    time.sleep(2.0)
    ser1.reset_input_buffer()
    ser2.reset_input_buffer()
    reset_esp1_tofs(ser1)

    telemetry = Telemetry()
    stop_event = threading.Event()
    threading.Thread(target=esp1_reader, args=(ser1, stop_event, telemetry), daemon=True).start()
    threading.Thread(target=serial_reader, args=(ser2, stop_event), daemon=True).start()

    send(ser2, "T")

    print("=" * 60)
    print(" ToF drive + 180-rotate tuner")
    print(f" speed={speed}  threshold={threshold:.0f}mm  burst_pulse={BURST_PULSE_MS}ms")
    print(" l/r   drive that direction until ToF trips, then rotate 180")
    print(" b     toggle burst <-> linear driving style")
    print(" x     stop and return to idle (no rotation)")
    print(" q     quit (stops motors first)")
    print("=" * 60)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    burst_mode = True
    state = "idle"
    direction = None
    target_heading = None
    last_linear_send = 0.0
    last_status = ""

    try:
        tty.setraw(fd)
        while True:
            key = read_key(fd)
            if key == "q":
                print("\r\nQuitting...")
                break
            elif key == "b":
                burst_mode = not burst_mode
                print(f"\r\nDrive style -> {'burst' if burst_mode else 'linear'}")
            elif key == "x":
                send(ser2, "X")
                state = "idle"
                direction = None
                print("\r\nStopped -> idle. Press l/r to drive.")
            elif key in ("l", "r") and state == "idle":
                direction = key.upper()
                state = "driving"
                last_linear_send = 0.0
                side_a, side_b = SIDE_TOF_LABELS[direction]
                print(f"\r\nDriving {direction} ({'burst' if burst_mode else 'linear'} @ {speed}), "
                      f"watching {side_a}/{side_b} for <{threshold:.0f}mm...")

            if state == "driving":
                side_a, side_b = SIDE_TOF_LABELS[direction]
                readings = (telemetry.tof.get(side_a), telemetry.tof.get(side_b))
                if any(r is not None and r < threshold for r in readings):
                    send(ser2, "X")
                    send(ser2, "M")
                    time.sleep(0.2)
                    state = "rotating"
                    target_heading = None
                    print(f"\r\nToF tripped ({side_a}={readings[0]} {side_b}={readings[1]}) "
                          f"— rotating {ROTATE_TARGET_DEG:.0f} deg.")
                elif burst_mode:
                    drive_burst(ser2, direction, speed=speed)
                else:
                    now = time.time()
                    if now - last_linear_send >= LINEAR_RESEND_S:
                        send(ser2, f"{direction}{speed}")
                        last_linear_send = now

            elif state == "rotating":
                if telemetry.heading is None:
                    send(ser2, "X")
                else:
                    if target_heading is None:
                        target_heading = ImuFusion.wrap(telemetry.heading + ROTATE_TARGET_DEG)
                    error = ImuFusion.wrap(target_heading - telemetry.heading)
                    if abs(error) <= ROTATE_TOLERANCE_DEG:
                        send(ser2, "X")
                        send(ser2, "T")
                        state = "idle"
                        direction = None
                        print("\r\nRotation complete. Press l/r to drive.")
                    else:
                        rot_dir = "CCW" if error > 0 else "CW"
                        pulse_ms = compute_pulse_ms(
                            error, ROTATE_TOLERANCE_DEG, ROTATE_ERROR_SATURATION_DEG,
                            ROTATE_MIN_PULSE_MS, ROTATE_MAX_PULSE_MS)
                        send(ser2, f"{rot_dir}{ROTATE_SPEED}")
                        time.sleep(pulse_ms / 1000.0)
                        send(ser2, "X")

            side_a, side_b = SIDE_TOF_LABELS.get(direction, ("--", "--"))
            tof_a = telemetry.tof.get(side_a) if direction else None
            tof_b = telemetry.tof.get(side_b) if direction else None
            heading_text = "--" if telemetry.heading is None else f"{telemetry.heading:+.1f}"
            status = (f"state={state:<8} dir={direction or '-':<1} "
                      f"style={'burst' if burst_mode else 'linear':<6} "
                      f"{side_a}={tof_a} {side_b}={tof_b} heading={heading_text}")
            if status != last_status:
                print(f"\r{status}", end="", flush=True)
                last_status = status

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\r\nStopped by user.")
    finally:
        send(ser2, "X")
        stop_event.set()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        ser1.close()
        ser2.close()
        print("\r\nCleanup complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
