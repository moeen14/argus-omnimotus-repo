import argparse
import csv
import json
import os
import re
import select
import statistics
import sys
import termios
import threading
import time
import tty
from datetime import datetime, timezone
from pathlib import Path

try:
    import serial
except ImportError:
    serial = None


BAUD = 921600
DEFAULT_ESP1_PORT = os.environ.get("ASABE_ESP1_PORT", "/dev/esp32_01")
DEFAULT_ESP2_PORT = os.environ.get("ASABE_ESP2_PORT", "/dev/esp32_02")
RUN_SECONDS = 5.0
MAX_PASSES = 100
DEFAULT_SPEED = 175
MAX_PWM = 255
READINESS_SECONDS = 6.0
IMU_CALIBRATION_SECONDS = 20.0
IMU_CALIBRATION_MIN_SAMPLES = 300
T_LEFT_FRONT_EXTRA_PWM_OFFSET = 6
T_RIGHT_BACK_EXTRA_PWM_OFFSET = 8
I_FORWARD_LEFT_EXTRA_PWM_OFFSET = -5
I_BACKWARD_RIGHT_EXTRA_PWM_OFFSET = -10
WHEEL_LABELS = ("BR", "FR", "BL", "FL")
RPM_DETAIL_RE = re.compile(r"(BR|FR|BL|FL):\s*(-?\d+(?:\.\d+)?)")
IMU_LABELS = ("ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw")
IMU_CORRECTED_LABELS = (
    "gx_corrected",
    "gy_corrected",
    "gz_corrected",
    "yaw_zeroed",
    "yaw_gyro_integrated",
)
MODE_COMMANDS = {
    "i": {"name": "initial", "command": "I"},
    "m": {"name": "mid", "command": "M"},
    "t": {"name": "perpendicular", "command": "T"},
}
SWERVE_MODE_TARGETS = {
    "initial": {"BR": 37.0, "FR": 6.0, "BL": 145.0, "FL": 221.0},
    "mid": {"BR": 306.0, "FR": 96.0, "BL": 235.0, "FL": 131.0},
    "perpendicular": {"BR": 220.0, "FR": 180.0, "BL": 325.0, "FL": 42.0},
}
ARROW_COMMANDS = {
    "i": {
        "up": ("forward", "F"),
        "down": ("backward", "B"),
    },
    "m": {
        "left": ("ccw", "CCW"),
        "right": ("cw", "CW"),
    },
    "t": {
        "left": ("left", "L"),
        "right": ("right", "R"),
    },
}
DIRECT_MOVEMENT_KEYS = {
    "f": ("i", "up"),
    "b": ("i", "down"),
    "l": ("t", "left"),
    "r": ("t", "right"),
    "c": ("m", "right"),
    "v": ("m", "left"),
}
BASE_BALANCED_PWM_TARGETS = {


    "forward": {"BR": 175, "FR": 178, "BL": 174, "FL": 172},
    "backward": {"BR": 175, "FR": 176, "BL": 175, "FL": 174},
    "cw": {"BR": 179, "FR": 179, "BL": 172, "FL": 171},
    "ccw": {"BR": 176, "FR": 181, "BL": 172, "FL": 172},
    "left": {"BR": 174, "FR": 178, "BL": 171, "FL": 177},
    "right": {"BR": 176, "FR": 175, "BL": 176, "FL": 173},
}


def utc_iso(ts=None):
    return datetime.fromtimestamp(ts or time.time(), timezone.utc).isoformat()


class SerialEndpoint:
    def __init__(self, name, port, raw_writer, raw_lock):
        self.name = name
        self.port = port
        self.raw_writer = raw_writer
        self.raw_lock = raw_lock
        self.ser = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.lines = []

    def open(self):
        self.ser = serial.Serial(self.port, BAUD, timeout=0.05)
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

    def send(self, command):
        command = command.strip()
        if not command:
            return
        with self.lock:
            self.ser.write((command + "\n").encode("ascii"))

    def recent_lines(self, seconds):
        cutoff = time.time() - seconds
        return [(ts, line) for ts, line in list(self.lines) if ts >= cutoff]

    def _read_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                ser = self.ser
            if not ser:
                break
            try:
                raw = ser.readline()
            except Exception as exc:
                self._record(f"SERIAL_READ_ERROR,{exc}")
                break
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if line:
                self._record(line)

    def _record(self, line):
        ts = time.time()
        self.lines.append((ts, line))
        if len(self.lines) > 2000:
            self.lines = self.lines[-1000:]
        with self.raw_lock:
            self.raw_writer.writerow({
                "ts_iso": utc_iso(ts),
                "t_unix": f"{ts:.6f}",
                "source": self.name,
                "line": line,
            })


class RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read_key(self):
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch.lower() == "q":
                return "quit"
            key = ch.lower()
            if key in MODE_COMMANDS or key in DIRECT_MOVEMENT_KEYS:
                return key
            if ch == "\x1b":
                seq = ch + sys.stdin.read(2)
                if seq == "\x1b[A":
                    return "up"
                if seq == "\x1b[B":
                    return "down"
                if seq == "\x1b[C":
                    return "right"
                if seq == "\x1b[D":
                    return "left"


def parse_rpm(line):
    values = {label: float(value) for label, value in RPM_DETAIL_RE.findall(line)}
    if len(values) == 4:
        return {"format": "per_wheel", **values}
    if line.startswith("RPM,"):
        parts = line.split(",", 1)
        try:
            return {"format": "avg_only", "avg": float(parts[1])}
        except (IndexError, ValueError):
            return None
    return None


def parse_imu(line):
    if not line.startswith("IMU,"):
        return None
    parts = line.split(",")
    if len(parts) != len(IMU_LABELS) + 1:
        return None
    try:
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    return dict(zip(IMU_LABELS, values))


def angle_delta_deg(start, end):
    delta = end - start
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def trimmed_mean(values, trim_fraction=0.15):
    values = sorted(values)
    if not values:
        return 0.0
    cut = int(len(values) * trim_fraction)
    if cut and len(values) > cut * 2:
        values = values[cut:-cut]
    return statistics.mean(values)


def collect_imu_calibration(esp1, seconds=IMU_CALIBRATION_SECONDS):
    print(f"\nIMU calibration: keep robot completely still for {seconds:.0f} s")
    deadline = time.time() + seconds
    seen_keys = set()
    samples = []

    while time.time() < deadline:
        for ts, line in esp1.recent_lines(0.75):
            key = (ts, line)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            imu = parse_imu(line)
            if imu:
                samples.append((ts, imu))
        time.sleep(0.03)

    if len(samples) < IMU_CALIBRATION_MIN_SAMPLES:
        print(f"WARNING: only {len(samples)} IMU samples for calibration; correction will be weak.")

    axes = {label: [imu[label] for _ts, imu in samples] for label in IMU_LABELS}
    calibration = {
        "seconds": seconds,
        "samples": len(samples),
        "gyro_bias": {
            "gx": trimmed_mean(axes["gx"]) if samples else 0.0,
            "gy": trimmed_mean(axes["gy"]) if samples else 0.0,
            "gz": trimmed_mean(axes["gz"]) if samples else 0.0,
        },
        "accel_mean": {
            "ax": trimmed_mean(axes["ax"]) if samples else 0.0,
            "ay": trimmed_mean(axes["ay"]) if samples else 0.0,
            "az": trimmed_mean(axes["az"]) if samples else 0.0,
        },
        "yaw_reference": trimmed_mean(axes["yaw"]) if samples else 0.0,
        "gyro_pstdev": {
            "gx": statistics.pstdev(axes["gx"]) if len(samples) > 1 else None,
            "gy": statistics.pstdev(axes["gy"]) if len(samples) > 1 else None,
            "gz": statistics.pstdev(axes["gz"]) if len(samples) > 1 else None,
        },
    }
    print(
        "IMU calibration done: "
        f"n={calibration['samples']} "
        f"gz_bias={calibration['gyro_bias']['gz']:.4f} deg/s "
        f"yaw_ref={calibration['yaw_reference']:.2f} deg"
    )
    return calibration


def wait_for(predicate, timeout, description):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for {description}")


def has_line(endpoint, checker, seconds=10.0):
    for _ts, line in endpoint.recent_lines(seconds):
        if checker(line):
            return line
    return None


def send_logged(endpoint, command_writer, source, command, mode="", arrow="", movement=""):
    ts = time.time()
    endpoint.send(command)
    command_writer.writerow({
        "ts_iso": utc_iso(ts),
        "t_unix": f"{ts:.6f}",
        "source": source,
        "command": command,
        "mode": mode,
        "arrow": arrow,
        "movement": movement,
    })


def build_mode_command(mode_name):
    targets = SWERVE_MODE_TARGETS[mode_name]
    values = [targets[label] for label in WHEEL_LABELS]
    return "MODE,{},{:.2f},{:.2f},{:.2f},{:.2f}".format(mode_name, *values)


def collect_readiness(esp1, esp2, command_writer):
    send_logged(esp1, command_writer, "esp1", "PING")
    send_logged(esp2, command_writer, "esp2", "PING")
    send_logged(esp2, command_writer, "esp2", "SYS")

    status = {}
    status["esp1_seen"] = wait_for(
        lambda: has_line(
            esp1,
            lambda line: line.startswith(("PONG,", "STATUS,", "READY,", "RPM,", "RPM [", "TOF,", "IMU,")),
        ),
        READINESS_SECONDS,
        "ESP1 status or telemetry",
    )
    status["esp2_pong"] = wait_for(
        lambda: has_line(esp2, lambda line: line.startswith(("PONG,", "SYS,", "READY,"))),
        READINESS_SECONDS,
        "ESP2 PONG/SYS/READY",
    )
    status["esp1_rpm"] = wait_for(
        lambda: has_line(esp1, lambda line: line.startswith("RPM,") or line.startswith("RPM [")),
        READINESS_SECONDS,
        "ESP1 RPM telemetry",
    )


    status["esp1_tof"] = has_line(esp1, lambda line: line.startswith("TOF,"))
    status["esp1_imu"] = has_line(esp1, lambda line: line.startswith("IMU,"))
    status["esp1_per_wheel_rpm"] = has_line(esp1, lambda line: line.startswith("RPM ["))

    send_logged(esp2, command_writer, "esp2", build_mode_command("initial"), mode="initial")
    status["esp2_initial_ack"] = wait_for(
        lambda: has_line(esp2, lambda line: line.startswith("ACK,MODE,initial")),
        READINESS_SECONDS,
        "ESP2 ACK,MODE,initial",
    )
    send_logged(esp2, command_writer, "esp2", "P")
    status["esp2_pos"] = wait_for(
        lambda: has_line(esp2, lambda line: line.startswith("POS,")),
        READINESS_SECONDS,
        "ESP2 servo POS",
    )
    return status


def set_mode(mode_key, esp2, command_writer):
    mode = MODE_COMMANDS[mode_key]
    send_logged(esp2, command_writer, "esp2", build_mode_command(mode["name"]), mode=mode["name"])
    wait_for(
        lambda: has_line(esp2, lambda line: line.startswith(f"ACK,MODE,{mode['name']}")),
        READINESS_SECONDS,
        f"ESP2 ACK,MODE,{mode['name']}",
    )
    return mode["name"]


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
    return "{prefix}{br},{fr},{bl},{fl}".format(
        prefix=prefix,
        br=values[0],
        fr=values[1],
        bl=values[2],
        fl=values[3],
    )


def command_for_arrow(mode_key, arrow):
    mode_map = ARROW_COMMANDS.get(mode_key, {})
    if arrow not in mode_map:
        return None, None
    movement, prefix = mode_map[arrow]
    return movement, build_motor_command(prefix, movement)


def run_motion_pass(
    pass_id,
    mode_key,
    mode_name,
    arrow,
    movement,
    command,
    esp1,
    esp2,
    command_writer,
    rpm_writer,
    imu_writer,
    imu_calibration,
    imu_state,
):
    print(f"\nPass {pass_id} {mode_name} {arrow}/{movement}: sending {command} for {RUN_SECONDS:.1f} s")
    send_logged(esp2, command_writer, "esp2", command, mode=mode_name, arrow=arrow, movement=movement)
    start = time.time()
    seen_rpm_keys = set()
    seen_imu_keys = set()

    while time.time() - start < RUN_SECONDS:
        for ts, line in esp1.recent_lines(0.25):
            if ts < start:
                continue
            rpm = parse_rpm(line)
            if rpm:
                key = (ts, line)
                if key not in seen_rpm_keys:
                    seen_rpm_keys.add(key)
                    rpm_writer.writerow({
                        "ts_iso": utc_iso(ts),
                        "t_unix": f"{ts:.6f}",
                        "pass_id": pass_id,
                        "mode": mode_name,
                        "arrow": arrow,
                        "movement": movement,
                        "command": command,
                        "elapsed_pass_sec": f"{ts - start:.3f}",
                        "format": rpm["format"],
                        "avg": rpm.get("avg", ""),
                        "BR": rpm.get("BR", ""),
                        "FR": rpm.get("FR", ""),
                        "BL": rpm.get("BL", ""),
                        "FL": rpm.get("FL", ""),
                        "raw": line,
                    })
                continue

            imu = parse_imu(line)
            if imu:
                key = (ts, line)
                if key not in seen_imu_keys:
                    seen_imu_keys.add(key)
                    gx_corrected = imu["gx"] - imu_calibration["gyro_bias"]["gx"]
                    gy_corrected = imu["gy"] - imu_calibration["gyro_bias"]["gy"]
                    gz_corrected = imu["gz"] - imu_calibration["gyro_bias"]["gz"]
                    yaw_zeroed = angle_delta_deg(imu_calibration["yaw_reference"], imu["yaw"])
                    last_ts = imu_state.get("last_ts")
                    if last_ts is not None and ts > last_ts:
                        imu_state["yaw_gyro_integrated"] += gz_corrected * (ts - last_ts)
                    imu_state["last_ts"] = ts
                    imu_writer.writerow({
                        "ts_iso": utc_iso(ts),
                        "t_unix": f"{ts:.6f}",
                        "pass_id": pass_id,
                        "mode": mode_name,
                        "arrow": arrow,
                        "movement": movement,
                        "command": command,
                        "elapsed_pass_sec": f"{ts - start:.3f}",
                        **imu,
                        "gx_corrected": gx_corrected,
                        "gy_corrected": gy_corrected,
                        "gz_corrected": gz_corrected,
                        "yaw_zeroed": yaw_zeroed,
                        "yaw_gyro_integrated": imu_state["yaw_gyro_integrated"],
                        "raw": line,
                    })
        time.sleep(0.03)

    send_logged(esp2, command_writer, "esp2", "X", mode=mode_name, arrow=arrow, movement=movement)
    print(f"Pass {pass_id} {mode_name} {arrow}/{movement}: stopped")
    time.sleep(0.5)


def summarize(rpm_csv_path):
    rows = []
    with rpm_csv_path.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row["format"] == "per_wheel":
                rows.append(row)
    if not rows:
        return {
            "per_wheel_samples": 0,
            "note": "No per-wheel RPM rows found. Flash updated board01_final or run Robot/new_testing/encoder_rpm on Board 01.",
        }

    summary = {"per_wheel_samples": len(rows), "by_mode_movement_command": {}}
    groups = sorted({(row["mode"], row["movement"], row["command"]) for row in rows})
    for mode, movement, command in groups:
        subset = [
            row for row in rows
            if row["mode"] == mode and row["movement"] == movement and row["command"] == command
        ]
        key = f"{mode}:{movement}:{command}"
        means = {}
        for label in WHEEL_LABELS:
            values = [abs(float(row[label])) for row in subset if row[label] != ""]
            means[label] = statistics.mean(values) if values else None
        valid_means = [value for value in means.values() if value is not None]
        overall = statistics.mean(valid_means) if valid_means else None
        suggested_scale = {}
        for label, value in means.items():
            suggested_scale[label] = overall / value if value and overall else None
        summary["by_mode_movement_command"][key] = {
            "mode": mode,
            "movement": movement,
            "command": command,
            "samples": len(subset),
            "mean_abs_rpm": means,
            "overall_mean_abs_rpm": overall,
            "suggested_pwm_scale_for_next_test": suggested_scale,
        }
    return summary


def summarize_imu(imu_csv_path):
    rows = []
    with imu_csv_path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    summary = {"imu_samples": len(rows), "by_mode_movement_command": {}}
    groups = sorted({(row["mode"], row["movement"], row["command"]) for row in rows})
    for mode, movement, command in groups:
        subset = [
            row for row in rows
            if row["mode"] == mode and row["movement"] == movement and row["command"] == command
        ]
        key = f"{mode}:{movement}:{command}"
        means = {}
        stdevs = {}
        for label in (*IMU_LABELS, *IMU_CORRECTED_LABELS):
            values = [float(row[label]) for row in subset if row[label] != ""]
            means[label] = statistics.mean(values) if values else None
            stdevs[label] = statistics.pstdev(values) if len(values) > 1 else None
        summary["by_mode_movement_command"][key] = {
            "mode": mode,
            "movement": movement,
            "command": command,
            "samples": len(subset),
            "mean": means,
            "pstdev": stdevs,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="mode-aware 5-second wheel RPM balance logger")
    parser.add_argument("--esp1", default=DEFAULT_ESP1_PORT)
    parser.add_argument("--esp2", default=DEFAULT_ESP2_PORT)
    parser.add_argument("--max-passes", type=int, default=MAX_PASSES,
                        help="maximum motion passes before exiting; 0 means unlimited")
    args = parser.parse_args()

    if serial is None:
        raise SystemExit("pyserial is missing. Install it with: python3 -m pip install pyserial")

    base_dir = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw_serial.csv"
    command_path = run_dir / "commands.csv"
    rpm_path = run_dir / "rpm_samples.csv"
    imu_path = run_dir / "imu_samples.csv"
    summary_path = run_dir / "summary.json"

    print(f"Run folder: {run_dir}")
    print(f"Opening ESP1 sensors: {args.esp1}")
    print(f"Opening ESP2 actuators: {args.esp2}")

    with raw_path.open("w", newline="", encoding="utf-8") as raw_fp,\
            command_path.open("w", newline="", encoding="utf-8") as cmd_fp,\
            rpm_path.open("w", newline="", encoding="utf-8") as rpm_fp,\
            imu_path.open("w", newline="", encoding="utf-8") as imu_fp:
        raw_writer = csv.DictWriter(raw_fp, ["ts_iso", "t_unix", "source", "line"])
        command_writer = csv.DictWriter(cmd_fp, ["ts_iso", "t_unix", "source", "command", "mode", "arrow", "movement"])
        rpm_writer = csv.DictWriter(rpm_fp, [
            "ts_iso", "t_unix", "pass_id", "mode", "arrow", "movement", "command", "elapsed_pass_sec",
            "format", "avg", "BR", "FR", "BL", "FL", "raw",
        ])
        imu_writer = csv.DictWriter(imu_fp, [
            "ts_iso", "t_unix", "pass_id", "mode", "arrow", "movement", "command", "elapsed_pass_sec",
            *IMU_LABELS, *IMU_CORRECTED_LABELS, "raw",
        ])
        raw_writer.writeheader()
        command_writer.writeheader()
        rpm_writer.writeheader()
        imu_writer.writeheader()

        raw_lock = threading.Lock()
        esp1 = SerialEndpoint("esp1", args.esp1, raw_writer, raw_lock)
        esp2 = SerialEndpoint("esp2", args.esp2, raw_writer, raw_lock)
        imu_calibration = {
            "seconds": 0.0,
            "samples": 0,
            "gyro_bias": {"gx": 0.0, "gy": 0.0, "gz": 0.0},
            "accel_mean": {"ax": 0.0, "ay": 0.0, "az": 0.0},
            "yaw_reference": 0.0,
            "gyro_pstdev": {"gx": None, "gy": None, "gz": None},
        }
        imu_state = {"last_ts": None, "yaw_gyro_integrated": 0.0}
        try:
            esp1.open()
            esp2.open()
            readiness = collect_readiness(esp1, esp2, command_writer)
            print("\nReadiness OK")
            for key, value in readiness.items():
                print(f"  {key}: {value or 'not seen'}")
            if not readiness.get("esp1_per_wheel_rpm"):
                print("\nWARNING: ESP1 is not sending per-wheel RPM lines.")
                print("Balance summary needs: RPM [BR:xx  FR:xx  BL:xx  FL:xx]")
                print("Flash updated board01_final or use Robot/new_testing/encoder_rpm on Board 01.")
            if not readiness.get("esp1_tof") or not readiness.get("esp1_imu"):
                print("\nWARNING: TOF/IMU were not seen during readiness.")
                print("That is expected with encoder_rpm, but not with board01_final.")

            imu_calibration = collect_imu_calibration(esp1)
            current_mode_key = "i"
            current_mode_name = "initial"
            print("\nControls:")
            print("  i/m/t: switch swerve state")
            print(f"  fixed balanced base speed: {DEFAULT_SPEED} PWM")
            print("  f/b/l/r/c/v: run F/B/L/R/CW/CCW with balanced per-wheel PWM")
            print("  Arrow mode: I Up=F Down=B, M Left=CCW Right=CW, T Left=L Right=R")
            print("  q: quit")
            with RawTerminal() as terminal:
                pass_id = 1
                while args.max_passes <= 0 or pass_id <= args.max_passes:
                    print(f"\nMode {current_mode_name}. Press mode, direct movement, or arrow.")
                    key = terminal.read_key()
                    if key == "quit":
                        break
                    if key in MODE_COMMANDS:
                        current_mode_key = key
                        current_mode_name = set_mode(current_mode_key, esp2, command_writer)
                        print(f"Switched to {current_mode_name}")
                        continue
                    if key in DIRECT_MOVEMENT_KEYS:
                        target_mode_key, arrow = DIRECT_MOVEMENT_KEYS[key]
                        if target_mode_key != current_mode_key:
                            current_mode_key = target_mode_key
                            current_mode_name = set_mode(current_mode_key, esp2, command_writer)
                            print(f"Switched to {current_mode_name}")
                        key = arrow
                    movement, command = command_for_arrow(current_mode_key, key)
                    if command is None:
                        print(f"Arrow {key} is not a motor move in {current_mode_name} mode.")
                        continue
                    run_motion_pass(
                        pass_id,
                        current_mode_key,
                        current_mode_name,
                        key,
                        movement,
                        command,
                        esp1,
                        esp2,
                        command_writer,
                        rpm_writer,
                        imu_writer,
                        imu_calibration,
                        imu_state,
                    )
                    pass_id += 1
                if args.max_passes > 0 and pass_id > args.max_passes:
                    print(f"\nReached max passes: {args.max_passes}")
        except KeyboardInterrupt:
            print("\nInterrupted. Sending X stop.")
            send_logged(esp2, command_writer, "esp2", "X")
        finally:
            try:
                send_logged(esp2, command_writer, "esp2", "X")
            except Exception:
                pass
            esp1.close()
            esp2.close()

    summary = summarize(rpm_path)
    summary["imu"] = summarize_imu(imu_path)
    summary.update({
        "run_id": run_id,
        "esp1_port": args.esp1,
        "esp2_port": args.esp2,
        "run_seconds": RUN_SECONDS,
        "default_speed": DEFAULT_SPEED,
        "max_pwm": MAX_PWM,
        "base_balanced_pwm_targets": BASE_BALANCED_PWM_TARGETS,
        "t_left_front_extra_pwm_offset": T_LEFT_FRONT_EXTRA_PWM_OFFSET,
        "t_right_back_extra_pwm_offset": T_RIGHT_BACK_EXTRA_PWM_OFFSET,
        "i_forward_left_extra_pwm_offset": I_FORWARD_LEFT_EXTRA_PWM_OFFSET,
        "i_backward_right_extra_pwm_offset": I_BACKWARD_RIGHT_EXTRA_PWM_OFFSET,
        "max_passes": args.max_passes,
        "imu_calibration": imu_calibration,
        "swerve_mode_targets": SWERVE_MODE_TARGETS,
        "mode_arrow_mapping": ARROW_COMMANDS,
        "direct_movement_keys": DIRECT_MOVEMENT_KEYS,
        "raw_serial_csv": str(raw_path),
        "commands_csv": str(command_path),
        "rpm_samples_csv": str(rpm_path),
        "imu_samples_csv": str(imu_path),
    })
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nSaved: {raw_path}")
    print(f"Saved: {command_path}")
    print(f"Saved: {rpm_path}")
    print(f"Saved: {imu_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

