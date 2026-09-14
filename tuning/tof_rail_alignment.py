from __future__ import annotations

import argparse
import math
import re
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass

try:
    import serial
except ImportError:
    serial = None


BAUD = 921600
DEFAULT_ESP1_PORT = "/dev/esp32_01"
DEFAULT_ESP2_PORT = "/dev/esp32_02"

MOTOR_PWM = 175
ROTATION_COMMAND_PWM = 85
TRANSLATION_COMMAND_PWM = 85
MIN_EFFECTIVE_PWM = 85

SAMPLE_SECONDS = 0.85
ROTATION_CONTROL_SAMPLE_SECONDS = 0.18
ROTATION_FINE_SAMPLE_SECONDS = 0.08
ROTATION_MAX_SECONDS = 6.0
ROTATION_SETTLE_SECONDS = 0.03
ROTATION_SETTLE_READ_SECONDS = 0.12
TRANSLATION_CONTROL_SAMPLE_SECONDS = 0.18
TRANSLATION_FINE_SAMPLE_SECONDS = 0.08
TRANSLATION_MAX_SECONDS = 6.0
TRANSLATION_SETTLE_SECONDS = 0.03
TRANSLATION_SETTLE_READ_SECONDS = 0.12
DIRECTION_CHANGE_GAP_SECONDS = 0.05
COMMAND_REFRESH_SECONDS = 0.03
WORSENED_ERROR_MARGIN_MM = 3.0
MAX_WORSENED_STEPS = 2
ALIGNMENT_CYCLES = 1
MAX_ROTATION_STEPS = 18
MAX_DISTANCE_STEPS = 18

ROTATION_PAIR_TOLERANCE_MM = 2.0
DISTANCE_AVERAGE_TOLERANCE_MM = 2.0
DISTANCE_OVERSHOOT_SETTLE_SECONDS = 0.35
FINE_ROTATION_ERROR_MM = 8.0
FINE_TRANSLATION_ERROR_MM = 8.0
MIN_VALID_MM = 20
MAX_VALID_MM = 2500
TOF_RE = re.compile(r"([A-Z]+):(-?\d+(?:\.\d+)?)")

WHEEL_LABELS = ("BR", "FR", "BL", "FL")
T_LEFT_FRONT_EXTRA_PWM_OFFSET = 6
T_RIGHT_BACK_EXTRA_PWM_OFFSET = 8
I_FORWARD_LEFT_EXTRA_PWM_OFFSET = -5
I_BACKWARD_RIGHT_EXTRA_PWM_OFFSET = -10
SWERVE_MODE_TARGETS = {
    "initial": {"BR": 37.0, "FR": 6.0, "BL": 145.0, "FL": 221.0},
    "mid": {"BR": 306.0, "FR": 96.0, "BL": 235.0, "FL": 131.0},
    "perpendicular": {"BR": 220.0, "FR": 180.0, "BL": 325.0, "FL": 42.0},
}
BASE_BALANCED_PWM_TARGETS = {
    "forward": {"BR": 175, "FR": 178, "BL": 174, "FL": 172},
    "backward": {"BR": 175, "FR": 176, "BL": 175, "FL": 174},
    "left": {"BR": 174, "FR": 178, "BL": 171, "FL": 177},
    "right": {"BR": 176, "FR": 175, "BL": 176, "FL": 173},
}
MOTOR_MOVEMENTS = {
    "F": "forward",
    "B": "backward",
    "L": "left",
    "R": "right",
}
TOF_LABELS = ("FRS", "BRS", "BLS", "FLF", "FLS", "CAM", "FRF")


@dataclass(frozen=True)
class AlignmentSide:
    name: str
    pair: tuple[str, str]
    mode_for_distance: str
    closer_command: str
    farther_command: str
    first_farther_rotation: str
    second_farther_rotation: str


SIDES = {
    "l": AlignmentSide(
        name="left side",
        pair=("FLS", "BLS"),
        mode_for_distance="perpendicular",
        closer_command="L",
        farther_command="R",
        first_farther_rotation="CCW",
        second_farther_rotation="CW",
    ),
    "r": AlignmentSide(
        name="right side",
        pair=("FRS", "BRS"),
        mode_for_distance="perpendicular",
        closer_command="R",
        farther_command="L",
        first_farther_rotation="CW",
        second_farther_rotation="CCW",
    ),
    "f": AlignmentSide(
        name="front",
        pair=("FLF", "FRF"),
        mode_for_distance="initial",
        closer_command="F",
        farther_command="B",
        first_farther_rotation="CW",
        second_farther_rotation="CCW",
    ),
}


def build_mode_command(mode_name: str) -> str:
    targets = SWERVE_MODE_TARGETS[mode_name]
    values = [targets[label] for label in WHEEL_LABELS]
    return "MODE,{},{:.2f},{:.2f},{:.2f},{:.2f}".format(mode_name, *values)


def balanced_pwm_targets(movement: str) -> dict[str, int]:
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
    return pwms


def build_balanced_motor_command(prefix: str, movement: str, pwm: int) -> str:
    scale = pwm / MOTOR_PWM
    targets = balanced_pwm_targets(movement)
    values = [
        max(MIN_EFFECTIVE_PWM, min(255, round(targets[label] * scale)))
        for label in WHEEL_LABELS
    ]
    return f"{prefix}{values[0]},{values[1]},{values[2]},{values[3]}"


def filtered_value(values: list[float]) -> float | None:
    clean = [value for value in values if MIN_VALID_MM <= value <= MAX_VALID_MM]
    if not clean:
        return None
    if len(clean) < 4:
        return statistics.median(clean)

    med = statistics.median(clean)
    deviations = [abs(value - med) for value in clean]
    mad = statistics.median(deviations)
    limit = max(18.0, 3.0 * mad)
    filtered = [value for value in clean if abs(value - med) <= limit]
    if not filtered:
        filtered = clean
    if len(filtered) >= 5:
        filtered = sorted(filtered)
        cut = max(1, len(filtered) // 5)
        if len(filtered) > cut * 2:
            filtered = filtered[cut:-cut]
    return statistics.fmean(filtered)


class SerialEndpoint:
    def __init__(self, name: str, port: str, timeout: float = 0.04):
        self.name = name
        self.port = port
        self.timeout = timeout
        self.ser = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.lines = deque(maxlen=2500)

    def open(self) -> None:
        self.ser = serial.Serial(self.port, BAUD, timeout=self.timeout)
        time.sleep(1.8)
        self.ser.reset_input_buffer()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def close(self) -> None:
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

    def send(self, command: str) -> None:
        command = command.strip()
        if not command:
            return
        with self.lock:
            if not self.ser:
                raise RuntimeError(f"{self.name} serial is not open")
            self.ser.write((command + "\n").encode("ascii"))
        print(f"-> {self.name}: {command}")

    def recent_lines(self, seconds: float) -> list[tuple[float, str]]:
        cutoff = time.time() - seconds
        return [(ts, line) for ts, line in list(self.lines) if ts >= cutoff]

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                ser = self.ser
            if not ser:
                break
            try:
                raw = ser.readline()
            except Exception as exc:
                self.lines.append((time.time(), f"SERIAL_READ_ERROR,{exc}"))
                break
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if line:
                self.lines.append((time.time(), line))


TOF_RESET_RETRY_DELAY_S = 2.0


def reset_esp1_tofs(esp1: "SerialEndpoint") -> None:


    attempt = 0
    while True:
        attempt += 1
        print(f"[init] Resetting Board 01 ToF sensors (attempt {attempt})...")
        t0 = time.time()
        esp1.send("F")

        deadline = time.time() + 6.0
        ok = total = None
        while time.time() < deadline and ok is None:
            for ts, line in list(esp1.lines):
                if ts < t0 or not line.startswith("ACK,TOF_RESET,DONE"):
                    continue
                parts = line.split(",")[-1].split("/")
                if len(parts) == 2:
                    try:
                        ok, total = int(parts[0]), int(parts[1])
                    except ValueError:
                        ok = total = None
                break
            if ok is None:
                time.sleep(0.05)

        if ok is not None and ok == total:
            print(f"[init] Board 01 ToF reset OK — {ok}/{total} sensors reporting. Safe to proceed.")
            esp1.lines.clear()
            return

        seen = f"{ok}/{total}" if ok is not None else "no ACK reply seen"
        print(f"[init] Board 01 ToF reset INCOMPLETE ({seen}) — will NOT proceed until every "
              f"sensor resets successfully. Retrying in {TOF_RESET_RETRY_DELAY_S:.1f}s...")
        time.sleep(TOF_RESET_RETRY_DELAY_S)


class TofRailAligner:
    def __init__(self, esp1: SerialEndpoint, esp2: SerialEndpoint, target_mm: float):
        self.esp1 = esp1
        self.esp2 = esp2
        self.target_mm = target_mm
        self.last_tof_packet_count = 0

    def send_mode(self, mode_name: str) -> None:
        self.esp2.send(build_mode_command(mode_name))
        time.sleep(0.35)

    def stop(self) -> None:
        self.esp2.send("X")

    def collect_tof_samples(
        self,
        start: float,
        samples: dict[str, list[float]],
        seen: set[tuple[float, str]],
    ) -> int:
        packet_count = 0
        for ts, line in self.esp1.recent_lines(SAMPLE_SECONDS + 0.25):
            if ts < start or (ts, line) in seen or not line.startswith("TOF,["):
                continue
            seen.add((ts, line))
            packet_count += 1
            for label, value in TOF_RE.findall(line):
                if label in samples:
                    try:
                        samples[label].append(float(value))
                    except ValueError:
                        pass
        return packet_count

    def filtered_samples(self, samples: dict[str, list[float]]) -> dict[str, float | None]:
        return {label: filtered_value(values) for label, values in samples.items()}

    def drive_and_read_tof(self, command: str, seconds: float) -> dict[str, float | None]:
        start = time.time()
        deadline = start + seconds
        samples = {label: [] for label in TOF_LABELS}
        seen: set[tuple[float, str]] = set()
        packet_count = 0

        while time.time() < deadline:
            self.esp2.send(command)
            packet_count += self.collect_tof_samples(start, samples, seen)
            time.sleep(COMMAND_REFRESH_SECONDS)

        packet_count += self.collect_tof_samples(start, samples, seen)
        self.last_tof_packet_count = packet_count
        return self.filtered_samples(samples)

    def drive_rotation_and_read_tof(self, command: str, seconds: float) -> dict[str, float | None]:
        return self.drive_and_read_tof(command, seconds)

    def drive_translation_and_read_tof(self, command: str, seconds: float) -> dict[str, float | None]:
        return self.drive_and_read_tof(command, seconds)

    def read_filtered_tof(self, seconds: float = SAMPLE_SECONDS) -> dict[str, float | None]:
        start = time.time()
        deadline = time.time() + seconds
        samples = {label: [] for label in TOF_LABELS}
        seen: set[tuple[float, str]] = set()
        packet_count = 0

        while time.time() < deadline:
            packet_count += self.collect_tof_samples(start, samples, seen)
            time.sleep(0.025)

        self.last_tof_packet_count = packet_count
        return self.filtered_samples(samples)

    def print_tof(self) -> dict[str, float | None]:
        readings = self.read_filtered_tof()
        print()
        print(f"Filtered ToF readings (mm), fresh packets: {getattr(self, 'last_tof_packet_count', 0)}")
        for label in TOF_LABELS:
            value = readings.get(label)
            text = "----" if value is None else f"{value:7.1f}"
            print(f"  {label}: {text}")
        print()
        return readings

    def pair_error(self, side: AlignmentSide, readings: dict[str, float | None]) -> float | None:
        first, second = side.pair
        a = readings.get(first)
        b = readings.get(second)
        if a is None or b is None:
            return None
        return a - b

    def side_distance(self, side: AlignmentSide, readings: dict[str, float | None]) -> float | None:
        values = [readings.get(label) for label in side.pair]
        valid = [value for value in values if value is not None]
        if len(valid) != 2:
            return None
        return statistics.fmean(valid)

    def side_status(
        self,
        side: AlignmentSide,
        readings: dict[str, float | None],
    ) -> dict[str, float] | None:
        first_label, second_label = side.pair
        first = readings.get(first_label)
        second = readings.get(second_label)
        if first is None or second is None:
            return None
        first_error = first - self.target_mm
        second_error = second - self.target_mm
        avg_distance = statistics.fmean([first, second])
        avg_error = avg_distance - self.target_mm
        pair_error = first - second
        max_endpoint_error = max(abs(first_error), abs(second_error))
        return {
            "first": first,
            "second": second,
            "first_error": first_error,
            "second_error": second_error,
            "avg_distance": avg_distance,
            "avg_error": avg_error,
            "pair_error": pair_error,
            "max_endpoint_error": max_endpoint_error,
        }

    def distance_aligned(self, status: dict[str, float]) -> bool:
        return abs(status["avg_error"]) <= DISTANCE_AVERAGE_TOLERANCE_MM

    def rotate_command(self, side: AlignmentSide, error: float, pwm: int) -> str:
        direction = (
            side.first_farther_rotation
            if error > 0
            else side.second_farther_rotation
        )
        return f"{direction}{pwm}"

    def rotation_control_seconds(self, error: float) -> float:
        if abs(error) <= FINE_ROTATION_ERROR_MM:
            return ROTATION_FINE_SAMPLE_SECONDS
        return ROTATION_CONTROL_SAMPLE_SECONDS

    def translation_control_seconds(self, avg_error: float) -> float:
        if abs(avg_error) <= FINE_TRANSLATION_ERROR_MM:
            return TRANSLATION_FINE_SAMPLE_SECONDS
        return TRANSLATION_CONTROL_SAMPLE_SECONDS

    def align_rotation(self, side_key: str) -> bool:
        side = SIDES[side_key]
        self.send_mode("mid")
        previous_abs = None
        started = time.time()
        step = 1

        print(f"Rotational alignment: {side.name} using {side.pair[0]} vs {side.pair[1]}")
        readings = self.read_filtered_tof()
        while time.time() - started < ROTATION_MAX_SECONDS and step <= MAX_ROTATION_STEPS:
            error = self.pair_error(side, readings)
            if error is None:
                print(f"  missing ToF pair for {side.name}; cannot rotate-align")
                return False

            print(f"  rotate step {step:02d}: pair error {error:+.1f} mm")
            if abs(error) <= ROTATION_PAIR_TOLERANCE_MM:
                self.stop()
                print("  rotation aligned")
                return True

            command = self.rotate_command(side, error, ROTATION_COMMAND_PWM)
            self.drive_rotation_and_read_tof(
                command,
                self.rotation_control_seconds(error),
            )
            self.stop()
            time.sleep(ROTATION_SETTLE_SECONDS)
            readings = self.read_filtered_tof(ROTATION_SETTLE_READ_SECONDS)

            new_error = self.pair_error(side, readings)
            if new_error is None:
                self.stop()
                return False
            if previous_abs is None:
                previous_abs = abs(error)
            if abs(new_error) > abs(error) + WORSENED_ERROR_MARGIN_MM and abs(new_error) > previous_abs:
                self.stop()
                time.sleep(DIRECTION_CHANGE_GAP_SECONDS)
                print("  correction worsened; stopping rotation pass")
                return False
            previous_abs = abs(new_error)
            step += 1

        self.stop()

        print("  rotation alignment reached max steps")
        return False

    def distance_command(self, side: AlignmentSide, distance_error: float, pwm: int) -> str:

        prefix = side.closer_command if distance_error > 0 else side.farther_command
        return build_balanced_motor_command(prefix, MOTOR_MOVEMENTS[prefix], pwm)

    def align_distance(self, side_key: str) -> bool:
        side = SIDES[side_key]
        self.send_mode(side.mode_for_distance)
        previous_abs = None
        worsened_steps = 0
        started = time.time()
        step = 1
        previous_command = None

        print(f"Distance alignment: {side.name}, target {self.target_mm:.1f} mm")
        status = self.side_status(side, self.read_filtered_tof())
        while time.time() - started < TRANSLATION_MAX_SECONDS and step <= MAX_DISTANCE_STEPS:
            if status is None:
                print(f"  missing ToF pair for {side.name}; cannot distance-align")
                return False

            print(
                f"  distance step {step:02d}: "
                f"{side.pair[0]} error {status['first_error']:+.1f} mm, "
                f"{side.pair[1]} error {status['second_error']:+.1f} mm, "
                f"pair {status['pair_error']:+.1f} mm, "
                f"avg error {status['avg_error']:+.1f} mm"
            )
            if self.distance_aligned(status):
                self.stop()
                print("  average distance aligned")
                return True

            command = self.distance_command(side, status["avg_error"], TRANSLATION_COMMAND_PWM)
            if previous_command and command[:1] != previous_command[:1]:
                self.stop()
                time.sleep(DIRECTION_CHANGE_GAP_SECONDS)
            readings = self.drive_translation_and_read_tof(
                command,
                self.translation_control_seconds(status["avg_error"]),
            )
            self.stop()
            time.sleep(TRANSLATION_SETTLE_SECONDS)
            previous_command = command

            quick_status = self.side_status(side, readings)
            new_status = self.side_status(
                side,
                self.read_filtered_tof(TRANSLATION_SETTLE_READ_SECONDS),
            )
            if new_status is None:
                self.stop()
                return False
            if quick_status and status["avg_error"] * quick_status["avg_error"] < 0:
                self.stop()
                print("  quick average crossed target; stopping to avoid overshoot")
                if self.distance_aligned(new_status):
                    print("  average distance aligned after overshoot stop")
                    return True
                return False
            if status["avg_error"] * new_status["avg_error"] < 0:
                self.stop()
                print("  average crossed target; stopping to avoid overshoot")
                time.sleep(DISTANCE_OVERSHOOT_SETTLE_SECONDS)
                settled = self.side_status(
                    side,
                    self.read_filtered_tof(TRANSLATION_SETTLE_READ_SECONDS),
                )
                if settled and self.distance_aligned(settled):
                    print("  average distance aligned after overshoot stop")
                    return True
                return False
            if previous_abs is None:
                previous_abs = abs(status["avg_error"])
            if (
                abs(new_status["avg_error"])
                > abs(status["avg_error"]) + WORSENED_ERROR_MARGIN_MM
                and abs(new_status["avg_error"]) > previous_abs
            ):
                worsened_steps += 1
                print("  correction worsened")
                if worsened_steps >= MAX_WORSENED_STEPS:
                    self.stop()
                    return False
            else:
                worsened_steps = 0
            previous_abs = abs(new_status["avg_error"])
            status = new_status
            step += 1

        self.stop()

        print("  distance alignment reached max steps")
        return False

    def verify_alignment(self, side_key: str) -> bool:
        side = SIDES[side_key]
        status = self.side_status(side, self.read_filtered_tof())

        print(f"Final verification: {side.name}")
        if status is None:
            print("  NOT ALIGNED: missing ToF pair")
            return False

        rotation_ok = abs(status["pair_error"]) <= ROTATION_PAIR_TOLERANCE_MM
        distance_ok = abs(status["avg_error"]) <= DISTANCE_AVERAGE_TOLERANCE_MM
        print(
            f"  pair error {status['pair_error']:+.1f} mm "
            f"(limit +/-{ROTATION_PAIR_TOLERANCE_MM:.1f})"
        )
        print(
            f"  {side.pair[0]} {status['first']:.1f} mm, "
            f"error {status['first_error']:+.1f} mm"
        )
        print(
            f"  {side.pair[1]} {status['second']:.1f} mm, "
            f"error {status['second_error']:+.1f} mm"
        )
        print(
            f"  average {status['avg_distance']:.1f} mm, "
            f"avg error {status['avg_error']:+.1f} mm "
            f"(limit +/-{DISTANCE_AVERAGE_TOLERANCE_MM:.1f})"
        )
        if rotation_ok and distance_ok:
            print("  ALIGNED")
            return True

        print("  NOT ALIGNED")
        return False

    def align_side(self, side_key: str) -> None:
        if side_key not in SIDES:
            raise ValueError(f"unknown side: {side_key}")


        aligned = False
        for cycle in range(1, ALIGNMENT_CYCLES + 1):
            print(f"\nSide alignment cycle {cycle}/{ALIGNMENT_CYCLES}")
            if not self.align_rotation(side_key):
                print("Rotation did not align; skipping distance for this cycle.")
                continue
            self.align_distance(side_key)
            if self.verify_alignment(side_key):
                aligned = True
                break

        if not aligned:
            print("Side alignment finished without meeting both final checks.")
        self.stop()
        self.print_tof()

    def align_corner(self, corner: str) -> None:
        side_key = "l" if corner == "fl" else "r"
        print(f"Corner alignment: {corner.upper()} using {SIDES[side_key].name} + front")


        for cycle in range(1, ALIGNMENT_CYCLES + 1):
            print(f"\nCorner cycle {cycle}/{ALIGNMENT_CYCLES}")
            side_ok = self.align_rotation(side_key)
            front_ok = self.align_rotation("f")
            if not (side_ok and front_ok):
                print("Rotation did not align; skipping distance for this corner cycle.")
                continue
            self.align_distance(side_key)
            self.align_distance("f")

        self.stop()
        self.print_tof()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align robot to rails using Board 01 ToF sensors.")
    parser.add_argument("--esp1", default=DEFAULT_ESP1_PORT, help="Board 01 sensor serial port.")
    parser.add_argument("--esp2", default=DEFAULT_ESP2_PORT, help="Board 02 actuator serial port.")
    parser.add_argument("--target", type=float, default=220.0, help="Target rail distance in mm.")
    return parser.parse_args()


def print_help(target_mm: float) -> None:
    print()
    print("Commands:")
    print("  d          show filtered ToF readings")
    print("  <number>   set target rail distance in mm")
    print("  l/r/f      align left side / right side / front")
    print("  fl/fr      align front-left / front-right corner")
    print("  stop/x     send motor stop")
    print("  q          quit")
    print(f"Current target distance: {target_mm:.1f} mm")
    print()


def main() -> int:
    args = parse_args()
    if serial is None:
        raise SystemExit("pyserial is missing. Install it with: python3 -m pip install pyserial")

    esp1 = SerialEndpoint("esp1", args.esp1)
    esp2 = SerialEndpoint("esp2", args.esp2)
    aligner = TofRailAligner(esp1, esp2, args.target)

    print(f"Opening ESP1 sensors: {args.esp1}")
    print(f"Opening ESP2 actuators: {args.esp2}")
    esp1.open()
    esp2.open()

    try:
        esp1.send("PING")
        reset_esp1_tofs(esp1)
        esp2.send("PING")
        print_help(aligner.target_mm)
        aligner.print_tof()

        while True:
            raw = input("align> ").strip().lower()
            if not raw:
                continue
            if raw in {"q", "quit", "exit"}:
                break
            if raw in {"h", "help", "?"}:
                print_help(aligner.target_mm)
                continue
            if raw in {"d", "tof", "read"}:
                aligner.print_tof()
                continue
            if raw in {"x", "stop"}:
                aligner.stop()
                continue
            try:
                aligner.target_mm = float(raw)
                print(f"Target distance set to {aligner.target_mm:.1f} mm")
                continue
            except ValueError:
                pass

            if raw in {"l", "r", "f"}:
                aligner.align_side(raw)
            elif raw in {"fl", "fr"}:
                aligner.align_corner(raw)
            else:
                print(f"Unknown command: {raw!r}. Type h for help.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            aligner.stop()
        except Exception:
            pass
        esp1.close()
        esp2.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
