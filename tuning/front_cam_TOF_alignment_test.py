from __future__ import annotations

import os
import threading
import time

try:
    import serial
except ImportError:
    serial = None

ESP1_PORT = os.environ.get("ASABE_ESP1_PORT", "/dev/esp32_01")
ESP2_PORT = os.environ.get("ASABE_ESP2_PORT", "/dev/esp32_02")
BAUD = 921600

FRONT_TOF_LABEL = "CAM"
TOF_LABELS = ["FRS", "BRS", "BLS", "FLF", "FLS", "CAM", "FRF"]

FRONT_TOF_TARGET_MM = 110.0


FRONT_TOF_DEADBAND_MM = 2.0


FRONT_TOF_MAX_SPEED = 120
FRONT_TOF_MIN_PULSE_MS = 5
FRONT_TOF_MAX_PULSE_MS = 30
FRONT_TOF_ERROR_SATURATION_MM = 200.0
LOOP_SLEEP_S = 0.05


ALIGN_RECHECK_DELAY_S = 1.0


def send(ser, cmd):
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


def compute_pulse_ms(error_mag, deadband, saturation, min_pulse_ms, max_pulse_ms):

    mag = min(abs(error_mag), saturation)
    if mag <= deadband:
        return 0.0
    span = saturation - deadband
    frac = (mag - deadband) / span
    return min_pulse_ms + frac * (max_pulse_ms - min_pulse_ms)


def esp1_reader(ser, stop_event, tof):


    while not stop_event.is_set():
        try:
            raw = ser.readline()
        except Exception:
            break
        if not raw:
            continue
        line = raw.decode(errors="ignore").strip()
        if not line.startswith("TOF,["):
            continue
        body = line[line.find("[") + 1:line.rfind("]")]
        for item in body.split(","):
            if ":" not in item:
                continue
            label, value = item.split(":", 1)
            label = label.strip()
            if label in tof:
                try:
                    tof[label] = float(value.strip())
                except ValueError:
                    pass


def serial_reader(ser, stop_event):

    while not stop_event.is_set():
        try:
            ser.readline()
        except Exception:
            break


def align_once(ser2, tof):


    pending_recheck = False
    while True:
        reading = tof.get(FRONT_TOF_LABEL)
        if reading is None:
            print("\r  waiting for ToF data...", end="", flush=True)
            time.sleep(LOOP_SLEEP_S)
            continue

        error = reading - FRONT_TOF_TARGET_MM
        pulse_ms = compute_pulse_ms(
            error, FRONT_TOF_DEADBAND_MM, FRONT_TOF_ERROR_SATURATION_MM,
            FRONT_TOF_MIN_PULSE_MS, FRONT_TOF_MAX_PULSE_MS)

        if pulse_ms <= 0:
            send(ser2, "X")
            if pending_recheck:
                print(f"\r  Aligned (CAM={reading:.0f}mm).                          ")
                return
            pending_recheck = True
            print(f"\r  In band (CAM={reading:.0f}mm) — confirming...    ", end="", flush=True)
            time.sleep(ALIGN_RECHECK_DELAY_S)
        else:
            pending_recheck = False
            direction = "F" if error > 0 else "B"
            print(f"\r  CAM={reading:.0f}mm -> {direction}{FRONT_TOF_MAX_SPEED} "
                  f"({pulse_ms:.0f}ms)            ", end="", flush=True)
            send(ser2, f"{direction}{FRONT_TOF_MAX_SPEED}")
            time.sleep(pulse_ms / 1000.0)
            send(ser2, "X")
            time.sleep(LOOP_SLEEP_S)


def main():
    if serial is None:
        print("pyserial not found. Install it or activate nanosam-env.")
        return 1

    print(f"Connecting to ESP1 (sensors) {ESP1_PORT} and ESP2 (actuators) {ESP2_PORT} at {BAUD} baud...")
    ser1 = serial.Serial(ESP1_PORT, BAUD, timeout=0.1)
    ser2 = serial.Serial(ESP2_PORT, BAUD, timeout=0.1)
    time.sleep(2.0)
    ser1.reset_input_buffer()
    ser2.reset_input_buffer()
    reset_esp1_tofs(ser1)

    tof = {label: None for label in TOF_LABELS}
    stop_event = threading.Event()
    threading.Thread(target=esp1_reader, args=(ser1, stop_event, tof), daemon=True).start()
    threading.Thread(target=serial_reader, args=(ser2, stop_event), daemon=True).start()

    send(ser2, "X")
    send(ser2, "I")
    time.sleep(0.2)

    print("=" * 60)
    print(" Front-camera ToF alignment test (CAM channel, no vision)")
    print(f" target={FRONT_TOF_TARGET_MM:.0f}mm  deadband=+/-{FRONT_TOF_DEADBAND_MM:.0f}mm")
    print(" Press Enter to align, Enter again to re-align, 'q' + Enter to quit.")
    print("=" * 60)

    try:
        while True:
            choice = input("\nPress Enter to align (or 'q' to quit): ").strip().lower()
            if choice == "q":
                break
            align_once(ser2, tof)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        send(ser2, "X")
        stop_event.set()
        ser1.close()
        ser2.close()
        print("Cleanup complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
