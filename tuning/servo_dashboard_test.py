import os
import sys
import threading
import time
import queue
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

try:
    import serial
except ImportError:
    serial = None

ESP1_PORT = os.environ.get("ASABE_ESP1_PORT", "/dev/esp32_01")
ESP2_PORT = os.environ.get("ASABE_ESP2_PORT", "/dev/esp32_02")
BAUD_RATE = 921600
SERIAL_TIMEOUT = 0.08
POSITION_QUERY_RETRY_S = 2.0

STS_SERVO_NAMES = {
    5: "Arm horiz",
    6: "Arm vert",
    7: "Cam pan",
    8: "Overhead cam",
    9: "Servo 9 (new, no safe arc)",
}
STS_ANGLE_MIN, STS_ANGLE_MAX = 0.0, 360.0
SG90_ANGLE_MIN, SG90_ANGLE_MAX = 0.0, 180.0

MAROON = "#7A1730"
BG = "#f0f2f5"
CARD = "#ffffff"
CARD_BORDER = "#dde3eb"
TEXT = "#1a1a2e"
MUTED = "#8898aa"
GREEN = "#1fa34a"
RED = "#d84a4a"


class SerialLink:


    def __init__(self, name, port, inbox):
        self.name = name
        self.port = port
        self.inbox = inbox
        self.ser = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.status = "starting"

    def start(self):
        if serial is None:
            self.status = "pyserial missing"
            return
        threading.Thread(target=self._run, daemon=True).start()

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
                self.status = "offline"
                return False
            try:
                self.ser.write((command.strip() + "\n").encode("ascii"))
                return True
            except Exception as exc:
                self.status = f"write error: {exc}"
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
                return False

    def _run(self):
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    self.ser = serial.Serial(self.port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
                time.sleep(2)
                with self.lock:
                    if self.ser:
                        self.ser.reset_input_buffer()
                self.status = "live"

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
                        self.inbox.put((self.name, line))
            except Exception as exc:
                self.status = f"error: {exc}"
            with self.lock:
                self.ser = None
            if not self.stop_event.is_set():
                time.sleep(POSITION_QUERY_RETRY_S)


class ServoPanel:


    def __init__(self, parent, label, send_fn, angle_min, angle_max, note=None,
                 resolution=0.5, raw_buttons=None):


        self.send_fn = send_fn
        self.var = tk.DoubleVar(value=0.0)

        self.frame = tk.Frame(parent, bg=CARD, highlightbackground=CARD_BORDER,
                               highlightthickness=1, padx=10, pady=8)

        tk.Label(self.frame, text=label, bg=CARD, fg=TEXT,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        if note:
            tk.Label(self.frame, text=note, bg=CARD, fg=RED,
                     font=("Helvetica", 8)).pack(anchor="w")

        self.current_label = tk.Label(self.frame, text="current: --", bg=CARD,
                                       fg=MUTED, font=("Helvetica", 9))
        self.current_label.pack(anchor="w", pady=(2, 4))

        self.scale = tk.Scale(self.frame, from_=angle_min, to=angle_max,
                               resolution=resolution,
                               orient="horizontal",
                               variable=self.var, length=220, bg=CARD,
                               highlightthickness=0, troughcolor=BG,
                               fg=TEXT)
        self.scale.pack(fill="x")
        self.scale.bind("<ButtonRelease-1>", self._on_slider_release)

        entry_row = tk.Frame(self.frame, bg=CARD)
        entry_row.pack(fill="x", pady=(6, 0))
        self.entry = tk.Entry(entry_row, width=8)
        self.entry.pack(side="left")
        self.entry.bind("<Return>", self._on_entry_enter)
        tk.Button(entry_row, text="Set", command=self._on_entry_enter,
                  bg=MAROON, fg="white", relief="flat",
                  activebackground="#5c1224").pack(side="left", padx=(6, 0))

        if raw_buttons:
            raw_row = tk.Frame(self.frame, bg=CARD)
            raw_row.pack(fill="x", pady=(6, 0))
            for btn_label, raw_command in raw_buttons:
                tk.Button(raw_row, text=btn_label,
                          command=lambda cmd=raw_command: self.send_fn(raw=cmd),
                          bg=MUTED, fg="white", relief="flat").pack(side="left")

    def _on_slider_release(self, _event):
        self.send_fn(self.var.get())

    def _on_entry_enter(self, _event=None):
        text = self.entry.get().strip()
        if not text:
            return
        try:
            angle = float(text)
        except ValueError:
            return
        self.var.set(angle)
        self.send_fn(angle)

    def set_current(self, angle, source="reported"):


        self.var.set(angle)
        self.current_label.config(text=f"current: {angle:g} ({source})")


class ServoDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Servo Dashboard Test")
        self.root.configure(bg=BG)

        self.inbox = queue.Queue()
        self.board2 = SerialLink("board02", ESP2_PORT, self.inbox)
        self.board1 = SerialLink("board01", ESP1_PORT, self.inbox)

        self.sts_panels = {}
        self.sg90_panels = {}

        self._build_ui()

        self.board2.start()
        self.board1.start()

        self.root.after(300, self._initial_position_query)
        self.root.after(150, self._poll_inbox)
        self.root.after(500, self._poll_status)


    def _build_ui(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(header, text="Servo Dashboard Test", bg=BG, fg=MAROON,
                 font=("Helvetica", 16, "bold")).pack(side="left")

        self.status_label = tk.Label(header, text="board02: -- | board01: --",
                                      bg=BG, fg=MUTED, font=("Helvetica", 10))
        self.status_label.pack(side="right")

        refresh_btn = tk.Button(header, text="Refresh positions",
                                 command=self._initial_position_query,
                                 bg=MAROON, fg="white", relief="flat",
                                 activebackground="#5c1224")
        refresh_btn.pack(side="right", padx=(0, 12))

        sts_label = tk.Label(self.root, text="STS bus servos (Board 02) — \"<id> <angle>\"",
                              bg=BG, fg=TEXT, font=("Helvetica", 12, "bold"))
        sts_label.pack(anchor="w", padx=14, pady=(6, 2))

        sts_grid = tk.Frame(self.root, bg=BG)
        sts_grid.pack(padx=10, pady=4)
        sts_ids = range(5, 10)
        for grid_idx, i in enumerate(sts_ids):
            note = "no safe arc yet — accepts any angle" if i in (8, 9) else None
            panel = ServoPanel(
                sts_grid, f"Servo {i} — {STS_SERVO_NAMES[i]}",
                lambda angle, sid=i: self._send_sts(sid, angle),
                STS_ANGLE_MIN, STS_ANGLE_MAX, note=note,
            )
            row, col = divmod(grid_idx, 3)
            panel.frame.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            self.sts_panels[i] = panel

        sg90_label = tk.Label(self.root, text="SG90 servos — no position feedback, starts at 0°",
                               bg=BG, fg=TEXT, font=("Helvetica", 12, "bold"))
        sg90_label.pack(anchor="w", padx=14, pady=(12, 2))

        sg90_grid = tk.Frame(self.root, bg=BG)
        sg90_grid.pack(padx=10, pady=4, fill="x")


        dispenser = ServoPanel(
            sg90_grid, "SG90 — Dispenser (Board 02) — \"S<angle>\"",
            self._send_sg90_dispenser, SG90_ANGLE_MIN, SG90_ANGLE_MAX,
            resolution=1,
            raw_buttons=[("Dispense (D) — full auto cycle", "D")],
        )
        dispenser.frame.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        self.sg90_panels["dispenser"] = dispenser

        push_plant = ServoPanel(
            sg90_grid, "SG90 — Push plant (Board 01) — \"SG90,<angle>\"",
            self._send_sg90_push_plant, SG90_ANGLE_MIN, SG90_ANGLE_MAX,
            resolution=1,
        )
        push_plant.frame.grid(row=0, column=1, padx=6, pady=6, sticky="nsew")
        self.sg90_panels["push_plant"] = push_plant

        log_label = tk.Label(self.root, text="Live serial log (both boards, all lines)",
                              bg=BG, fg=TEXT, font=("Helvetica", 12, "bold"))
        log_label.pack(anchor="w", padx=14, pady=(12, 2))
        self.log_text = scrolledtext.ScrolledText(
            self.root, height=8, bg="#0d1117", fg="#d0d0d0",
            insertbackground="#d0d0d0", font=("Courier", 9), wrap="none",
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_text.configure(state="disabled")


    def _send_sts(self, sid, angle):
        self.board2.send(f"{sid} {angle:g}")
        self.sts_panels[sid].set_current(angle, source="sent")

    def _send_sg90_dispenser(self, angle=None, raw=None):
        if raw is not None:
            self.board2.send(raw)
            return
        angle = int(round(angle))
        self.board2.send(f"S{angle}")
        self.sg90_panels["dispenser"].set_current(angle, source="sent")

    def _send_sg90_push_plant(self, angle):
        angle = int(round(angle))
        self.board1.send(f"SG90,{angle}")
        self.sg90_panels["push_plant"].set_current(angle, source="sent")


    def _initial_position_query(self):
        self.board2.send("P")


    def _poll_inbox(self):
        try:
            while True:
                name, line = self.inbox.get_nowait()
                self._handle_line(name, line)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_inbox)

    def _handle_line(self, name, line):
        self._log(f"[{name}] {line}")
        if name == "board02" and line.startswith("POS,"):
            values = line.split(",")[1:]
            for sid, value in enumerate(values, start=1):
                if sid not in self.sts_panels:
                    continue
                if value == "ERR":
                    continue
                try:
                    angle = float(value)
                except ValueError:
                    continue
                self.sts_panels[sid].set_current(angle, source="board")

    def _log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")

        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 500:
            self.log_text.delete("1.0", f"{line_count - 500}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _poll_status(self):
        self.status_label.config(
            text=f"board02 ({ESP2_PORT}): {self.board2.status} | "
                 f"board01 ({ESP1_PORT}): {self.board1.status}"
        )
        self.root.after(500, self._poll_status)

    def shutdown(self):
        self.board2.stop()
        self.board1.stop()


def main():
    if serial is None:
        print("WARNING: pyserial not installed — commands will not be sent, UI-only.")
    root = tk.Tk()
    app = ServoDashboard(root)

    def on_close():
        app.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
