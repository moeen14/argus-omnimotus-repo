from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

try:
    import cv2
except ImportError:
    cv2 = None


WINDOW_NAME = "ASABE Dual USB Camera Viewer"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "camera_snapshots"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
BOTTOM_CAMERA_DEVICE = "/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_9342D410-video-index0"
OVERHEAD_CAMERA_DEVICE = "/dev/v4l/by-id/usb-046d_0825_3D19ADE0-video-index0"
DEFAULT_CAMERA_SOURCES = [BOTTOM_CAMERA_DEVICE, OVERHEAD_CAMERA_DEVICE]
DEFAULT_CAMERA_LABELS = ["Bottom", "Overhead"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show two USB camera live feeds.")
    parser.add_argument(
        "--cams",
        nargs=2,
        metavar=("CAM_A", "CAM_B"),
        help="Two camera devices or numeric indices, for example /dev/video0 /dev/video2.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Requested capture FPS.")
    parser.add_argument("--list", action="store_true", help="Print detected video devices and exit.")
    return parser.parse_args()


def video_devices() -> list[str]:
    devices = sorted(Path("/dev").glob("video*"), key=lambda path: natural_video_key(path.name))
    return [str(path) for path in devices]


def natural_video_key(name: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return prefix, int(digits or 0)


def by_id_links() -> dict[str, list[str]]:
    links: dict[str, list[str]] = {}
    by_id = Path("/dev/v4l/by-id")
    if not by_id.exists():
        return links

    for link in sorted(by_id.iterdir()):
        try:
            target = str(link.resolve())
        except OSError:
            continue
        links.setdefault(target, []).append(str(link))
    return links


def camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def describe_devices() -> None:
    links = by_id_links()
    devices = video_devices()
    if not devices:
        print("No /dev/video* devices found.")
        return

    print("Detected video devices:")
    for device in devices:
        print(f"  {device}")
        for link in links.get(device, []):
            print(f"    by-id: {link}")


def configure_capture(cap: "cv2.VideoCapture", width: int, height: int, fps: int) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)


def open_capture(source: str, width: int, height: int, fps: int) -> "cv2.VideoCapture | None":
    cap = cv2.VideoCapture(camera_source(source))
    if not cap.isOpened():
        cap.release()
        return None

    configure_capture(cap, width, height, fps)
    ok, _frame = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def auto_select_cameras(width: int, height: int, fps: int) -> tuple[list[str], list["cv2.VideoCapture"]]:
    selected_sources: list[str] = []
    selected_caps: list["cv2.VideoCapture"] = []
    tried = set()

    for source in DEFAULT_CAMERA_SOURCES + video_devices():
        if source in tried:
            continue
        tried.add(source)
        cap = open_capture(source, width, height, fps)
        if cap is None:
            continue
        selected_sources.append(source)
        selected_caps.append(cap)
        if len(selected_caps) == 2:
            break

    return selected_sources, selected_caps


def draw_label(frame, label: str, status: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(frame, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, status, (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 240, 30), 2)


def resize_for_display(frame, height: int):
    if frame.shape[0] == height:
        return frame
    scale = height / frame.shape[0]
    width = max(1, int(frame.shape[1] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def save_snapshot(frame_a, frame_b, sources: list[str]) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOT_DIR / f"dual_usb_camera_{stamp}.jpg"
    combined = cv2.hconcat([frame_a, frame_b])
    cv2.imwrite(str(path), combined)
    meta_path = SNAPSHOT_DIR / f"dual_usb_camera_{stamp}.txt"
    meta_path.write_text(
        f"Bottom={sources[0]}\nOverhead={sources[1]}\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    args = parse_args()

    if cv2 is None:
        print("OpenCV is required. Install it with: python3 -m pip install opencv-python", file=sys.stderr)
        return 2

    describe_devices()
    if args.list:
        return 0

    if args.cams:
        sources = list(args.cams)
        caps = []
        for source in sources:
            cap = open_capture(source, args.width, args.height, args.fps)
            if cap is None:
                print(f"Could not open camera: {source}", file=sys.stderr)
                for opened in caps:
                    opened.release()
                return 1
            caps.append(cap)
    else:
        sources, caps = auto_select_cameras(args.width, args.height, args.fps)
        if len(caps) < 2:
            for cap in caps:
                cap.release()
            print("Could not auto-open two cameras. Try --cams /dev/videoX /dev/videoY.", file=sys.stderr)
            return 1

    print()
    print("Showing two feeds:")
    print(f"  {DEFAULT_CAMERA_LABELS[0]}: {sources[0]}")
    print(f"  {DEFAULT_CAMERA_LABELS[1]}: {sources[1]}")
    print("Keys: q/Esc quit, s save snapshot")

    last_status = ""
    try:
        while True:
            frames = []
            ok_all = True
            for cap in caps:
                ok, frame = cap.read()
                ok_all = ok_all and ok
                if not ok:
                    frame = blank_frame(args.width, args.height)
                frames.append(frame)

            display_h = min(frame.shape[0] for frame in frames)
            frame_a = resize_for_display(frames[0], display_h)
            frame_b = resize_for_display(frames[1], display_h)

            draw_label(frame_a, f"{DEFAULT_CAMERA_LABELS[0]}  {sources[0]}", "fixed USB")
            draw_label(frame_b, f"{DEFAULT_CAMERA_LABELS[1]}  {sources[1]}", "fixed USB")
            combined = cv2.hconcat([frame_a, frame_b])

            footer = "q/Esc quit | s snapshot | bottom left, overhead right"
            if last_status:
                footer = f"{last_status} | {footer}"
            cv2.putText(
                combined,
                footer,
                (12, combined.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

            cv2.imshow(WINDOW_NAME, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = save_snapshot(frame_a, frame_b, sources)
                last_status = f"saved {path.name}"
                print(last_status)

            if not ok_all:
                time.sleep(0.05)
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

    return 0


def blank_frame(width: int, height: int):
    frame = cv2.UMat(height, width, cv2.CV_8UC3).get()
    frame[:] = (30, 30, 30)
    cv2.putText(frame, "NO FRAME", (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
