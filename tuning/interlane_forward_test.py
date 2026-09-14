from __future__ import annotations

import contextlib
import gc
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

try:
    import serial
except ImportError:
    print("pyserial not found. Install it or activate nanosam-env.")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from applications import navigate_testbed as nav


nav.LOG_PATH = Path(__file__).resolve().parent / "logs" / "interlane_forward_test.log"

FORWARD_BASE_PWM = 100
APPROACH_MIN_PWM = 60
APPROACH_SLOWDOWN_START_PX = 300.0
ANGLE_PWM_DELTA_MIN = 8
ANGLE_PWM_DELTA_MAX = 40
DIST_PWM_DELTA_MIN = 8
DIST_PWM_DELTA_MAX = 35

TARGET_CENTER_DEADBAND_PX = 8.0
BACKWARD_SEARCH_AFTER_S = 3.0
BACKWARD_SEARCH_PWM = 70
OLD_CENTERLINE_LOSS_CONFIRM_S = 0.30


NEW_LANE_ENTRY_MARGIN_PX = 60.0


MAX_FORWARD_DURATION_S = 3.0

WINDOW_NAME = "INTERLANE FORWARD TEST"


def clamp_pwm(value):
    return max(0, min(255, int(round(value))))


def approach_base_pwm(dist_px):
    if dist_px >= APPROACH_SLOWDOWN_START_PX:
        return float(FORWARD_BASE_PWM)
    if dist_px <= TARGET_CENTER_DEADBAND_PX:
        return float(APPROACH_MIN_PWM)

    span = APPROACH_SLOWDOWN_START_PX - TARGET_CENTER_DEADBAND_PX
    frac = (dist_px - TARGET_CENTER_DEADBAND_PX) / span
    return APPROACH_MIN_PWM + ((FORWARD_BASE_PWM - APPROACH_MIN_PWM) *
                               max(0.0, min(1.0, frac)))


def effective_side_label(side_label, camera_side):


    if camera_side != "right" or side_label not in ("LEFT", "RIGHT"):
        return side_label
    return "RIGHT" if side_label == "LEFT" else "LEFT"


def forward_pwms(angle_deg, dist_px, side_label, camera_side, enable_slowdown):


    left_boost = 0.0
    right_boost = 0.0
    base_pwm = approach_base_pwm(dist_px) if enable_slowdown else float(FORWARD_BASE_PWM)
    correction_scale = base_pwm / FORWARD_BASE_PWM if FORWARD_BASE_PWM else 0.0

    rotation = nav.rotation_for_angle(angle_deg)
    angle_delta = nav.compute_pulse_ms(
        angle_deg,
        nav.ANGLE_DEADBAND_DEG,
        nav.ANGLE_ERROR_SATURATION_DEG,
        ANGLE_PWM_DELTA_MIN,
        ANGLE_PWM_DELTA_MAX,
    ) * correction_scale
    if rotation == "CW":
        left_boost += angle_delta
    elif rotation == "CCW":
        right_boost += angle_delta

    if dist_px > nav.CENTER_DEADBAND_PX:
        dist_delta = nav.compute_pulse_ms(
            dist_px,
            nav.CENTER_DEADBAND_PX,
            nav.CENTER_ERROR_SATURATION_PX,
            DIST_PWM_DELTA_MIN,
            DIST_PWM_DELTA_MAX,
        ) * correction_scale
        effective_label = effective_side_label(side_label, camera_side)
        if effective_label == "LEFT":
            right_boost += dist_delta
        elif effective_label == "RIGHT":
            left_boost += dist_delta

    br = clamp_pwm(base_pwm + right_boost)
    fr = clamp_pwm(base_pwm + right_boost)
    bl = clamp_pwm(base_pwm + left_boost)
    fl = clamp_pwm(base_pwm + left_boost)
    return f"F{br},{fr},{bl},{fl}", rotation, left_boost, right_boost, base_pwm


def new_lane_entry_ok(bottom_cx, camera_side):


    if camera_side == "right":
        return bottom_cx <= nav.IMAGE_CENTER_X - NEW_LANE_ENTRY_MARGIN_PX
    return bottom_cx >= nav.IMAGE_CENTER_X + NEW_LANE_ENTRY_MARGIN_PX


def send_ready_pose(ser2, servo_pos):
    nav.log_event("[init] Sending interlane ready pose...")
    for command, label in (
        ("I", "swerve initial / forward mode"),
        (f"5 {nav.SERVO_5_INITIAL}", "arm horizontal initial"),
        ("6 50", "arm vertical initial"),
        ("7 51", "front camera initial"),
        (f"8 {nav.CAMERA_LEFT_ANGLE}", "overhead camera left / lane search"),
        (f"9 {nav.SERVO_9_STRAIGHT_ANGLE}", "servo 9 straight"),
    ):
        nav.send_and_settle(ser2, servo_pos, command)
        nav.log_event(f"  ({label})")


def main():
    nav.log_event("=" * 60)
    nav.log_event(f"RUN START {datetime.now().isoformat(timespec='seconds')}")
    nav.log_event(f"Connecting to ESP2 (actuators) {nav.ESP2_PORT} at {nav.BAUD} baud...")

    ser2 = serial.Serial(nav.ESP2_PORT, nav.BAUD, timeout=0.1)
    time.sleep(0.5)
    ser2.reset_input_buffer()

    stop_event = threading.Event()
    servo_pos = {}
    threading.Thread(target=nav.serial_reader, args=(ser2, stop_event, servo_pos), daemon=True).start()

    send_ready_pose(ser2, servo_pos)

    processor = None
    cap = None
    phase = "search_initial"
    no_rails_since = None
    camera_side = "left"
    move_started_at = None
    movement_direction = "forward"
    travel_stage = "leaving_old"
    old_loss_since = None
    new_search_started_at = None
    continuous_forward_since = None
    align_pending_recheck = False

    def arm_next_move():
        nonlocal move_started_at, movement_direction, travel_stage, old_loss_since, new_search_started_at
        nonlocal continuous_forward_since, align_pending_recheck
        move_started_at = time.time()
        movement_direction = "forward"
        travel_stage = "leaving_old"
        old_loss_since = None
        new_search_started_at = None
        continuous_forward_since = None
        align_pending_recheck = False
        nav.log_event("[interlane] Leaving current lane at F100; old centerline corrections disabled.")

    try:
        processor = nav.LaneAlignmentProcessor(
            engine_path=nav.ENGINE_PATH,
            image_center_x=nav.IMAGE_CENTER_X,
        )
        cap = nav.open_overhead_camera()

        nav.log_event("=" * 60)
        nav.log_event(" INTERLANE FORWARD TEST")
        nav.log_event(" Servo 8 hunts left/right until the initial centerline is visible.")
        nav.log_event(" Press Enter while stopped on a centerline to drive to the next lane.")
        nav.log_event(" It ignores the old centerline until that line disappears.")
        nav.log_event(" It ramps down and aligns only after the next centerline appears.")
        nav.log_event(" If no new line appears within 3s after old-line loss, it reverses at PWM 70.")
        nav.log_event(f" New lane only accepted once it enters from the expected side "
                      f"({NEW_LANE_ENTRY_MARGIN_PX:.0f}px past center) — rejects old-line flicker.")
        nav.log_event(f" Hard cap: never drives F for more than {MAX_FORWARD_DURATION_S:.1f}s continuously; "
                      f"trips into a distance-only safety alignment if exceeded.")
        nav.log_event(" 'q' in the video window or Ctrl+C to quit.")
        nav.log_event("=" * 60)

        with nav.suppress_stderr():
            while True:
                ret, frame = cap.read()
                if not ret:
                    nav.log_event("[warn] frame grab failed, stopping.")
                    break

                reference_x = nav.reference_x_for_camera_side(camera_side)
                overlay, have_rails, angle_deg, dist_px, side_label, bottom_cx = processor.process_frame(
                    frame,
                    reference_x=reference_x,
                )

                now = time.time()
                motor_cmd = None
                rotation = nav.rotation_for_angle(angle_deg) if have_rails else "NO RAILS"
                base_pwm = 0.0

                if phase == "search_initial":
                    if have_rails:
                        no_rails_since = None
                    else:
                        if no_rails_since is None:
                            no_rails_since = now
                        elif now - no_rails_since >= nav.NO_RAILS_SWITCH_SECONDS:
                            camera_side = "right" if camera_side == "left" else "left"
                            angle = nav.CAMERA_RIGHT_ANGLE if camera_side == "right" else nav.CAMERA_LEFT_ANGLE
                            nav.send(ser2, "X")
                            nav.send_and_settle(ser2, servo_pos, f"8 {angle}")
                            no_rails_since = None
                            nav.log_event(f"[hunt] Initial lane not found; servo 8 -> {angle} ({camera_side}).")
                            continue

                entry_ok = have_rails and new_lane_entry_ok(bottom_cx, camera_side)

                if phase == "search_initial":
                    label = (f"INITIAL FOUND {camera_side} - press Enter"
                             if have_rails else f"SEARCHING INITIAL {camera_side}")
                elif phase == "moving_initial":
                    if travel_stage == "leaving_old":
                        label = "LEAVING OLD CENTERLINE"
                        if have_rails:
                            old_loss_since = None
                        else:
                            if old_loss_since is None:
                                old_loss_since = now
                            elif now - old_loss_since >= OLD_CENTERLINE_LOSS_CONFIRM_S:
                                travel_stage = "searching_new"
                                new_search_started_at = now
                                old_loss_since = None
                                nav.log_event("[interlane] Old centerline disappeared; searching for new centerline.")

                    elif travel_stage == "searching_new":
                        if entry_ok:
                            label = "SEARCHING NEW CENTERLINE"
                            travel_stage = "tracking_new"
                            movement_direction = "forward"
                            nav.log_event(f"[interlane] New centerline detected entering from the expected side "
                                          f"(bottom_x={bottom_cx:.0f}px, dist={dist_px:.1f}px, "
                                          f"angle={angle_deg:+.2f}deg); enabling slowdown and alignment.")
                        elif have_rails:


                            label = f"SEARCHING NEW CENTERLINE (rejected, wrong side x={bottom_cx:.0f}px)"
                        else:
                            label = "SEARCHING NEW CENTERLINE"
                        if not entry_ok and (movement_direction == "forward" and new_search_started_at is not None and
                              now - new_search_started_at >= BACKWARD_SEARCH_AFTER_S):
                            movement_direction = "backward"
                            travel_stage = "backtracking_new"
                            nav.log_event(f"[interlane] No new centerline within "
                                          f"{BACKWARD_SEARCH_AFTER_S:.1f}s after old-line loss; "
                                          f"reversing slowly at PWM {BACKWARD_SEARCH_PWM}.")

                    elif travel_stage == "tracking_new":
                        label = "TRACKING NEW CENTERLINE"
                        if have_rails and dist_px <= TARGET_CENTER_DEADBAND_PX:
                            nav.send(ser2, "X")
                            phase = "stopped"
                            label = "STOPPED ON NEXT"
                            nav.log_event(f"[interlane] New centerline aligned "
                                          f"(dist={dist_px:.1f}px, angle={angle_deg:+.2f}deg); stopped.")
                        elif not have_rails:
                            label = "TRACKING NEW CENTERLINE NO RAILS"

                    elif travel_stage == "backtracking_new":
                        label = "BACKTRACKING FOR NEW CENTERLINE"
                        if entry_ok and dist_px <= TARGET_CENTER_DEADBAND_PX:
                            nav.send(ser2, "X")
                            phase = "stopped"
                            label = "STOPPED ON NEXT"
                            nav.log_event(f"[interlane] New centerline found while backtracking "
                                          f"(dist={dist_px:.1f}px, angle={angle_deg:+.2f}deg); stopped.")
                        elif have_rails and not entry_ok:
                            label = f"BACKTRACKING (rejected, wrong side x={bottom_cx:.0f}px)"
                        elif not have_rails:
                            label = "BACKTRACKING NO RAILS"
                elif phase == "safety_align":
                    if not have_rails:
                        label = "SAFETY ALIGN: NO RAILS, WAITING"
                    elif dist_px <= TARGET_CENTER_DEADBAND_PX:
                        label = "SAFETY ALIGN: DISTANCE OK, CONFIRMING"
                    else:
                        label = f"SAFETY ALIGN: CENTERING (dist={dist_px:.0f}px)"
                elif phase == "stopped":
                    label = "STOPPED ON NEXT"
                else:
                    label = phase

                if phase == "moving_initial":
                    if movement_direction == "backward":
                        motor_cmd = f"B{BACKWARD_SEARCH_PWM}"
                        left_boost = right_boost = 0.0
                        base_pwm = float(BACKWARD_SEARCH_PWM)
                        continuous_forward_since = None
                        nav.send(ser2, motor_cmd)
                    else:
                        if travel_stage == "tracking_new" and have_rails:
                            motor_cmd, rotation, left_boost, right_boost, base_pwm = forward_pwms(
                                angle_deg,
                                dist_px,
                                side_label,
                                camera_side,
                                True,
                            )
                        else:
                            motor_cmd = f"F{FORWARD_BASE_PWM}"
                            left_boost = right_boost = 0.0
                            base_pwm = float(FORWARD_BASE_PWM)

                        if continuous_forward_since is None:
                            continuous_forward_since = now
                        elif now - continuous_forward_since >= MAX_FORWARD_DURATION_S:
                            nav.send(ser2, "X")
                            phase = "safety_align"
                            align_pending_recheck = False
                            continuous_forward_since = None
                            motor_cmd = "X"
                            label = "SAFETY ALIGN: FORWARD CAP TRIPPED"
                            nav.log_event(f"[safety] Drove F continuously for over "
                                          f"{MAX_FORWARD_DURATION_S:.1f}s (stage={travel_stage}) without "
                                          "reaching the target deadband; stopping and aligning distance "
                                          "against whatever centerline is currently visible.")
                        else:
                            nav.send(ser2, motor_cmd)

                elif phase == "safety_align":
                    left_boost = right_boost = 0.0
                    if not have_rails:
                        align_pending_recheck = False
                        motor_cmd = "X"
                    elif dist_px <= TARGET_CENTER_DEADBAND_PX:
                        nav.send(ser2, "X")
                        motor_cmd = "X"
                        if align_pending_recheck:
                            phase = "stopped"
                            label = "STOPPED ON NEXT"
                            align_pending_recheck = False
                            nav.log_event(f"[safety] Distance aligned after forward-cap trip "
                                          f"(dist={dist_px:.2f}px); stopped, waiting for Enter.")
                        else:
                            align_pending_recheck = True
                            time.sleep(nav.ALIGN_RECHECK_DELAY_S)
                    else:
                        align_pending_recheck = False
                        direction = nav.movement_for_distance(
                            dist_px, effective_side_label(side_label, camera_side))
                        pulse_ms = nav.compute_pulse_ms(
                            dist_px, TARGET_CENTER_DEADBAND_PX, nav.CENTER_ERROR_SATURATION_PX,
                            nav.CENTER_MIN_PULSE_MS, nav.CENTER_MAX_PULSE_MS)
                        motor_cmd = f"{direction}{nav.CENTER_MAX_SPEED}"
                        nav.send(ser2, motor_cmd)
                        time.sleep(pulse_ms / 1000.0)
                        nav.send(ser2, "X")
                    base_pwm = 0.0
                else:
                    left_boost = right_boost = 0.0

                color = (0, 255, 0) if phase == "stopped" else ((0, 165, 255) if have_rails else (0, 0, 255))
                cv2.putText(overlay, f"STATUS: {label}", (10, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(overlay, f"PHASE:{phase} CMD:{motor_cmd or '-'}",
                            (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(overlay, f"ROT:{rotation} BASE:{base_pwm:.0f} L+{left_boost:.0f} R+{right_boost:.0f}",
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                fwd_elapsed_hud = now - continuous_forward_since if continuous_forward_since else 0.0
                cv2.putText(overlay, f"ENTRY_OK:{entry_ok} BOTTOM_X:{bottom_cx:.0f}px "
                            f"FWD_S:{fwd_elapsed_hud:.1f}/{MAX_FORWARD_DURATION_S:.0f}",
                            (10, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow(WINDOW_NAME, overlay)

                forward_elapsed = now - continuous_forward_since if continuous_forward_since else 0.0
                nav.log_event(
                    f"[frame] phase={phase} status={label} rails={have_rails} "
                    f"angle={angle_deg:+.2f}deg dist={dist_px:.1f}px({side_label}) "
                    f"bottom_x={bottom_cx:.0f}px entry_ok={entry_ok} camera_side={camera_side} "
                    f"rotation={rotation} base={base_pwm:.1f} stage={travel_stage} "
                    f"slowdown={travel_stage == 'tracking_new'} fwd_s={forward_elapsed:.1f} "
                    f"cmd={motor_cmd} fps={processor.fps_s:.1f}"
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    nav.log_event("[info] stopped by user.")
                    break
                if key in nav.ENTER_KEY_CODES:
                    if phase == "search_initial" and have_rails:
                        phase = "moving_initial"
                        no_rails_since = None
                        arm_next_move()
                        nav.log_event(f"[interlane] Enter pressed; driving forward from initial centerline "
                                      f"(camera_side={camera_side}).")
                    elif phase == "stopped" and have_rails:
                        phase = "moving_initial"
                        arm_next_move()
                        nav.log_event("[interlane] Enter pressed after stop; driving to next lane.")
                    elif phase == "stopped":
                        nav.log_event("[interlane] Enter pressed after stop, but no rails are visible; holding.")

    except KeyboardInterrupt:
        nav.log_event("[info] stopped by user.")
    finally:
        with contextlib.suppress(Exception):
            nav.send(ser2, "X")
        stop_event.set()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        ser2.close()
        if processor:
            del processor.detector
        gc.collect()
        nav._CUDA_CONTEXT.pop()
        nav._CUDA_DEVICE.retain_primary_context().detach()
        nav.log_event("Cleanup complete.")
        nav.log_event(f"RUN END {datetime.now().isoformat(timespec='seconds')}")
        nav.log_event("=" * 60)


if __name__ == "__main__":
    main()
