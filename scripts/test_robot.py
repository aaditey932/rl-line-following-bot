"""
Full robot hardware test for the line-following bot on Raspberry Pi.

Tests (in order):
  1. Motor forward  — both wheels drive forward briefly
  2. Motor backward — both wheels reverse briefly
  3. Left spin      — left back / right forward (spin in place)
  4. Right spin     — left forward / right back (spin in place)
  5. IR sensors     — stream digital readings for a few seconds

Wiring (L298N, BCM numbering):
  Left motor:   IN1=GPIO17 (Pin 11), IN2=GPIO27 (Pin 13)
  Right motor:  IN3=GPIO22 (Pin 15), IN4=GPIO23 (Pin 16)
  GND → Pi GND (Pin 6)
  ENA/ENB jumpers left in place (always-on, full speed)

IR sensors (digital output modules):
  Left IR:  GPIO 5  (LOW = on line / black)
  Right IR: GPIO 6  (LOW = on line / black)

Run on the Pi:
  python scripts/test_robot.py

Optional flags:
  --no-motors       skip motor tests (safe if wheels are off the ground)
  --no-ir           skip IR sensor test
  --drive-sec N     seconds to run each motor phase  (default 1.0)
  --ir-sec N        seconds to stream IR readings    (default 4.0)
  --ir-left-pin P   BCM pin for left IR sensor       (default 5)
  --ir-right-pin P  BCM pin for right IR sensor      (default 6)
  --pwm             use software PWM at reduced duty instead of full-on
  --speed DUTY      PWM duty cycle 0-100              (default 60)
"""
from __future__ import annotations

import argparse
import sys
import time

# ---------------------------------------------------------------------------
# GPIO import
# ---------------------------------------------------------------------------
try:
    import RPi.GPIO as GPIO
except ImportError as exc:
    raise SystemExit(
        "RPi.GPIO not found. On the Pi: sudo apt install python3-rpi.gpio"
    ) from exc

# ---------------------------------------------------------------------------
# Pin map (BCM) — actual robot wiring
# ---------------------------------------------------------------------------
LEFT_IN1, LEFT_IN2 = 17, 27
RIGHT_IN1, RIGHT_IN2 = 22, 23
# Software-PWM enable pins (only used when --pwm flag is passed)
# ENA/ENB jumpers are normally left in place for full speed
LEFT_ENA = 12
RIGHT_ENB = 13

DEFAULT_IR_LEFT_PIN = 5
DEFAULT_IR_RIGHT_PIN = 6

DRIVE_SEC = 1.0
PAUSE_SEC = 0.4
IR_SEC = 4.0
PWM_FREQ = 1000  # Hz


# ---------------------------------------------------------------------------
# Motor helpers
# ---------------------------------------------------------------------------

class Motors:
    def __init__(self, use_pwm: bool, duty: float) -> None:
        self._pins = (LEFT_IN1, LEFT_IN2, RIGHT_IN1, RIGHT_IN2)
        self._use_pwm = use_pwm
        self._duty = max(0.0, min(100.0, duty))
        self._pwm_left: GPIO.PWM | None = None
        self._pwm_right: GPIO.PWM | None = None

        GPIO.setmode(GPIO.BCM)
        for p in self._pins:
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, 0)

        if use_pwm:
            GPIO.setup(LEFT_ENA, GPIO.OUT)
            GPIO.setup(RIGHT_ENB, GPIO.OUT)
            self._pwm_left = GPIO.PWM(LEFT_ENA, PWM_FREQ)
            self._pwm_right = GPIO.PWM(RIGHT_ENB, PWM_FREQ)
            self._pwm_left.start(0)
            self._pwm_right.start(0)

    def _set_wheel(self, in1: int, in2: int, direction: int) -> None:
        """direction: +1 forward, -1 backward, 0 stop."""
        if direction > 0:
            GPIO.output(in1, 1)
            GPIO.output(in2, 0)
        elif direction < 0:
            GPIO.output(in1, 0)
            GPIO.output(in2, 1)
        else:
            GPIO.output(in1, 0)
            GPIO.output(in2, 0)

    def drive(self, left: int, right: int) -> None:
        """Set both wheels. left/right: +1 forward, -1 backward, 0 stop."""
        self._set_wheel(LEFT_IN1, LEFT_IN2, left)
        self._set_wheel(RIGHT_IN1, RIGHT_IN2, right)
        if self._use_pwm and self._pwm_left and self._pwm_right:
            lduty = self._duty if left != 0 else 0.0
            rduty = self._duty if right != 0 else 0.0
            self._pwm_left.ChangeDutyCycle(lduty)
            self._pwm_right.ChangeDutyCycle(rduty)

    def stop(self) -> None:
        self.drive(0, 0)
        if self._use_pwm and self._pwm_left and self._pwm_right:
            self._pwm_left.ChangeDutyCycle(0)
            self._pwm_right.ChangeDutyCycle(0)

    def cleanup(self) -> None:
        self.stop()
        if self._pwm_left:
            self._pwm_left.stop()
        if self._pwm_right:
            self._pwm_right.stop()


# ---------------------------------------------------------------------------
# IR helpers
# ---------------------------------------------------------------------------

def setup_ir(left_pin: int, right_pin: int) -> tuple[int, int]:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(left_pin, GPIO.IN)
    GPIO.setup(right_pin, GPIO.IN)
    return left_pin, right_pin


def read_ir(left_pin: int, right_pin: int) -> tuple[int, int]:
    """Return (left, right) digital readings. Most IR modules: 0 = on line."""
    return GPIO.input(left_pin), GPIO.input(right_pin)


# ---------------------------------------------------------------------------
# Test routines
# ---------------------------------------------------------------------------

def test_motors(args: argparse.Namespace) -> None:
    print("\n=== Motor test ===")
    m = Motors(use_pwm=args.pwm, duty=args.speed)
    phases = [
        ("Forward",    +1, +1),
        ("Backward",   -1, -1),
        ("Spin left",  -1, +1),
        ("Spin right", +1, -1),
    ]
    try:
        for label, left, right in phases:
            print(f"  {label} ({args.drive_sec:.1f} s) ... ", end="", flush=True)
            m.drive(left, right)
            time.sleep(args.drive_sec)
            m.stop()
            time.sleep(PAUSE_SEC)
            print("done")
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        m.cleanup()
    print("Motor test complete.")


def test_ir(args: argparse.Namespace) -> None:
    print(f"\n=== IR sensor test ({args.ir_sec:.1f} s) ===")
    print(f"  Pins: left={args.ir_left_pin} right={args.ir_right_pin}")
    print("  Place sensor over line and off line to verify readings.")
    print("  (Most modules: 0 = on line / black, 1 = off line / white)\n")

    left_pin, right_pin = setup_ir(args.ir_left_pin, args.ir_right_pin)
    t0 = time.monotonic()
    try:
        prev = (-1, -1)
        while time.monotonic() - t0 < args.ir_sec:
            lv, rv = read_ir(left_pin, right_pin)
            if (lv, rv) != prev:
                label_l = "ON-LINE" if lv == 0 else "off    "
                label_r = "ON-LINE" if rv == 0 else "off    "
                print(f"  left={lv} ({label_l})   right={rv} ({label_r})")
                prev = (lv, rv)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    print("IR test complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robot hardware smoke-test (motors + IR sensors)")
    p.add_argument("--no-motors", action="store_true", help="Skip motor tests")
    p.add_argument("--no-ir", action="store_true", help="Skip IR sensor test")
    p.add_argument("--drive-sec", type=float, default=DRIVE_SEC,
                   help=f"Seconds per motor phase (default {DRIVE_SEC})")
    p.add_argument("--ir-sec", type=float, default=IR_SEC,
                   help=f"Seconds to stream IR readings (default {IR_SEC})")
    p.add_argument("--ir-left-pin", type=int, default=DEFAULT_IR_LEFT_PIN,
                   help=f"BCM pin for left IR sensor (default {DEFAULT_IR_LEFT_PIN})")
    p.add_argument("--ir-right-pin", type=int, default=DEFAULT_IR_RIGHT_PIN,
                   help=f"BCM pin for right IR sensor (default {DEFAULT_IR_RIGHT_PIN})")
    p.add_argument("--pwm", action="store_true",
                   help="Use software PWM on ENA/ENB pins instead of full-on")
    p.add_argument("--speed", type=float, default=60.0,
                   help="PWM duty cycle 0-100 when --pwm is set (default 60)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_motors and args.no_ir:
        print("Nothing to test (--no-motors and --no-ir both set).")
        sys.exit(0)

    try:
        if not args.no_motors:
            test_motors(args)
        if not args.no_ir:
            test_ir(args)
    finally:
        GPIO.cleanup()

    print("\nAll tests done.")


if __name__ == "__main__":
    main()
