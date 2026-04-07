"""
Smoke-test both DC motors forward (BCM pins), e.g. L298N: IN1/IN2 + IN3/IN4.

Run on the Raspberry Pi (requires RPi.GPIO):

  python scripts/test_motor_gpio.py

Stops cleanly on Ctrl+C or errors.
"""
from __future__ import annotations

import time

try:
    import RPi.GPIO as GPIO
except ImportError as e:
    raise SystemExit(
        "RPi.GPIO is required. Install on the Pi: sudo apt install python3-rpi.gpio"
    ) from e

# BCM pins — left wheel (motor A), right wheel (motor B); match your L298N wiring
LEFT_IN1, LEFT_IN2 = 17, 27
RIGHT_IN1, RIGHT_IN2 = 22, 23

FORWARD_SEC = 5.0


def main() -> None:
    pins = (LEFT_IN1, LEFT_IN2, RIGHT_IN1, RIGHT_IN2)
    GPIO.setmode(GPIO.BCM)
    for p in pins:
        GPIO.setup(p, GPIO.OUT)

    def forward() -> None:
        GPIO.output(LEFT_IN1, 1)
        GPIO.output(LEFT_IN2, 0)
        GPIO.output(RIGHT_IN1, 1)
        GPIO.output(RIGHT_IN2, 0)

    def stop() -> None:
        for p in pins:
            GPIO.output(p, 0)

    try:
        forward()
        time.sleep(FORWARD_SEC)
        stop()
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
