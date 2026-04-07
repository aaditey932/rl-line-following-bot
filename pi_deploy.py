"""
Raspberry Pi deployment script — matched to the Hack Lab kit hardware.

Hardware (from the Hack Lab PDF)
---------------------------------
  Chassis  : Standard 2-wheel acrylic kit (~20 x 15 cm)
  Motors   : 2x TT DC geared motors (FA-130 style, yellow gearbox)
               Wheel diameter ≈ 65 mm  (radius 0.033 m)
               No encoders on the standard kit
  H-bridge : L298N with ENA/ENB jumpers hardwired HIGH
               Speed is controlled by software-PWM on the IN pins
               (exactly how gpiozero.Robot works)
  Wiring from the lab PDF:
               Left  motor  → IN1 = GPIO 4,  IN2 = GPIO 14
               Right motor  → IN3 = GPIO 17, IN4 = GPIO 18
               (Note: gpiozero.Robot(left=(4,14), right=(17,18)) )
  IR sensors : 5x TCRT5000 or similar reflectance modules
               Digital output (comparator) → GPIO pins 5,6,13,19,26
               OR analogue via MCP3008 SPI ADC channels 0–4
               Switch with --ir-mode digital|adc
  Power      : 9V battery → L298N  |  USB-C power bank → Pi

Usage
-----
  # Install (on Pi):
  pip install gpiozero RPi.GPIO spidev stable-baselines3 torch numpy

  # Test without real hardware (mock sensors):
  python3 pi_deploy.py --model models/ppo_mujoco.zip --dry-run

  # Run with digital IR sensors (no ADC needed):
  python3 pi_deploy.py --model models/ppo_mujoco.zip --ir-mode digital

  # Run with analogue IR sensors via MCP3008 SPI ADC:
  python3 pi_deploy.py --model models/ppo_mujoco.zip --ir-mode adc

  # Calibrate ADC IR sensors first:
  python3 pi_deploy.py --model models/ppo_mujoco.zip --calibrate

Sim2Real note
-------------
The observation is built identically to training:
  obs = [ir0..ir4,  omega_left_norm, omega_right_norm,  cmd_left, cmd_right]

Because the kit has no encoders, wheel speed is *estimated* from the
commanded duty cycle through the same first-order motor model used in
training (pwm_first_order).  This keeps the observation consistent with
what the policy expects without needing physical encoders.

Control loop runs at CONTROL_HZ (50 Hz by default).
"""
from __future__ import annotations

import argparse
import time
import signal
import sys
from pathlib import Path

import numpy as np

# ── Hardware constants ────────────────────────────────────────────────────────
CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ

# Match the value used during training (EnvConfig.wheel_vel_max)
WHEEL_VEL_MAX = 18.0          # rad/s  (top wheel speed at full duty)

N_IR = 5                      # number of IR sensors

# Default GPIO pin assignments (BCM numbering) — from the Hack Lab PDF
# L298N wiring: left=(IN1,IN2), right=(IN3,IN4)  — ENA/ENB jumpers are ON
LEFT_IN1  = 4
LEFT_IN2  = 14
RIGHT_IN1 = 17
RIGHT_IN2 = 18

# Digital IR sensor GPIO pins (BCM) — one per sensor, outermost to innermost
# Adjust to match how you've wired your IR array
IR_DIGITAL_PINS = [5, 6, 13, 19, 26]   # GPIO for ir0..ir4

# MCP3008 SPI ADC (for analogue IR sensors)
SPI_BUS    = 0
SPI_DEVICE = 0
IR_ADC_CHANNELS = list(range(N_IR))    # ADC channels 0-4

# Motor dynamics (must match training Sim2RealConfig)
MOTOR_DEADBAND    = 0.05   # duty fraction below which motor stalls
MOTOR_SLEW_PER_S  = 8.0    # max duty change per second
MOTOR_TAU_S       = 0.08   # first-order lag time constant (s)
ACTION_DELAY_STEPS = 1     # steps of deliberate delay (match training)

# IR ADC normalisation (override with --calibrate)
IR_ADC_BLACK = 50.0        # raw ADC reading over black tape  (0-1023)
IR_ADC_WHITE = 950.0       # raw ADC reading over white floor (0-1023)
IR_GAMMA     = 0.93        # photodiode gamma (match training)


# ── Mock classes for --dry-run testing ───────────────────────────────────────

class MockRobot:
    """gpiozero.Robot substitute that prints instead of moving."""
    def __init__(self, *a, **kw): pass
    def value(self, left=0.0, right=0.0):
        pass  # gpiozero Robot does not have a .value() setter this way
    def stop(self): pass
    def close(self): pass

    def _set(self, left: float, right: float):
        pass   # no-op in dry-run


class GpioZeroRobotWrapper:
    """Thin wrapper around gpiozero.Robot that accepts continuous duty [-1,1]."""

    def __init__(self, left_pins: tuple, right_pins: tuple, dry_run: bool = False):
        self._dry_run = dry_run
        if dry_run:
            self._robot = None
        else:
            from gpiozero import Robot  # type: ignore
            self._robot = Robot(left=left_pins, right=right_pins)

    def set_duty(self, left: float, right: float) -> None:
        """Set wheel duties.  left/right in [-1, 1]; positive = forward."""
        left  = float(np.clip(left,  -1.0, 1.0))
        right = float(np.clip(right, -1.0, 1.0))
        if self._dry_run or self._robot is None:
            return
        # gpiozero Robot accepts value as (left_speed, right_speed) in [-1,1]
        self._robot.value = (left, right)

    def stop(self) -> None:
        if self._robot is not None:
            self._robot.stop()

    def close(self) -> None:
        if self._robot is not None:
            self._robot.close()


class MockSPI:
    def open(self, bus, device): pass
    def xfer2(self, data): return [0, 1, 200]   # ≈ mid-scale 456
    def close(self): pass


class MockGPIO:
    BCM = "BCM"; IN = "IN"; PUD_UP = "PUD_UP"
    @staticmethod
    def setmode(*a): pass
    @staticmethod
    def setup(*a, **kw): pass
    @staticmethod
    def input(pin): return 0   # always "black" in dry-run
    @staticmethod
    def cleanup(): pass


# ── IR sensor readers ─────────────────────────────────────────────────────────

class DigitalIRArray:
    """Read N digital IR sensor outputs (0 = black, 1 = white on most modules)."""

    def __init__(self, pins: list[int], dry_run: bool = False):
        self._pins = pins
        self._dry_run = dry_run
        if dry_run:
            self._GPIO = MockGPIO()
        else:
            import RPi.GPIO as GPIO  # type: ignore
            GPIO.setmode(GPIO.BCM)
            for p in pins:
                GPIO.setup(p, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            self._GPIO = GPIO

    def read(self) -> np.ndarray:
        """Returns float32 array of [0,1]: 0.0 = black, 1.0 = white."""
        vals = np.array([float(self._GPIO.input(p)) for p in self._pins], dtype=np.float32)
        return vals

    def cleanup(self):
        if not self._dry_run:
            self._GPIO.cleanup()


class MCP3008IRArray:
    """Read N analogue IR sensors through MCP3008 10-bit SPI ADC."""

    def __init__(
        self,
        channels: list[int],
        spi_bus: int = 0,
        spi_device: int = 0,
        ir_black: float = IR_ADC_BLACK,
        ir_white: float = IR_ADC_WHITE,
        ir_gamma: float = IR_GAMMA,
        dry_run: bool = False,
    ):
        self._channels = channels
        self._black = ir_black
        self._white = ir_white
        self._gamma = max(ir_gamma, 1e-6)
        if dry_run:
            self._spi = MockSPI()
        else:
            import spidev  # type: ignore
            self._spi = spidev.SpiDev()
            self._spi.open(spi_bus, spi_device)
            self._spi.max_speed_hz = 1_000_000
            self._spi.mode = 0

    def _read_raw(self, channel: int) -> int:
        assert 0 <= channel <= 7
        r = self._spi.xfer2([1, (8 + channel) << 4, 0])
        return ((r[1] & 3) << 8) | r[2]

    def read(self) -> np.ndarray:
        """Returns float32 [0,1] per channel after normalisation and gamma."""
        raw = np.array([self._read_raw(c) for c in self._channels], dtype=np.float64)
        denom = max(self._white - self._black, 1.0)
        u = np.clip((raw - self._black) / denom, 0.0, 1.0)
        u = np.power(u, self._gamma)
        return u.astype(np.float32)

    def close(self):
        self._spi.close()


def calibrate_adc(
    channels: list[int] = IR_ADC_CHANNELS,
    spi_bus: int = SPI_BUS,
    spi_device: int = SPI_DEVICE,
) -> None:
    """Interactive ADC calibration — prints raw readings for black/white."""
    import spidev  # type: ignore
    spi = spidev.SpiDev()
    spi.open(spi_bus, spi_device)
    spi.max_speed_hz = 1_000_000
    spi.mode = 0

    def read_all():
        vals = []
        for ch in channels:
            r = spi.xfer2([1, (8 + ch) << 4, 0])
            vals.append(((r[1] & 3) << 8) | r[2])
        return np.array(vals)

    print("=== IR ADC Calibration ===")
    input("Place ALL sensors over WHITE floor, then press Enter...")
    white = read_all()
    print(f"  White readings: {white}  (mean: {white.mean():.0f})")

    input("Place ALL sensors over BLACK tape, then press Enter...")
    black = read_all()
    print(f"  Black readings: {black}  (mean: {black.mean():.0f})")

    spi.close()
    print("\nAdd to your run command:")
    print(f"  --ir-black {black.mean():.0f} --ir-white {white.mean():.0f}")


# ── Main controller ───────────────────────────────────────────────────────────

class PiLineFollower:
    """50 Hz control loop: IR sensors → PPO inference → L298N motors."""

    def __init__(
        self,
        model_path: str,
        ir_mode: str = "digital",       # "digital" or "adc"
        dry_run: bool = False,
        # GPIO wiring (BCM) — from the Hack Lab PDF
        left_pins:  tuple = (LEFT_IN1,  LEFT_IN2),
        right_pins: tuple = (RIGHT_IN1, RIGHT_IN2),
        ir_digital_pins: list = IR_DIGITAL_PINS,
        ir_adc_channels: list = IR_ADC_CHANNELS,
        ir_adc_black: float = IR_ADC_BLACK,
        ir_adc_white: float = IR_ADC_WHITE,
        # Training-aligned parameters
        wheel_vel_max: float = WHEEL_VEL_MAX,
        motor_deadband: float = MOTOR_DEADBAND,
        motor_slew_per_s: float = MOTOR_SLEW_PER_S,
        motor_tau_s: float = MOTOR_TAU_S,
        action_delay_steps: int = ACTION_DELAY_STEPS,
        ir_gamma: float = IR_GAMMA,
        verbose: bool = False,
    ):
        self._verbose = verbose
        self._wheel_vel_max = wheel_vel_max
        self._motor_deadband = motor_deadband
        self._motor_slew = motor_slew_per_s * DT
        self._motor_tau = motor_tau_s
        self._action_delay = action_delay_steps
        self._action_history: list[np.ndarray] = []

        # Motor state (estimated, since no encoders)
        self._u_left  = 0.0    # current slew-limited duty
        self._u_right = 0.0
        self._omega_left  = 0.0   # estimated wheel speed (rad/s)
        self._omega_right = 0.0
        self._last_cmd = np.zeros(2, dtype=np.float32)

        # Motors via gpiozero.Robot (handles software PWM on IN pins)
        self._motors = GpioZeroRobotWrapper(
            left_pins=left_pins, right_pins=right_pins, dry_run=dry_run
        )

        # IR sensors
        if ir_mode == "digital":
            self._ir = DigitalIRArray(ir_digital_pins, dry_run=dry_run)
            self._ir_mode = "digital"
        else:
            self._ir = MCP3008IRArray(
                ir_adc_channels,
                ir_black=ir_adc_black,
                ir_white=ir_adc_white,
                ir_gamma=ir_gamma,
                dry_run=dry_run,
            )
            self._ir_mode = "adc"

        # Load trained PPO policy
        from stable_baselines3 import PPO
        self._model = PPO.load(model_path)
        print(f"[Pi] Loaded model: {model_path}")
        print(f"[Pi] Obs space:    {self._model.observation_space}")
        print(f"[Pi] Act space:    {self._model.action_space}")
        print(f"[Pi] IR mode:      {ir_mode}")
        print(f"[Pi] Motor pins:   left={left_pins}  right={right_pins}")

    # ── Motor helpers ─────────────────────────────────────────────────────────

    def _deadband(self, u: float) -> float:
        return 0.0 if abs(u) < self._motor_deadband else u

    def _slew(self, u_prev: float, u_tgt: float) -> float:
        return float(np.clip(
            u_prev + np.clip(u_tgt - u_prev, -self._motor_slew, self._motor_slew),
            -1.0, 1.0,
        ))

    def _apply_action(self, action: np.ndarray) -> None:
        """Delayed + slewed + first-order motor model, then write to gpiozero."""
        # Action delay queue
        self._action_history.append(np.clip(action.astype(np.float64), -1, 1))
        t = len(self._action_history) - 1
        cmd = self._action_history[max(0, t - self._action_delay)]

        # Slew-rate limit
        self._u_left  = self._slew(self._u_left,  self._deadband(float(cmd[0])))
        self._u_right = self._slew(self._u_right, self._deadband(float(cmd[1])))

        # Send to motors
        self._motors.set_duty(self._u_left, self._u_right)

        # Estimate wheel speeds (first-order model) for the observation
        alpha = min(1.0, DT / max(self._motor_tau, 1e-4))
        self._omega_left  += alpha * (self._u_left  * self._wheel_vel_max - self._omega_left)
        self._omega_right += alpha * (self._u_right * self._wheel_vel_max - self._omega_right)
        self._last_cmd = np.array([self._u_left, self._u_right], dtype=np.float32)

    # ── Control loop ──────────────────────────────────────────────────────────

    def run(self, max_seconds: float = 60.0) -> None:
        print(f"[Pi] Starting at {CONTROL_HZ:.0f} Hz for up to {max_seconds:.0f} s.")
        print("[Pi] Press Ctrl+C to stop.")

        t_start = time.monotonic()
        t_next  = t_start
        step = 0

        try:
            while True:
                elapsed = time.monotonic() - t_start
                if elapsed >= max_seconds:
                    print(f"[Pi] Time limit ({max_seconds:.0f} s). Stopping.")
                    break

                sleep_for = t_next - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                t_next += DT

                # ── 1. Read IR sensors ─────────────────────────────────────
                ir = self._ir.read()          # float32 [0,1], length N_IR

                # ── 2. Build observation ───────────────────────────────────
                from line_follow_env_mujoco import LineFollowEnvMuJoCo
                obs = LineFollowEnvMuJoCo.obs_from_hardware(
                    ir_readings   = ir,
                    omega_left    = self._omega_left,
                    omega_right   = self._omega_right,
                    last_cmd      = self._last_cmd,
                    wheel_vel_max = self._wheel_vel_max,
                )

                # ── 3. Policy inference ────────────────────────────────────
                action, _ = self._model.predict(obs, deterministic=True)

                # ── 4. Apply to motors ─────────────────────────────────────
                self._apply_action(action)

                step += 1
                if self._verbose and step % 50 == 0:
                    # Compute centroid lateral error for display
                    w = np.maximum(0.0, ir.max() - ir)
                    s = float(np.sum(w))
                    if s > 1e-6:
                        idx = np.arange(N_IR, dtype=np.float64)
                        lat = (float(np.dot(w, idx) / s) - 0.5*(N_IR-1)) / max(0.5*(N_IR-1), 1e-6)
                    else:
                        lat = 0.0
                    print(
                        f"[Pi] step={step:5d} t={elapsed:5.1f}s "
                        f"ir={np.round(ir, 2)} lat={lat:+.3f} "
                        f"cmd=[{action[0]:+.2f},{action[1]:+.2f}] "
                        f"u=[{self._u_left:+.2f},{self._u_right:+.2f}]"
                    )

        except KeyboardInterrupt:
            print("\n[Pi] Stopped by user.")
        finally:
            self._stop()

    def _stop(self) -> None:
        print("[Pi] Stopping motors.")
        self._motors.stop()
        self._motors.close()
        if hasattr(self._ir, "cleanup"):
            self._ir.cleanup()
        elif hasattr(self._ir, "close"):
            self._ir.close()
        print("[Pi] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi PPO line-follower — Hack Lab kit hardware",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="models/ppo_mujoco.zip",
                        help="Path to trained SB3 .zip model")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Max run time in seconds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without real hardware (mock sensors/motors)")

    # IR sensor mode
    parser.add_argument("--ir-mode", choices=("digital", "adc"), default="digital",
                        help="digital = GPIO comparator outputs; adc = MCP3008 SPI ADC")
    parser.add_argument("--calibrate", action="store_true",
                        help="Interactive ADC calibration (--ir-mode adc only), then exit")
    parser.add_argument("--ir-black", type=float, default=IR_ADC_BLACK,
                        help="Raw ADC reading over black tape (from --calibrate)")
    parser.add_argument("--ir-white", type=float, default=IR_ADC_WHITE,
                        help="Raw ADC reading over white floor (from --calibrate)")
    parser.add_argument("--ir-gamma", type=float, default=IR_GAMMA,
                        help="Photodiode gamma correction (match training)")

    # GPIO pins — defaults match the Hack Lab PDF wiring
    parser.add_argument("--left-in1",  type=int, default=LEFT_IN1,
                        help="L298N IN1 (left motor forward)")
    parser.add_argument("--left-in2",  type=int, default=LEFT_IN2,
                        help="L298N IN2 (left motor backward)")
    parser.add_argument("--right-in1", type=int, default=RIGHT_IN1,
                        help="L298N IN3 (right motor forward)")
    parser.add_argument("--right-in2", type=int, default=RIGHT_IN2,
                        help="L298N IN4 (right motor backward)")
    parser.add_argument("--ir-pins", type=int, nargs=5,
                        default=IR_DIGITAL_PINS,
                        help="5 GPIO pins for digital IR sensors (ir0..ir4)")

    # Training-alignment parameters
    parser.add_argument("--wheel-vel-max", type=float, default=WHEEL_VEL_MAX,
                        help="Max wheel speed used in training (rad/s)")
    parser.add_argument("--motor-deadband", type=float, default=MOTOR_DEADBAND)
    parser.add_argument("--motor-slew",     type=float, default=MOTOR_SLEW_PER_S,
                        help="Motor slew rate (duty/s)")
    parser.add_argument("--motor-tau",      type=float, default=MOTOR_TAU_S,
                        help="Motor first-order time constant (s)")
    parser.add_argument("--action-delay",   type=int,   default=ACTION_DELAY_STEPS)

    parser.add_argument("--verbose", action="store_true",
                        help="Print sensor/command info every 50 steps (1 s)")

    args = parser.parse_args()

    if args.calibrate:
        calibrate_adc()
        return

    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path = model_path + ".zip"
    if not Path(model_path).is_file() and not args.dry_run:
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    follower = PiLineFollower(
        model_path       = model_path,
        ir_mode          = args.ir_mode,
        dry_run          = args.dry_run,
        left_pins        = (args.left_in1,  args.left_in2),
        right_pins       = (args.right_in1, args.right_in2),
        ir_digital_pins  = args.ir_pins,
        ir_adc_black     = args.ir_black,
        ir_adc_white     = args.ir_white,
        wheel_vel_max    = args.wheel_vel_max,
        motor_deadband   = args.motor_deadband,
        motor_slew_per_s = args.motor_slew,
        motor_tau_s      = args.motor_tau,
        action_delay_steps = args.action_delay,
        ir_gamma         = args.ir_gamma,
        verbose          = args.verbose,
    )
    follower.run(max_seconds=args.duration)


if __name__ == "__main__":
    main()
