"""
Measure lateral IR error immediately after reset vs termination threshold.

Mirrors `train.py` LineFollowEnv construction. Run from repo root:

  python scripts/diagnose_reset_lateral.py --samples 2000
  python scripts/diagnose_reset_lateral.py --parity

Findings (see `IR_TERMINATE_LATERAL_NORM` and `lat_fail` in line_follow_env.py):
- With 2 IR sensors, poses that pass reset line-strength often have |lat|==1.0 (centroid saturated).
- Using `abs(lat) >= 0.92` terminated every first step (ep_len_mean=1). Fix: strict `>` and default
  threshold 1.0 disables lateral termination; episodes end on line_lost or horizon.

Uses private env methods `_compute_ir_reflectance` and `_lateral_norm_from_ir`
to match `step()` logic (noise on/off for parity checks).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from line_follow_env import (  # noqa: E402
    IR_TERMINATE_LATERAL_NORM,
    LineFollowEnv,
    Sim2RealConfig,
)


def build_env(args: argparse.Namespace) -> LineFollowEnv:
    sim2real = Sim2RealConfig(
        domain_randomization=args.domain_rand,
        action_delay_steps=args.action_delay,
        ir_noise_std=float(args.ir_noise),
        ir_photodiode_gamma=args.ir_gamma,
        ir_adc_bits=args.ir_adc_bits,
        ir_digital_output=args.ir_digital,
        ir_comparator_level=args.ir_comparator_level,
    )
    return LineFollowEnv(
        render_mode=None,
        max_episode_steps=args.max_episode_steps,
        sim2real=sim2real,
        n_ir_sensors=args.ir_sensors,
        line_half_width=0.5 * float(args.line_width_m),
        show_ir_gui=False,
        verbose_episode=False,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose |lateral| at reset vs IR_TERMINATE_LATERAL_NORM")
    p.add_argument("--samples", type=int, default=2000, help="Number of reset() calls")
    p.add_argument("--seed", type=int, default=0, help="Base seed for reset(seed=base+i)")
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--domain-rand", action="store_true")
    p.add_argument("--action-delay", type=int, default=0)
    p.add_argument("--ir-sensors", type=int, default=2)
    p.add_argument("--line-width-m", type=float, default=0.05)
    p.add_argument("--ir-noise", type=float, default=0.0)
    p.add_argument("--ir-gamma", type=float, default=0.93)
    p.add_argument("--ir-adc-bits", type=int, default=10)
    p.add_argument("--ir-digital", action="store_true")
    p.add_argument("--ir-comparator-level", type=float, default=0.5)
    p.add_argument(
        "--parity",
        action="store_true",
        help="Print train.py vs evaluate.py LineFollowEnv default differences and exit",
    )
    args = p.parse_args()

    if args.parity:
        print("LineFollowEnv defaults: train.py vs evaluate.py")
        print("  Same when CLI flags match: n_ir_sensors, line_width_m, max_episode_steps,")
        print("  Sim2RealConfig (ir_noise, ir_gamma, ir_adc_bits, ir_digital, ir_comparator_level).")
        print("  evaluate.py only: render_mode='human', gui_camera_mode, fixed_camera_*, show_ir_gui.")
        print("  train.py only: domain_randomization / action_delay via Sim2RealConfig if set on CLI.")
        print("  train.py: show_ir_gui=False; evaluate: show_ir_gui=True unless --no-ir-gui.")
        return

    env = build_env(args)
    thr = IR_TERMINATE_LATERAL_NORM

    lat_clean: list[float] = []
    lat_noisy: list[float] = []

    for i in range(args.samples):
        env.reset(seed=args.seed + i)
        assert env._robot is not None
        r0 = env._compute_ir_reflectance(add_noise=False)
        lat_clean.append(float(abs(env._lateral_norm_from_ir(r0))))
        r1 = env._compute_ir_reflectance(add_noise=True)
        lat_noisy.append(float(abs(env._lateral_norm_from_ir(r1))))

    env.close()

    a_clean = np.array(lat_clean, dtype=np.float64)
    a_noisy = np.array(lat_noisy, dtype=np.float64)

    def report(name: str, a: np.ndarray) -> None:
        # step() uses strict inequality: lat_fail = abs(lat) > IR_TERMINATE_LATERAL_NORM
        gt = float(np.mean(a > thr))
        print(f"\n{name}")
        print(f"  P(|lat| > {thr}) = {gt:.4f}  (matches lateral termination in step())")
        print(f"  min={a.min():.4f}  max={a.max():.4f}  mean={a.mean():.4f}  std={a.std():.4f}")
        for pct in (5, 25, 50, 75, 95):
            print(f"  p{pct}={np.percentile(a, pct):.4f}")

    print(f"Threshold IR_TERMINATE_LATERAL_NORM = {thr}")
    print(f"Samples = {args.samples} (train-matched env, DIRECT)")
    report("After reset, add_noise=False (same as try_pose check)", a_clean)
    report("After reset, add_noise=True (same as first step())", a_noisy)

    if args.ir_noise > 0.0:
        delta = np.abs(a_noisy - a_clean)
        print("\n|lat_noisy - lat_clean|")
        print(f"  mean={delta.mean():.6f}  max={delta.max():.6f}")


if __name__ == "__main__":
    main()
