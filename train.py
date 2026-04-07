"""
Train PPO on LineFollowEnv (PyBullet). Headless (DIRECT) by default; use --gui to watch training.

Exploration decays over training by default: entropy coefficient (--entropy-start → --entropy-end),
optionally with --entropy-schedule exp. Learning rate can decay linearly (--learning-rate → --learning-rate-end).

Sim2Real: optional JSON overrides via --sim2real-config (see Sim2RealConfig / sim2real_config.json).
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from line_follow_env import EnvConfig, LineFollowEnv, Sim2RealConfig

DEFAULT_SIM2REAL_CONFIG_PATH = Path(__file__).resolve().parent / "sim2real_config.json"
DEFAULT_ENV_CONFIG_PATH = Path(__file__).resolve().parent / "environment_config.json"


def linear_learning_rate_schedule(lr_start: float, lr_end: float):
    """SB3 progress_remaining goes 1 → 0 over training."""

    def schedule(progress_remaining: float) -> float:
        return float(lr_end + progress_remaining * (lr_start - lr_end))

    return schedule


def set_reproducible_training(seed: int) -> None:
    """Tighter reproducibility: Python/NumPy/Torch seeds + deterministic CuDNN (may be slower)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ExplorationDecayCallback(BaseCallback):
    """Decay PPO entropy coefficient over training (less exploration over time)."""

    def __init__(
        self,
        ent_start: float,
        ent_end: float,
        total_timesteps: int,
        *,
        schedule: str = "linear",
        exp_k: float = 5.0,
    ):
        super().__init__(0)
        self.ent_start = float(ent_start)
        self.ent_end = float(ent_end)
        self._total = max(1, int(total_timesteps))
        self._schedule = schedule
        self._exp_k = float(exp_k)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        p = min(1.0, float(self.num_timesteps) / float(self._total))
        if self._schedule == "exp":
            # ent_end + (ent_start - ent_end) * exp(-k * p); p=0 → start, p→1 → ~end
            self.model.ent_coef = self.ent_end + (self.ent_start - self.ent_end) * math.exp(
                -self._exp_k * p
            )
        else:
            self.model.ent_coef = self.ent_start + p * (self.ent_end - self.ent_start)


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO line-following in PyBullet (IR observations)")
    parser.add_argument("--timesteps", type=int, default=100_000, help="PPO learn steps")
    parser.add_argument(
        "--save",
        type=str,
        default="models/ppo_line_follow",
        help="Path prefix for saved model (SB3 adds .zip)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument(
        "--domain-rand",
        action="store_true",
        help="Sim2Real: randomize mass, friction, gravity, motor force / wheel ω max, rear joint damping",
    )
    parser.add_argument(
        "--scene-rand",
        action="store_true",
        help="Randomize IR scene appearance: black/white levels, line width/edges, floor texture, and sensor drift",
    )
    parser.add_argument("--action-delay", type=int, default=0, help="Steps of action delay")
    parser.add_argument(
        "--ir-sensors",
        type=int,
        default=2,
        help="IR sensors across the front (2 = left/right; spaced by line geometry)",
    )
    parser.add_argument(
        "--line-width-m",
        type=float,
        default=0.05,
        help="Full width of the black line strip (m); half used for IR geometry",
    )
    parser.add_argument(
        "--ir-noise",
        type=float,
        default=0.0,
        help="Per-step Gaussian IR noise std (0=off by default; e.g. 0.025 to enable)",
    )
    parser.add_argument(
        "--reproducible-training",
        action="store_true",
        help="Stricter RNG: seed Python/NumPy/Torch and use deterministic CuDNN (slower, more repeatable)",
    )
    parser.add_argument(
        "--ir-gamma",
        type=float,
        default=0.93,
        help="Photodiode response: normalized signal raised to this power (1=linear)",
    )
    parser.add_argument(
        "--ir-model",
        choices=("analytic", "ray_bundle"),
        default="analytic",
        help="IR scene model: analytic line-distance blend or PyBullet ray bundle footprint sampling",
    )
    parser.add_argument(
        "--ir-spot-rays",
        type=int,
        default=9,
        help="When --ir-model ray_bundle: rays per sensor footprint",
    )
    parser.add_argument(
        "--ir-spot-radius-m",
        type=float,
        default=0.006,
        help="When --ir-model ray_bundle: sensor footprint radius on the floor",
    )
    parser.add_argument(
        "--ir-adc-bits",
        type=int,
        default=10,
        help="ADC resolution (e.g. 10-bit MCU); 0 = no quantization",
    )
    parser.add_argument(
        "--ir-digital",
        action="store_true",
        help="Comparator output: snap each channel to black or white (digital line sensor)",
    )
    parser.add_argument(
        "--ir-comparator-level",
        type=float,
        default=0.5,
        help="Comparator threshold as fraction from black toward white (only with --ir-digital)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open PyBullet GUI while training (slower than headless DIRECT mode)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-episode print logs from the environment",
    )
    parser.add_argument(
        "--entropy-start",
        type=float,
        default=0.01,
        help="PPO entropy bonus coefficient at the beginning of training (higher = more exploration)",
    )
    parser.add_argument(
        "--entropy-end",
        type=float,
        default=0.0,
        help="PPO entropy bonus coefficient at the end of training (decays from --entropy-start)",
    )
    parser.add_argument(
        "--entropy-schedule",
        choices=("linear", "exp"),
        default="linear",
        help="How entropy decays over time: linear in progress, or exp (faster early decay with --entropy-exp-k)",
    )
    parser.add_argument(
        "--entropy-exp-k",
        type=float,
        default=5.0,
        help="With --entropy-schedule exp: multiplier in exp(-k * progress); larger = faster decay",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Initial PPO learning rate (see also --learning-rate-end)",
    )
    parser.add_argument(
        "--learning-rate-end",
        type=float,
        default=None,
        help="Final learning rate (linear decay). Default: same as --learning-rate (no LR decay)",
    )
    parser.add_argument(
        "--sim2real-config",
        type=str,
        default=str(DEFAULT_SIM2REAL_CONFIG_PATH),
        help="JSON file merged into Sim2RealConfig after CLI defaults (motor_dynamics, IR, domain rand, …)",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default=str(DEFAULT_ENV_CONFIG_PATH),
        help="JSON file merged into EnvConfig after CLI defaults (reward, reset, physics, sensor layout, …)",
    )
    args = parser.parse_args()

    if args.reproducible_training:
        set_reproducible_training(args.seed)

    sim2real = Sim2RealConfig(
        domain_randomization=args.domain_rand,
        scene_randomization=args.scene_rand,
        action_delay_steps=args.action_delay,
        ir_model=args.ir_model,
        ir_spot_rays=args.ir_spot_rays,
        ir_spot_radius_m=float(args.ir_spot_radius_m),
        ir_noise_std=float(args.ir_noise),
        ir_photodiode_gamma=args.ir_gamma,
        ir_adc_bits=args.ir_adc_bits,
        ir_digital_output=args.ir_digital,
        ir_comparator_level=args.ir_comparator_level,
    )
    if args.sim2real_config:
        sim2real = Sim2RealConfig.merge_json(args.sim2real_config, sim2real)
    env_config = EnvConfig(
        max_episode_steps=args.max_episode_steps,
        n_ir_sensors=args.ir_sensors,
        line_half_width=0.5 * float(args.line_width_m),
    )
    if args.env_config:
        env_config = EnvConfig.merge_json(args.env_config, env_config)

    def make_env() -> LineFollowEnv:
        return LineFollowEnv(
            render_mode="human" if args.gui else None,
            sim2real=sim2real,
            env_config=env_config,
            show_ir_gui=False,
            verbose_episode=not args.quiet,
        )

    env = make_env()

    lr_end = args.learning_rate if args.learning_rate_end is None else float(args.learning_rate_end)
    lr_schedule = linear_learning_rate_schedule(float(args.learning_rate), lr_end)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=lr_schedule,
        ent_coef=args.entropy_start,
    )
    explore_cb = ExplorationDecayCallback(
        args.entropy_start,
        args.entropy_end,
        args.timesteps,
        schedule=args.entropy_schedule,
        exp_k=args.entropy_exp_k,
    )
    model.learn(total_timesteps=args.timesteps, callback=explore_cb)

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    env.close()
    print(f"Saved model to {out}.zip")


if __name__ == "__main__":
    main()
