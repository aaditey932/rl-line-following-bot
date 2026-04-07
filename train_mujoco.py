from __future__ import annotations

import argparse
import math
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from line_follow_env_mujoco import (
    EnvConfig,
    LineFollowEnvMuJoCo,
    Sim2RealConfig,
    apply_physics_preset,
    apply_scene_preset,
    config_digest,
)

DEFAULT_SIM2REAL = Path(__file__).resolve().parent / "mujoco_sim2real_config.json"
DEFAULT_ENV_CFG = Path(__file__).resolve().parent / "mujoco_env_config.json"


def linear_lr(lr_start: float, lr_end: float):
    def schedule(progress_remaining: float) -> float:
        return float(lr_end + progress_remaining * (lr_start - lr_end))

    return schedule


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EntropyDecayCallback(BaseCallback):
    def __init__(self, ent_start: float, ent_end: float, total: int, schedule: str = "linear", k: float = 5.0):
        super().__init__(0)
        self.ent_start = ent_start
        self.ent_end = ent_end
        self._total = max(1, total)
        self._schedule = schedule
        self._k = k

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        p = min(1.0, self.num_timesteps / self._total)
        if self._schedule == "exp":
            self.model.ent_coef = self.ent_end + (self.ent_start - self.ent_end) * math.exp(-self._k * p)
        else:
            self.model.ent_coef = self.ent_start + p * (self.ent_end - self.ent_start)


def load_configs(args: argparse.Namespace) -> tuple[EnvConfig, Sim2RealConfig]:
    env_config = EnvConfig()
    sim2real = Sim2RealConfig()

    if Path(args.env_config).is_file():
        env_config = EnvConfig.merge_json(args.env_config, env_config, strict=True)
    if Path(args.sim2real_config).is_file():
        sim2real = Sim2RealConfig.merge_json(args.sim2real_config, sim2real, strict=True)

    sim2real = apply_scene_preset(sim2real, args.scene_preset)
    sim2real = apply_physics_preset(sim2real, args.physics_preset)

    if args.scene_rand:
        sim2real = replace(sim2real, scene_randomization=True)
    if args.domain_rand:
        sim2real = replace(sim2real, domain_randomization=True)
    if args.motor_dynamics is not None:
        sim2real = replace(sim2real, motor_dynamics=args.motor_dynamics)
    if args.action_delay is not None:
        sim2real = replace(sim2real, action_delay_steps=args.action_delay)
    if args.ir_noise is not None:
        sim2real = replace(sim2real, ir_noise_std=args.ir_noise)
    if args.ir_gamma is not None:
        sim2real = replace(sim2real, ir_photodiode_gamma=args.ir_gamma)
    if args.ir_adc_bits is not None:
        sim2real = replace(sim2real, ir_adc_bits=args.ir_adc_bits)
    if args.ir_digital:
        sim2real = replace(sim2real, ir_digital_output=True)
    if args.ir_comparator_level is not None:
        sim2real = replace(sim2real, ir_comparator_level=args.ir_comparator_level)

    if args.max_episode_steps is not None:
        env_config = replace(env_config, max_episode_steps=args.max_episode_steps)
    if args.track_type is not None:
        env_config = replace(env_config, track_type=args.track_type)
    if args.randomize_path:
        env_config = replace(env_config, randomize_path=True)
    if args.ir_sensors is not None:
        env_config = replace(env_config, n_ir_sensors=args.ir_sensors)
    if args.line_width_m is not None:
        env_config = replace(env_config, line_half_width=0.5 * args.line_width_m)

    return env_config, sim2real


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on the MuJoCo line-following environment")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--save", type=str, default="models/ppo_mujoco")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=None)

    parser.add_argument(
        "--track-type",
        choices=("straight", "curve", "arc", "s_curve", "turn_sequence"),
        default=None,
    )
    parser.add_argument("--randomize-path", action="store_true")

    parser.add_argument("--ir-sensors", type=int, default=None)
    parser.add_argument("--line-width-m", type=float, default=None)
    parser.add_argument("--ir-noise", type=float, default=None)
    parser.add_argument("--ir-gamma", type=float, default=None)
    parser.add_argument("--ir-adc-bits", type=int, default=None)
    parser.add_argument("--ir-digital", action="store_true")
    parser.add_argument("--ir-comparator-level", type=float, default=None)

    parser.add_argument("--motor-dynamics", choices=("accel", "pwm_first_order"), default=None)
    parser.add_argument("--action-delay", type=int, default=None)

    parser.add_argument("--scene-preset", choices=("none", "balanced", "aggressive"), default="none")
    parser.add_argument("--physics-preset", choices=("none", "balanced", "aggressive"), default="none")
    parser.add_argument("--scene-rand", action="store_true")
    parser.add_argument("--domain-rand", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate-end", type=float, default=None)
    parser.add_argument("--entropy-start", type=float, default=0.01)
    parser.add_argument("--entropy-end", type=float, default=0.0)
    parser.add_argument("--entropy-schedule", choices=("linear", "exp"), default="linear")
    parser.add_argument("--entropy-exp-k", type=float, default=5.0)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)

    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--reproducible", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-fps", type=float, default=50.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sim2real-config", type=str, default=str(DEFAULT_SIM2REAL))
    parser.add_argument("--env-config", type=str, default=str(DEFAULT_ENV_CFG))
    args = parser.parse_args()

    if args.reproducible:
        set_seeds(args.seed)

    env_config, sim2real = load_configs(args)
    digest = config_digest(env_config, sim2real)
    print(
        f"Config digest={digest} track={env_config.track_type} sensors={env_config.n_ir_sensors} "
        f"line_width={2.0 * env_config.line_half_width:.3f}m motor={sim2real.motor_dynamics} "
        f"delay={sim2real.action_delay_steps} scene_rand={sim2real.scene_randomization} "
        f"domain_rand={sim2real.domain_randomization}"
    )

    def make_env(rank: int = 0):
        def _init():
            return LineFollowEnvMuJoCo(
                render_mode="human" if (args.render and rank == 0) else None,
                sim2real=sim2real,
                env_config=env_config,
                verbose_episode=(not args.quiet and rank == 0),
                render_fps=args.render_fps,
            )
        return _init

    n_envs = max(1, args.n_envs)
    if n_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method="fork")
    print(f"Using {n_envs} parallel env(s)")

    lr_end = args.learning_rate if args.learning_rate_end is None else args.learning_rate_end
    lr_schedule = linear_lr(args.learning_rate, lr_end)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=lr_schedule,
        ent_coef=args.entropy_start,
        n_steps=args.n_steps,
        batch_size=args.batch_size * n_envs,
        n_epochs=args.n_epochs,
    )

    callback = EntropyDecayCallback(
        args.entropy_start,
        args.entropy_end,
        args.timesteps,
        schedule=args.entropy_schedule,
        k=args.entropy_exp_k,
    )
    model.learn(total_timesteps=args.timesteps, callback=callback)

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    env.close()
    print(f"Saved model -> {out}.zip")


if __name__ == "__main__":
    main()
