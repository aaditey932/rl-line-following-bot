from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from stable_baselines3 import PPO

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
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"


def prepare_run_dir(root: Path, stamp: str) -> Path:
    out_dir = root / f"eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def open_video_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


def resolve_model_path(raw_path: str) -> Path:
    path = Path(raw_path)
    candidates = [path] if path.suffix == ".zip" else [path, path.with_suffix(".zip")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    available = sorted(DEFAULT_MODEL_DIR.glob("*.zip")) if DEFAULT_MODEL_DIR.is_dir() else []
    available_msg = (
        "Available models: " + ", ".join(str(p.relative_to(Path.cwd())) for p in available)
        if available
        else f"No .zip models found under {DEFAULT_MODEL_DIR}"
    )
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Model file not found. Tried: {tried}. {available_msg}")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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

    if args.max_episode_steps is not None:
        env_config = replace(env_config, max_episode_steps=args.max_episode_steps)
    if args.ir_sensors is not None:
        env_config = replace(env_config, n_ir_sensors=args.ir_sensors)
    if args.line_width_m is not None:
        env_config = replace(env_config, line_half_width=0.5 * args.line_width_m)
    if args.track_type is not None:
        env_config = replace(env_config, track_type=args.track_type)
    if args.randomize_path:
        env_config = replace(env_config, randomize_path=True)

    return env_config, sim2real


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO on the MuJoCo line-following environment")
    parser.add_argument("--model", type=str, default="models/ppo_kit")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-fps", type=float, default=50.0)
    parser.add_argument("--pause-between-episodes", type=float, default=0.75)
    parser.add_argument("--save-video-root", type=str, default=None)
    parser.add_argument("--save-metadata-root", type=str, default=None)
    parser.add_argument("--video-fps", type=float, default=50.0)
    parser.add_argument("--save-trajectory", type=str, default=None, help="Path to save JSON trajectory for Three.js viewer")

    parser.add_argument("--ir-sensors", type=int, default=None)
    parser.add_argument("--line-width-m", type=float, default=None)
    parser.add_argument("--ir-noise", type=float, default=None)
    parser.add_argument("--scene-rand", action="store_true")
    parser.add_argument("--domain-rand", action="store_true")
    parser.add_argument("--scene-preset", choices=("none", "balanced", "aggressive"), default="none")
    parser.add_argument("--physics-preset", choices=("none", "balanced", "aggressive"), default="none")
    parser.add_argument("--motor-dynamics", choices=("accel", "pwm_first_order"), default=None)
    parser.add_argument("--action-delay", type=int, default=None)
    parser.add_argument(
        "--track-type",
        choices=("straight", "curve", "arc", "s_curve", "turn_sequence"),
        default=None,
    )
    parser.add_argument("--randomize-path", action="store_true")

    parser.add_argument("--sim2real-config", type=str, default=str(DEFAULT_SIM2REAL))
    parser.add_argument("--env-config", type=str, default=str(DEFAULT_ENV_CFG))
    args = parser.parse_args()

    env_config, sim2real = load_configs(args)
    digest = config_digest(env_config, sim2real)
    print(
        f"Config digest={digest} track={env_config.track_type} sensors={env_config.n_ir_sensors} "
        f"line_width={2.0 * env_config.line_half_width:.3f}m motor={sim2real.motor_dynamics} "
        f"delay={sim2real.action_delay_steps} scene_rand={sim2real.scene_randomization} "
        f"domain_rand={sim2real.domain_randomization}"
    )

    env = LineFollowEnvMuJoCo(
        render_mode="human" if args.render else None,
        sim2real=sim2real,
        env_config=env_config,
        verbose_episode=False,
        render_fps=args.render_fps,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir: Path | None = None
    metadata_dir: Path | None = None
    if args.save_video_root:
        video_dir = prepare_run_dir(Path(args.save_video_root), stamp)
        print(f"Saving episode videos to {video_dir}")
    if args.save_metadata_root:
        metadata_dir = prepare_run_dir(Path(args.save_metadata_root), stamp)
        print(f"Saving episode metadata to {metadata_dir}")
        run_meta = {
            "model": args.model,
            "stamp": stamp,
            "config_digest": digest,
            "env_snapshot": env.config_snapshot(),
            "cli_args": vars(args),
        }
        (metadata_dir / "run_config.json").write_text(json.dumps(to_jsonable(run_meta), indent=2), encoding="utf-8")

    model_path = resolve_model_path(args.model)
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env)

    returns = []
    best_traj: dict[str, Any] | None = None  # track best episode for --save-trajectory
    best_return = float("-inf")

    for ep in range(args.episodes):
        obs, reset_info = env.reset(seed=args.seed + ep)
        total_r = 0.0
        end_info: dict[str, Any] = {}
        writer: cv2.VideoWriter | None = None
        traj_frames: list[dict[str, Any]] = [] if args.save_trajectory else []
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, term, trunc, info = env.step(action)
            total_r += float(reward)
            if args.save_trajectory:
                rx, ry, ryaw = env._robot_pose_world()
                traj_frames.append({
                    "x": rx, "y": ry, "yaw": ryaw,
                    "ir": obs[:env.ec.n_ir_sensors].tolist(),
                    "action": action.tolist() if hasattr(action, "tolist") else list(action),
                    "reward": float(reward),
                })
            if video_dir is not None:
                frame = env.last_rendered_frame()
                if frame is None:
                    frame = env.render()
                if frame is not None:
                    if writer is None:
                        writer = open_video_writer(video_dir / f"episode_{ep + 1:03d}.mp4", frame, args.video_fps)
                    writer.write(frame[:, :, ::-1])
            if term or trunc:
                end_info = info
                break
        if writer is not None:
            writer.release()

        if args.save_trajectory and total_r > best_return:
            best_return = total_r
            best_traj = {
                "episode": ep + 1,
                "return": total_r,
                "track_points": env._track.points.tolist(),
                "track_type": env._track.track_type,
                "line_half_width": env.ec.line_half_width,
                "n_ir_sensors": env.ec.n_ir_sensors,
                "ir_sensor_x": float(env.ec.ir_sensor_x) if hasattr(env.ec, "ir_sensor_x") else 0.08,
                "ir_sensor_y_span": float(env.ec.ir_sensor_y_span) if hasattr(env.ec, "ir_sensor_y_span") else 0.08,
                "control_hz": 50,
                "frames": traj_frames,
            }

        returns.append(total_r)
        reason = end_info.get("termination_reason", "?")
        lat = float(end_info.get("lateral_norm", float("nan")))
        strength = float(end_info.get("line_strength", float("nan")))
        lost = int(end_info.get("line_lost_count", -1))
        steps = int(end_info.get("episode_steps", -1))
        print(
            f"ep {ep + 1:3d} | return={total_r:8.3f} reason={reason:<18s} "
            f"steps={steps:4d} |lat|={abs(lat):.3f} line_str={strength:.3f} line_lost={lost}"
        )

        if metadata_dir is not None:
            payload = {
                "episode": ep + 1,
                "seed": args.seed + ep,
                "return": total_r,
                "reset_info": reset_info,
                "end_info": end_info,
                "episode_metadata": end_info.get("episode_metadata", env.episode_metadata()),
            }
            (metadata_dir / f"episode_{ep + 1:03d}.json").write_text(
                json.dumps(to_jsonable(payload), indent=2),
                encoding="utf-8",
            )

        if args.render and args.pause_between_episodes > 0 and ep + 1 < args.episodes:
            time.sleep(args.pause_between_episodes)

    env.close()
    print(f"\nMean return: {np.mean(returns):.3f} +/- {np.std(returns):.3f}")

    if args.save_trajectory and best_traj is not None:
        traj_path = Path(args.save_trajectory)
        traj_path.write_text(json.dumps(best_traj), encoding="utf-8")
        print(f"Best trajectory (ep {best_traj['episode']}, return={best_traj['return']:.1f}) saved to {traj_path}")


if __name__ == "__main__":
    main()
