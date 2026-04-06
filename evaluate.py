"""
Load a trained PPO and roll out in the line-following env (PyBullet GUI always on).
"""
from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from line_follow_env import (
    DEFAULT_FIXED_CAM_DISTANCE,
    DEFAULT_FIXED_CAM_PITCH_DEG,
    DEFAULT_FIXED_CAM_YAW_DEG,
    LineFollowEnv,
    Sim2RealConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained PPO on LineFollowEnv (IR observations)")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_line_follow",
        help="Path passed to train.py --save (SB3 saved .zip next to it)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Greedy policy (no sampling)")
    parser.add_argument("--ir-sensors", type=int, default=2)
    parser.add_argument(
        "--line-width-m",
        type=float,
        default=0.05,
        help="Full width of black line (m); must match training",
    )
    parser.add_argument(
        "--ir-noise",
        type=float,
        default=0.0,
        help="IR Gaussian noise std (0=off; match training, e.g. 0.025 if trained with noise)",
    )
    parser.add_argument(
        "--scene-rand",
        action="store_true",
        help="Match training: randomized IR scene appearance and slow sensor/light drift",
    )
    parser.add_argument(
        "--ir-model",
        choices=("analytic", "ray_bundle"),
        default="analytic",
        help="Match training: analytic line-distance IR or PyBullet ray bundle footprint sampling",
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
    parser.add_argument("--ir-gamma", type=float, default=0.93)
    parser.add_argument("--ir-adc-bits", type=int, default=10)
    parser.add_argument(
        "--ir-digital",
        action="store_true",
        help="Match training: comparator (digital) IR output",
    )
    parser.add_argument("--ir-comparator-level", type=float, default=0.5)
    parser.add_argument(
        "--gui-camera",
        choices=("fixed", "follow"),
        default="fixed",
        help="PyBullet main view: fixed overview of scene or chase behind robot",
    )
    parser.add_argument(
        "--no-ir-gui",
        action="store_true",
        help="Do not open matplotlib window for IR bars/strip (PyBullet GUI still opens)",
    )
    parser.add_argument(
        "--fixed-cam-dist",
        type=float,
        default=None,
        help="Override fixed GUI camera distance (default from env)",
    )
    parser.add_argument("--fixed-cam-yaw", type=float, default=None, help="Fixed camera yaw (deg)")
    parser.add_argument("--fixed-cam-pitch", type=float, default=None, help="Fixed camera pitch (deg)")
    parser.add_argument(
        "--sim2real-config",
        type=str,
        default=None,
        help="Optional JSON merged into Sim2RealConfig (match training motor_dynamics / IR / domain rand)",
    )
    args = parser.parse_args()

    render_mode = "human"
    sim2real = Sim2RealConfig(
        scene_randomization=args.scene_rand,
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

    fd = DEFAULT_FIXED_CAM_DISTANCE if args.fixed_cam_dist is None else args.fixed_cam_dist
    fy = DEFAULT_FIXED_CAM_YAW_DEG if args.fixed_cam_yaw is None else args.fixed_cam_yaw
    fp = DEFAULT_FIXED_CAM_PITCH_DEG if args.fixed_cam_pitch is None else args.fixed_cam_pitch

    def make_env() -> LineFollowEnv:
        return LineFollowEnv(
            render_mode=render_mode,
            max_episode_steps=args.max_episode_steps,
            sim2real=sim2real,
            n_ir_sensors=args.ir_sensors,
            line_half_width=0.5 * float(args.line_width_m),
            gui_camera_mode=args.gui_camera,
            fixed_camera_distance=fd,
            fixed_camera_yaw_deg=fy,
            fixed_camera_pitch_deg=fp,
            show_ir_gui=not args.no_ir_gui,
        )

    env = make_env()

    path = args.model
    if not path.endswith(".zip"):
        path = path + ".zip"
    model = PPO.load(path, env=env)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)

        total_r = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, term, trunc, _ = env.step(action)
            total_r += float(reward)
            if term or trunc:
                break
        print(f"episode {ep + 1}: return = {total_r:.3f}")

    env.close()


if __name__ == "__main__":
    main()
