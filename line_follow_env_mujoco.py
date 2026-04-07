"""
MuJoCo line-following environment with sampled track geometry and richer IR realism.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent
MJCF_PATH = ROOT / "robots" / "diff_drive.xml"

CONTROL_FREQUENCY = 50.0
PHYSICS_TIMESTEP = 0.004
SUBSTEPS = max(1, round((1.0 / CONTROL_FREQUENCY) / PHYSICS_TIMESTEP))

WHEEL_VEL_MAX = 18.0
WHEEL_RADIUS = 0.033
WHEEL_BASELINE = 0.14

N_IR_DEFAULT = 5
IR_SENSOR_X = 0.08
IR_SENSOR_Y_SPAN = 0.08

ALIVE_BONUS = 0.02
LATERAL_WEIGHT = 1.2
PROGRESS_WEIGHT = 0.8
ACCEL_WEIGHT = 0.25
RECOVERY_WEIGHT = 0.4
CMD_PENALTY = 0.01
JITTER_PENALTY = 0.01
BAD_TRACK_PENALTY = 0.2
LINE_LOST_PENALTY = 2.0
LATERAL_TERMINAL_PENALTY = 1.0

LATERAL_TERM_THRESH = 1.0
LINE_LOST_STEPS = 12
IR_STRENGTH_EPS = 0.04
IR_LOST_STRENGTH_MULT = 0.4

DEFAULT_LINE_HALF_WIDTH = 0.025
DEFAULT_TRACK_LENGTH = 7.0
DEFAULT_TRACK_SAMPLE_STEP = 0.04
TRACK_RENDER_SEGMENTS = 64
TRACK_RENDER_THICKNESS = 0.0015

RESET_ALONG_M = 0.18
RESET_PERP_M = 0.14
RESET_YAW_RAD = 0.38
RESET_TRIES = 60
RESET_MIN_STRENGTH_MULT = 2.0
RESET_MIN_SINGLE_MULT = 2.0
RESET_FALLBACK_PERP = (0.0, 0.06, -0.06, 0.10, -0.10, 0.12, -0.12)


def _clip01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def _range_center(bounds: tuple[float, float]) -> float:
    return 0.5 * (float(bounds[0]) + float(bounds[1]))


def _json_unknown_keys(raw: dict[str, Any], allowed: set[str]) -> list[str]:
    return sorted(k for k in raw if not k.startswith("_") and k not in allowed)


def _parse_float_range(value: Any, key: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be a 2-element array")
    return float(value[0]), float(value[1])


def _rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _yaw_quat(theta: float) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    return quat


def config_digest(env_config: "EnvConfig", sim2real: "Sim2RealConfig") -> str:
    payload = json.dumps(
        {"env": asdict(env_config), "sim2real": asdict(sim2real)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class Sim2RealConfig:
    motor_dynamics: Literal["accel", "pwm_first_order"] = "pwm_first_order"
    motor_deadband: float = 0.05
    motor_slew_rate_per_s: float = 8.0
    motor_time_constant_s: float = 0.08
    pwm_discrete_levels: int = 0
    action_delay_steps: int = 1

    domain_randomization: bool = False
    mass_scale_range: tuple[float, float] = (0.90, 1.10)
    lateral_friction_range: tuple[float, float] = (0.70, 1.15)
    gravity_z_range: tuple[float, float] = (-10.05, -9.55)
    motor_force_scale_range: tuple[float, float] = (0.90, 1.10)
    wheel_vel_max_scale_range: tuple[float, float] = (0.90, 1.10)
    wheel_damping_range: tuple[float, float] = (0.02, 0.12)

    scene_randomization: bool = False
    ir_patch_width_m: float = 0.010
    ir_patch_length_m: float = 0.014
    ir_patch_samples_x: int = 3
    ir_patch_samples_y: int = 3
    ir_reflectance_black: float = 0.12
    ir_reflectance_white: float = 0.92
    ir_edge_scale: float = 0.004
    ir_reflectance_black_range: tuple[float, float] = (0.05, 0.20)
    ir_reflectance_white_range: tuple[float, float] = (0.85, 0.98)
    ir_line_half_width_range: tuple[float, float] = (0.020, 0.030)
    ir_edge_scale_range: tuple[float, float] = (0.002, 0.008)
    ir_floor_bias_amp_range: tuple[float, float] = (0.00, 0.04)
    ir_floor_bias_wavelength_range: tuple[float, float] = (0.20, 0.80)
    ir_floor_fine_noise_range: tuple[float, float] = (0.00, 0.02)
    ir_line_wear_amp_range: tuple[float, float] = (0.00, 0.03)
    ir_line_wear_wavelength_range: tuple[float, float] = (0.12, 0.45)
    ir_edge_waviness_range: tuple[float, float] = (0.000, 0.002)
    ir_edge_waviness_wavelength_range: tuple[float, float] = (0.12, 0.40)
    ir_sensor_x_jitter_std_m: float = 0.0
    ir_sensor_y_jitter_std_m: float = 0.0
    ir_sensor_gain_range: tuple[float, float] = (1.0, 1.0)
    ir_sensor_offset_range: tuple[float, float] = (0.0, 0.0)
    ir_sensor_bias_drift_std: float = 0.0
    ir_sensor_bias_clip: float = 0.02
    ir_global_light_drift_std: float = 0.0
    ir_global_light_clip: float = 0.03
    ir_global_light_gain_range: tuple[float, float] = (1.0, 1.0)
    ir_global_light_gain_drift_std: float = 0.0
    ir_global_light_gain_clip: float = 0.15

    ir_photodiode_gamma: float = 0.93
    ir_noise_std: float = 0.0
    ir_adc_bits: int = 10
    ir_digital_output: bool = False
    ir_comparator_level: float = 0.5
    ir_clip_min: float = 0.0
    ir_clip_max: float = 1.0
    ir_dropout_probability: float = 0.0
    ir_dropout_value: float = 1.0
    ir_dropout_sticky_steps: int = 1

    @staticmethod
    def merge_json(
        path: Path | str,
        base: Optional["Sim2RealConfig"] = None,
        *,
        strict: bool = False,
    ) -> "Sim2RealConfig":
        cfg = base or Sim2RealConfig()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Sim2Real JSON must be an object: {path}")

        field_names = {f.name for f in fields(Sim2RealConfig)}
        if strict:
            unknown = _json_unknown_keys(raw, field_names)
            if unknown:
                joined = ", ".join(unknown)
                raise ValueError(f"Unknown Sim2RealConfig keys in {path}: {joined}")

        float_range_keys = {
            f.name for f in fields(Sim2RealConfig)
            if f.name.endswith("_range")
        }
        int_keys = {
            "pwm_discrete_levels",
            "action_delay_steps",
            "ir_patch_samples_x",
            "ir_patch_samples_y",
            "ir_adc_bits",
            "ir_dropout_sticky_steps",
        }
        bool_keys = {"domain_randomization", "scene_randomization", "ir_digital_output"}
        kw: dict[str, Any] = {}
        for k, v in raw.items():
            if k.startswith("_") or k not in field_names:
                continue
            if k in float_range_keys:
                kw[k] = _parse_float_range(v, k)
            elif k == "motor_dynamics":
                if v not in ("accel", "pwm_first_order"):
                    raise ValueError(f"Unsupported motor_dynamics in {path}: {v}")
                kw[k] = v
            elif k in bool_keys:
                kw[k] = bool(v)
            elif k in int_keys:
                kw[k] = int(v)
            elif isinstance(v, (int, float)):
                kw[k] = float(v)
        return replace(cfg, **kw)


@dataclass
class EnvConfig:
    max_episode_steps: int = 500
    n_ir_sensors: int = N_IR_DEFAULT
    line_half_width: float = DEFAULT_LINE_HALF_WIDTH
    ir_sensor_x_body: float = IR_SENSOR_X
    ir_sensor_y_span: float = IR_SENSOR_Y_SPAN

    track_type: Literal["straight", "curve", "arc", "s_curve", "turn_sequence"] = "straight"
    randomize_path: bool = False
    path_theta_range: tuple[float, float] = (-math.pi, math.pi)
    path_offset_range: tuple[float, float] = (-0.15, 0.15)
    track_length_m: float = DEFAULT_TRACK_LENGTH
    track_sample_step_m: float = DEFAULT_TRACK_SAMPLE_STEP
    curve_amplitude_range: tuple[float, float] = (0.0, 0.30)
    curve_wavelength_range: tuple[float, float] = (1.0, 3.0)
    curve_phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    arc_radius_range: tuple[float, float] = (0.5, 2.0)
    arc_sweep_range: tuple[float, float] = (0.8, 2.2)
    s_curve_amplitude_range: tuple[float, float] = (0.06, 0.22)
    s_curve_length_range: tuple[float, float] = (1.2, 2.8)
    turn_sequence_turn_count_min: int = 2
    turn_sequence_turn_count_max: int = 4
    turn_sequence_straight_range: tuple[float, float] = (0.35, 1.00)
    turn_sequence_radius_range: tuple[float, float] = (0.45, 1.50)
    turn_sequence_angle_range: tuple[float, float] = (0.35, 1.05)

    alive_bonus: float = ALIVE_BONUS
    lateral_weight: float = LATERAL_WEIGHT
    progress_weight: float = PROGRESS_WEIGHT
    accel_weight: float = ACCEL_WEIGHT
    recovery_weight: float = RECOVERY_WEIGHT
    cmd_penalty: float = CMD_PENALTY
    jitter_penalty: float = JITTER_PENALTY
    bad_track_penalty: float = BAD_TRACK_PENALTY
    bad_track_threshold: float = 0.8
    line_lost_penalty: float = LINE_LOST_PENALTY
    lateral_terminal_penalty: float = LATERAL_TERMINAL_PENALTY
    line_visibility_target: float = 0.16

    lateral_term_thresh: float = LATERAL_TERM_THRESH
    line_lost_steps: int = LINE_LOST_STEPS
    ir_strength_eps: float = IR_STRENGTH_EPS
    ir_lost_strength_mult: float = IR_LOST_STRENGTH_MULT

    reset_along_m: float = RESET_ALONG_M
    reset_perp_m: float = RESET_PERP_M
    reset_yaw_rad: float = RESET_YAW_RAD
    reset_tries: int = RESET_TRIES
    reset_min_strength_mult: float = RESET_MIN_STRENGTH_MULT
    reset_min_single_mult: float = RESET_MIN_SINGLE_MULT
    reset_fallback_perp: tuple[float, ...] = RESET_FALLBACK_PERP

    wheel_vel_max: float = WHEEL_VEL_MAX
    kv_gain: float = 10.0

    @staticmethod
    def merge_json(
        path: Path | str,
        base: Optional["EnvConfig"] = None,
        *,
        strict: bool = False,
    ) -> "EnvConfig":
        cfg = base or EnvConfig()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"EnvConfig JSON must be an object: {path}")

        field_names = {f.name for f in fields(EnvConfig)}
        if strict:
            unknown = _json_unknown_keys(raw, field_names)
            if unknown:
                joined = ", ".join(unknown)
                raise ValueError(f"Unknown EnvConfig keys in {path}: {joined}")

        float_range_keys = {
            "path_theta_range",
            "path_offset_range",
            "curve_amplitude_range",
            "curve_wavelength_range",
            "curve_phase_range",
            "arc_radius_range",
            "arc_sweep_range",
            "s_curve_amplitude_range",
            "s_curve_length_range",
            "turn_sequence_straight_range",
            "turn_sequence_radius_range",
            "turn_sequence_angle_range",
        }
        int_keys = {
            "max_episode_steps",
            "n_ir_sensors",
            "line_lost_steps",
            "reset_tries",
            "turn_sequence_turn_count_min",
            "turn_sequence_turn_count_max",
        }
        bool_keys = {"randomize_path"}
        kw: dict[str, Any] = {}
        for k, v in raw.items():
            if k.startswith("_") or k not in field_names:
                continue
            if k in float_range_keys:
                kw[k] = _parse_float_range(v, k)
            elif k == "reset_fallback_perp":
                if not isinstance(v, (list, tuple)):
                    raise ValueError(f"{k} must be an array")
                kw[k] = tuple(float(x) for x in v)
            elif k == "track_type":
                if v not in ("straight", "curve", "arc", "s_curve", "turn_sequence"):
                    raise ValueError(f"Unsupported track_type in {path}: {v}")
                kw[k] = v
            elif k in bool_keys:
                kw[k] = bool(v)
            elif k in int_keys:
                kw[k] = int(v)
            elif isinstance(v, (int, float)):
                kw[k] = float(v)
        return replace(cfg, **kw)


def apply_scene_preset(sim2real: Sim2RealConfig, preset: str) -> Sim2RealConfig:
    if preset == "none":
        return replace(sim2real, scene_randomization=False)
    if preset == "balanced":
        return replace(
            sim2real,
            scene_randomization=True,
            ir_reflectance_black_range=(0.06, 0.18),
            ir_reflectance_white_range=(0.86, 0.98),
            ir_line_half_width_range=(0.021, 0.029),
            ir_edge_scale_range=(0.002, 0.007),
            ir_floor_bias_amp_range=(0.00, 0.03),
            ir_floor_bias_wavelength_range=(0.22, 0.75),
            ir_floor_fine_noise_range=(0.00, 0.015),
            ir_line_wear_amp_range=(0.00, 0.025),
            ir_line_wear_wavelength_range=(0.14, 0.40),
            ir_edge_waviness_range=(0.000, 0.0016),
            ir_edge_waviness_wavelength_range=(0.14, 0.35),
            ir_sensor_x_jitter_std_m=0.0008,
            ir_sensor_y_jitter_std_m=0.0015,
            ir_sensor_gain_range=(0.95, 1.05),
            ir_sensor_offset_range=(-0.025, 0.025),
            ir_sensor_bias_drift_std=max(sim2real.ir_sensor_bias_drift_std, 0.0015),
            ir_global_light_drift_std=max(sim2real.ir_global_light_drift_std, 0.0015),
            ir_global_light_gain_range=(0.94, 1.06),
            ir_global_light_gain_drift_std=max(sim2real.ir_global_light_gain_drift_std, 0.0012),
            ir_dropout_probability=max(sim2real.ir_dropout_probability, 0.002),
        )
    if preset == "aggressive":
        return replace(
            sim2real,
            scene_randomization=True,
            ir_reflectance_black_range=(0.04, 0.22),
            ir_reflectance_white_range=(0.82, 1.00),
            ir_line_half_width_range=(0.019, 0.032),
            ir_edge_scale_range=(0.0015, 0.009),
            ir_floor_bias_amp_range=(0.00, 0.05),
            ir_floor_bias_wavelength_range=(0.18, 0.90),
            ir_floor_fine_noise_range=(0.00, 0.02),
            ir_line_wear_amp_range=(0.00, 0.035),
            ir_line_wear_wavelength_range=(0.10, 0.48),
            ir_edge_waviness_range=(0.000, 0.0024),
            ir_edge_waviness_wavelength_range=(0.10, 0.42),
            ir_sensor_x_jitter_std_m=0.0012,
            ir_sensor_y_jitter_std_m=0.0020,
            ir_sensor_gain_range=(0.92, 1.08),
            ir_sensor_offset_range=(-0.035, 0.035),
            ir_sensor_bias_drift_std=max(sim2real.ir_sensor_bias_drift_std, 0.0025),
            ir_global_light_drift_std=max(sim2real.ir_global_light_drift_std, 0.0025),
            ir_global_light_gain_range=(0.90, 1.10),
            ir_global_light_gain_drift_std=max(sim2real.ir_global_light_gain_drift_std, 0.0018),
            ir_dropout_probability=max(sim2real.ir_dropout_probability, 0.005),
        )
    raise ValueError(f"Unsupported scene preset: {preset}")


def apply_physics_preset(sim2real: Sim2RealConfig, preset: str) -> Sim2RealConfig:
    if preset == "none":
        return replace(sim2real, domain_randomization=False)
    if preset == "balanced":
        return replace(
            sim2real,
            domain_randomization=True,
            mass_scale_range=(0.92, 1.10),
            lateral_friction_range=(0.70, 1.15),
            gravity_z_range=(-10.05, -9.55),
            motor_force_scale_range=(0.92, 1.10),
            wheel_vel_max_scale_range=(0.92, 1.10),
            wheel_damping_range=(0.02, 0.11),
        )
    if preset == "aggressive":
        return replace(
            sim2real,
            domain_randomization=True,
            mass_scale_range=(0.85, 1.18),
            lateral_friction_range=(0.60, 1.20),
            gravity_z_range=(-10.20, -9.40),
            motor_force_scale_range=(0.85, 1.18),
            wheel_vel_max_scale_range=(0.85, 1.16),
            wheel_damping_range=(0.01, 0.16),
        )
    raise ValueError(f"Unsupported physics preset: {preset}")


class TrackGeometry:
    def __init__(self, track_type: str, points: np.ndarray, metadata: Optional[dict[str, Any]] = None):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("TrackGeometry requires at least two 2-D points")
        deltas = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(deltas, axis=1)
        keep = seg_lengths > 1e-9
        if not np.all(keep):
            pts = np.vstack([pts[0], pts[1:][keep]])
            deltas = np.diff(pts, axis=0)
            seg_lengths = np.linalg.norm(deltas, axis=1)
        if len(seg_lengths) == 0:
            raise ValueError("TrackGeometry collapsed to zero length")
        self.track_type = track_type
        self.points = pts
        self.segment_starts = pts[:-1]
        self.segment_ends = pts[1:]
        self.segment_vectors = deltas
        self.segment_lengths = seg_lengths
        self.segment_tangents = deltas / seg_lengths[:, None]
        self.cumulative_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.length = float(self.cumulative_s[-1])
        self.metadata = metadata or {}

    @staticmethod
    def _transform(points: np.ndarray, theta: float, offset: float) -> np.ndarray:
        rot = _rotation_matrix(theta)
        normal = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)
        return points @ rot.T + offset * normal

    @staticmethod
    def _sample_x_range(total_length: float, step: float) -> np.ndarray:
        n = max(2, int(math.ceil(max(total_length, step) / max(step, 1e-3))) + 1)
        return np.linspace(-0.5 * total_length, 0.5 * total_length, n, dtype=np.float64)

    @classmethod
    def straight(cls, total_length: float, step: float, theta: float, offset: float) -> "TrackGeometry":
        x = cls._sample_x_range(total_length, step)
        pts = np.stack([x, np.zeros_like(x)], axis=1)
        pts = cls._transform(pts, theta, offset)
        return cls(
            "straight",
            pts,
            {
                "track_type": "straight",
                "theta": float(theta),
                "offset": float(offset),
                "track_length_m": float(total_length),
            },
        )

    @classmethod
    def curve(
        cls,
        total_length: float,
        step: float,
        theta: float,
        offset: float,
        amplitude: float,
        wavelength: float,
        phase: float,
    ) -> "TrackGeometry":
        x = cls._sample_x_range(total_length, step)
        y = amplitude * np.sin((2.0 * math.pi / max(wavelength, 1e-3)) * x + phase)
        pts = np.stack([x, y], axis=1)
        pts = cls._transform(pts, theta, offset)
        return cls(
            "curve",
            pts,
            {
                "track_type": "curve",
                "theta": float(theta),
                "offset": float(offset),
                "track_length_m": float(total_length),
                "curve_amplitude": float(amplitude),
                "curve_wavelength": float(wavelength),
                "curve_phase": float(phase),
            },
        )

    @classmethod
    def arc(
        cls,
        step: float,
        theta: float,
        offset: float,
        radius: float,
        sign: int,
        sweep_angle: float,
    ) -> "TrackGeometry":
        n = max(3, int(math.ceil(abs(radius * sweep_angle) / max(step, 1e-3))) + 1)
        angles = np.linspace(-0.5 * sweep_angle, 0.5 * sweep_angle, n, dtype=np.float64)
        x = radius * np.sin(angles)
        y = float(sign) * radius * (1.0 - np.cos(angles))
        pts = np.stack([x, y], axis=1)
        pts = cls._transform(pts, theta, offset)
        turn = "left" if sign > 0 else "right"
        return cls(
            "arc",
            pts,
            {
                "track_type": "arc",
                "theta": float(theta),
                "offset": float(offset),
                "arc_radius": float(radius),
                "arc_sign": int(sign),
                "arc_turn": turn,
                "arc_sweep_angle": float(sweep_angle),
            },
        )

    @classmethod
    def s_curve(
        cls,
        length_m: float,
        step: float,
        theta: float,
        offset: float,
        amplitude: float,
    ) -> "TrackGeometry":
        x = cls._sample_x_range(length_m, step)
        u = (x + 0.5 * length_m) / max(length_m, 1e-6)
        y = amplitude * np.sin(2.0 * math.pi * u)
        pts = np.stack([x, y], axis=1)
        pts = cls._transform(pts, theta, offset)
        return cls(
            "s_curve",
            pts,
            {
                "track_type": "s_curve",
                "theta": float(theta),
                "offset": float(offset),
                "s_curve_amplitude": float(amplitude),
                "s_curve_length_m": float(length_m),
            },
        )

    @staticmethod
    def _append_straight(points: list[np.ndarray], heading: float, length: float, step: float) -> float:
        n = max(1, int(math.ceil(length / max(step, 1e-3))))
        ds = length / n
        pos = points[-1].copy()
        for _ in range(n):
            pos = pos + ds * np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
            points.append(pos.copy())
        return heading

    @staticmethod
    def _append_arc(
        points: list[np.ndarray],
        heading: float,
        radius: float,
        angle: float,
        sign: int,
        step: float,
    ) -> float:
        arc_len = abs(radius * angle)
        n = max(1, int(math.ceil(arc_len / max(step, 1e-3))))
        ds = arc_len / n
        curvature = float(sign) / max(radius, 1e-6)
        pos = points[-1].copy()
        for _ in range(n):
            mid_heading = heading + 0.5 * curvature * ds
            pos = pos + ds * np.array([math.cos(mid_heading), math.sin(mid_heading)], dtype=np.float64)
            heading = heading + curvature * ds
            points.append(pos.copy())
        return heading

    @classmethod
    def turn_sequence(
        cls,
        rng: np.random.Generator,
        step: float,
        theta: float,
        offset: float,
        turn_count: int,
        straight_range: tuple[float, float],
        radius_range: tuple[float, float],
        angle_range: tuple[float, float],
    ) -> "TrackGeometry":
        points = [np.array([0.0, 0.0], dtype=np.float64)]
        heading = 0.0
        segments: list[dict[str, Any]] = []
        prev = points[-1].copy()
        heading = cls._append_straight(points, heading, float(rng.uniform(*straight_range)), step)
        segments.append({"kind": "straight", "length": float(np.linalg.norm(points[-1] - prev))})

        for turn_idx in range(turn_count):
            radius = float(rng.uniform(*radius_range))
            angle = float(rng.uniform(*angle_range))
            sign = int(rng.choice([-1, 1]))
            heading = cls._append_arc(points, heading, radius, angle, sign, step)
            segments.append(
                {
                    "kind": "arc",
                    "radius": radius,
                    "angle": angle,
                    "sign": sign,
                    "turn": "left" if sign > 0 else "right",
                }
            )
            if turn_idx != turn_count - 1:
                straight_len = float(rng.uniform(*straight_range))
                prev = points[-1].copy()
                heading = cls._append_straight(points, heading, straight_len, step)
                segments.append(
                    {
                        "kind": "straight",
                        "length": float(np.linalg.norm(points[-1] - prev)),
                    }
                )
        prev = points[-1].copy()
        heading = cls._append_straight(points, heading, float(rng.uniform(*straight_range)), step)
        segments.append({"kind": "straight", "length": float(np.linalg.norm(points[-1] - prev))})

        pts = np.asarray(points, dtype=np.float64)
        tmp = cls("turn_sequence", pts)
        mid_s = 0.5 * tmp.length
        mid_pt, mid_tan = tmp.point_and_tangent_at_s(mid_s)
        local = pts - mid_pt
        mid_heading = math.atan2(mid_tan[1], mid_tan[0])
        local = local @ _rotation_matrix(-mid_heading).T
        pts = cls._transform(local, theta, offset)
        return cls(
            "turn_sequence",
            pts,
            {
                "track_type": "turn_sequence",
                "theta": float(theta),
                "offset": float(offset),
                "turn_count": int(turn_count),
                "segments": segments,
            },
        )

    def nearest_point_and_tangent(self, px: float, py: float) -> tuple[float, float, float, float, float, float]:
        # Vectorised over all segments
        p = np.array([px, py], dtype=np.float64)
        rel = p[None, :] - self.segment_starts          # (N,2)
        seg_len2 = self.segment_lengths ** 2             # (N,)
        t = np.clip(
            np.einsum("ni,ni->n", rel, self.segment_vectors) / np.maximum(seg_len2, 1e-9),
            0.0, 1.0,
        )                                                 # (N,)
        proj = self.segment_starts + t[:, None] * self.segment_vectors  # (N,2)
        diff = p[None, :] - proj                         # (N,2)
        dist2 = np.einsum("ni,ni->n", diff, diff)        # (N,)
        idx = int(np.argmin(dist2))
        tan = self.segment_tangents[idx]
        normal = np.array([-tan[1], tan[0]], dtype=np.float64)
        d = diff[idx]
        return (
            float(proj[idx, 0]),
            float(proj[idx, 1]),
            float(tan[0]),
            float(tan[1]),
            float(np.dot(d, normal)),
            float(self.cumulative_s[idx] + t[idx] * self.segment_lengths[idx]),
        )

    def nearest_points_batch(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised batch projection. pts: (M,2) -> lat_d: (M,), s_along: (M,), idx: (M,)"""
        # pts (M,2), segment_starts (N,2), segment_vectors (N,2)
        M = len(pts)
        rel = pts[:, None, :] - self.segment_starts[None, :, :]   # (M,N,2)
        seg_len2 = self.segment_lengths ** 2                        # (N,)
        dot_rv = np.einsum("mni,ni->mn", rel, self.segment_vectors) # (M,N)
        t = np.clip(dot_rv / np.maximum(seg_len2[None, :], 1e-9), 0.0, 1.0)  # (M,N)
        proj = self.segment_starts[None, :, :] + t[:, :, None] * self.segment_vectors[None, :, :]  # (M,N,2)
        diff = pts[:, None, :] - proj                               # (M,N,2)
        dist2 = np.einsum("mni,mni->mn", diff, diff)                # (M,N)
        best_idx = np.argmin(dist2, axis=1)                         # (M,)
        m_idx = np.arange(M)
        best_diff = diff[m_idx, best_idx, :]                        # (M,2)
        best_tan = self.segment_tangents[best_idx]                  # (M,2)
        normal = np.stack([-best_tan[:, 1], best_tan[:, 0]], axis=1)  # (M,2)
        lat_d = np.einsum("mi,mi->m", best_diff, normal)            # (M,)
        s_along = self.cumulative_s[best_idx] + t[m_idx, best_idx] * self.segment_lengths[best_idx]  # (M,)
        return lat_d, s_along, best_idx

    def point_and_tangent_at_s(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        s = float(np.clip(s, 0.0, self.length))
        idx = int(np.searchsorted(self.cumulative_s, s, side="right") - 1)
        idx = max(0, min(idx, len(self.segment_lengths) - 1))
        local_s = s - float(self.cumulative_s[idx])
        seg_len = float(self.segment_lengths[idx])
        t = 0.0 if seg_len < 1e-9 else local_s / seg_len
        point = self.segment_starts[idx] + t * self.segment_vectors[idx]
        tangent = self.segment_tangents[idx]
        return point.copy(), tangent.copy()

    def resample_segment_pairs(self, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
        count = max(1, count)
        edges = np.linspace(0.0, self.length, count + 1, dtype=np.float64)
        pairs = []
        for s0, s1 in zip(edges[:-1], edges[1:]):
            p0, _ = self.point_and_tangent_at_s(float(s0))
            p1, _ = self.point_and_tangent_at_s(float(s1))
            if np.linalg.norm(p1 - p0) > 1e-6:
                pairs.append((p0, p1))
        return pairs


class LineFollowEnvMuJoCo(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        sim2real: Optional[Sim2RealConfig] = None,
        env_config: Optional[EnvConfig] = None,
        *,
        verbose_episode: bool = False,
        render_fps: Optional[float] = CONTROL_FREQUENCY,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.sim2real = sim2real or Sim2RealConfig()
        self.ec = env_config or EnvConfig()

        if not MJCF_PATH.is_file():
            raise FileNotFoundError(f"Missing MuJoCo model: {MJCF_PATH}")

        self._model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = PHYSICS_TIMESTEP
        self._model.opt.integrator = 3

        self._body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._left_jnt = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
        self._right_jnt = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")
        self._act_left = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_vel")
        self._act_right = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_vel")
        self._geom_floor = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._geom_line = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "line_strip")
        self._line_geom_ids = []
        for idx in range(TRACK_RENDER_SEGMENTS):
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"line_seg_{idx:03d}")
            if gid < 0:
                break
            self._line_geom_ids.append(gid)

        self._ir_site_ids = []
        for i in range(5):
            sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, f"ir{i}")
            if sid >= 0:
                self._ir_site_ids.append(sid)
        if not 1 <= self.ec.n_ir_sensors <= len(self._ir_site_ids):
            raise ValueError(f"n_ir_sensors must be in [1, {len(self._ir_site_ids)}], got {self.ec.n_ir_sensors}")

        self._ctrl_range_left = tuple(float(x) for x in self._model.actuator_ctrlrange[self._act_left])
        self._ctrl_range_right = tuple(float(x) for x in self._model.actuator_ctrlrange[self._act_right])

        n = self.ec.n_ir_sensors
        lo = np.concatenate([np.zeros(n, np.float32), np.full(4, -1.0, np.float32)])
        hi = np.concatenate([np.ones(n, np.float32), np.ones(4, np.float32)])
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._step_count = 0
        self._line_lost_ctr = 0
        self._prev_fwd_speed = 0.0
        self._prev_line_strength = 0.0
        self._last_cmd = np.zeros(2, np.float64)
        self._prev_cmd = np.zeros(2, np.float64)
        self._last_ir = np.ones(n, np.float64)
        self._action_history: list[np.ndarray] = []
        self._omega_left = 0.0
        self._omega_right = 0.0
        self._motor_u_left = 0.0
        self._motor_u_right = 0.0
        self._motor_force_scale = 1.0
        self._wheel_vel_max = float(self.ec.wheel_vel_max)

        self._renderer: Optional[mujoco.Renderer] = None
        self._viewer: Any = None
        self._human_render_backend: Optional[str] = None
        self._render_period_s = None if render_fps is None or float(render_fps) <= 0 else 1.0 / float(render_fps)
        self._last_render_t: Optional[float] = None
        self._last_rendered_frame: Optional[np.ndarray] = None
        self._verbose_episode = bool(verbose_episode)
        self._episode_idx = 0
        self._episode_return = 0.0

        self._base_mass_nominal = float(self._model.body_mass[self._body_id])
        self._wheel_damping_nominal = np.array(
            [
                float(self._model.dof_damping[self._model.joint("left_wheel_joint").dofadr[0]]),
                float(self._model.dof_damping[self._model.joint("right_wheel_joint").dofadr[0]]),
            ],
            dtype=np.float64,
        )
        self._wheel_friction_nominal = np.array(
            [
                float(self._model.geom_friction[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel"), 0]),
                float(self._model.geom_friction[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel"), 0]),
            ],
            dtype=np.float64,
        )
        self._gravity_nominal = self._model.opt.gravity.copy()

        self._ep_ir_black = self.sim2real.ir_reflectance_black
        self._ep_ir_white = self.sim2real.ir_reflectance_white
        self._ep_line_hw = self.ec.line_half_width
        self._ep_edge_scale = self.sim2real.ir_edge_scale
        self._floor_bias_amp = 0.0
        self._floor_bias_wl = 1.0
        self._floor_bias_phase = np.zeros(3, dtype=np.float64)
        self._floor_fine_amp = 0.0
        self._floor_fine_phase = np.zeros(2, dtype=np.float64)
        self._line_wear_amp = 0.0
        self._line_wear_wl = 1.0
        self._line_wear_phase = np.zeros(2, dtype=np.float64)
        self._edge_wav_amp = 0.0
        self._edge_wav_wl = 1.0
        self._edge_wav_phase = np.zeros(2, dtype=np.float64)
        self._sensor_pose_offsets = np.zeros((n, 2), dtype=np.float64)
        self._sensor_gain = np.ones(n, dtype=np.float64)
        self._sensor_offset = np.zeros(n, dtype=np.float64)
        self._sensor_bias = np.zeros(n, dtype=np.float64)
        self._global_light_bias = 0.0
        self._global_light_gain = 1.0
        self._dropout_remaining = np.zeros(n, dtype=np.int32)
        self._track = TrackGeometry.straight(self.ec.track_length_m, self.ec.track_sample_step_m, 0.0, 0.0)
        self._config_digest = config_digest(self.ec, self.sim2real)
        self._scene_params: dict[str, Any] = {}
        self._domain_params: dict[str, Any] = {}

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    def _ir_patch_local_points(self) -> np.ndarray:
        sx = max(1, int(self.sim2real.ir_patch_samples_x))
        sy = max(1, int(self.sim2real.ir_patch_samples_y))
        xs = np.linspace(-0.5 * self.sim2real.ir_patch_length_m, 0.5 * self.sim2real.ir_patch_length_m, sx)
        ys = np.linspace(-0.5 * self.sim2real.ir_patch_width_m, 0.5 * self.sim2real.ir_patch_width_m, sy)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)

    def _ir_sensor_world_samples(self) -> np.ndarray:
        mujoco.mj_kinematics(self._model, self._data)
        local_patch = self._ir_patch_local_points()
        sample_count = len(local_patch)
        pts = np.zeros((self.ec.n_ir_sensors, sample_count, 2), dtype=np.float64)
        for i, sid in enumerate(self._ir_site_ids[:self.ec.n_ir_sensors]):
            origin = self._data.site_xpos[sid]
            rot = self._data.site_xmat[sid].reshape(3, 3)
            local = np.zeros((sample_count, 3), dtype=np.float64)
            local[:, :2] = local_patch + self._sensor_pose_offsets[i]
            world = origin[None, :] + local @ rot.T
            pts[i] = world[:, :2]
        return pts

    def _scene_intensity(self, world_pts: np.ndarray) -> np.ndarray:
        # Fully vectorised — no Python loop over points
        denom = max(self._ep_ir_white - self._ep_ir_black, 1e-6)
        lat_d, s_along, _ = self._track.nearest_points_batch(world_pts)  # (M,)

        # Edge waviness
        if self._edge_wav_amp > 0:
            k = 2.0 * math.pi / max(self._edge_wav_wl, 1e-6)
            p0, p1 = self._edge_wav_phase
            wav_off = self._edge_wav_amp * (
                0.75 * np.sin(k * s_along + p0) + 0.25 * np.sin(0.5 * k * s_along + p1)
            )
        else:
            wav_off = 0.0
        eff_d = lat_d - wav_off
        sigma = max(self._ep_edge_scale, 1e-6)
        mix = 1.0 / (1.0 + np.exp(-np.clip((self._ep_line_hw - np.abs(eff_d)) / sigma, -60.0, 60.0)))

        # Floor bias
        x, y = world_pts[:, 0], world_pts[:, 1]
        if self._floor_bias_amp > 0:
            k = 2.0 * math.pi / max(self._floor_bias_wl, 1e-6)
            p0, p1, p2 = self._floor_bias_phase
            floor_bias = self._floor_bias_amp * (
                0.55 * np.sin(k * x + p0)
                + 0.30 * np.sin(k * y + p1)
                + 0.15 * np.sin(k * (0.7 * x + 0.3 * y) + p2)
            )
        else:
            floor_bias = np.zeros(len(world_pts), dtype=np.float64)

        if self._floor_fine_amp > 0:
            p0, p1 = self._floor_fine_phase
            fine_noise = self._floor_fine_amp * np.sin(29.0 * x + p0) * np.sin(23.0 * y + p1)
        else:
            fine_noise = 0.0

        floor_r = np.clip(self._ep_ir_white + floor_bias + fine_noise, 0.0, 1.0)

        # Line wear
        if self._line_wear_amp > 0:
            k = 2.0 * math.pi / max(self._line_wear_wl, 1e-6)
            p0, p1 = self._line_wear_phase
            wear = self._line_wear_amp * (
                0.7 * np.sin(k * s_along + p0) + 0.3 * np.sin(0.5 * k * s_along + p1)
            )
        else:
            wear = 0.0

        line_r = np.clip(
            self._ep_ir_black + wear + 0.2 * floor_bias,
            0.0,
            np.maximum(floor_r - 1e-3, 1e-3),
        )
        reflectance = floor_r * (1.0 - mix) + line_r * mix
        return np.clip((reflectance - self._ep_ir_black) / denom, 0.0, 1.0)

    def _compute_ir(self, add_noise: bool) -> np.ndarray:
        samples = self._ir_sensor_world_samples()
        whiteness = self._scene_intensity(samples.reshape(-1, 2)).reshape(samples.shape[0], samples.shape[1]).mean(axis=1)
        whiteness = np.power(np.clip(whiteness, 0.0, 1.0), max(self.sim2real.ir_photodiode_gamma, 1e-6))

        out = whiteness * self._sensor_gain + self._sensor_offset
        if add_noise:
            if self.sim2real.ir_global_light_drift_std > 0:
                self._global_light_bias = float(np.clip(
                    self._global_light_bias + self.np_random.normal(0.0, self.sim2real.ir_global_light_drift_std),
                    -self.sim2real.ir_global_light_clip,
                    self.sim2real.ir_global_light_clip,
                ))
            if self.sim2real.ir_global_light_gain_drift_std > 0:
                self._global_light_gain = float(np.clip(
                    self._global_light_gain + self.np_random.normal(0.0, self.sim2real.ir_global_light_gain_drift_std),
                    1.0 - self.sim2real.ir_global_light_gain_clip,
                    1.0 + self.sim2real.ir_global_light_gain_clip,
                ))
            if self.sim2real.ir_sensor_bias_drift_std > 0:
                self._sensor_bias = np.clip(
                    self._sensor_bias + self.np_random.normal(0.0, self.sim2real.ir_sensor_bias_drift_std, size=out.shape),
                    -self.sim2real.ir_sensor_bias_clip,
                    self.sim2real.ir_sensor_bias_clip,
                )
            out = self._global_light_gain * out + self._global_light_bias + self._sensor_bias
        if add_noise and self.sim2real.ir_noise_std > 0:
            out = out + self.np_random.normal(0.0, self.sim2real.ir_noise_std, size=out.shape)

        clip_lo = float(np.clip(min(self.sim2real.ir_clip_min, self.sim2real.ir_clip_max), 0.0, 1.0))
        clip_hi = float(np.clip(max(self.sim2real.ir_clip_min, self.sim2real.ir_clip_max), clip_lo, 1.0))
        out = np.clip(out, clip_lo, clip_hi)

        if add_noise and self.sim2real.ir_dropout_probability > 0:
            sticky_steps = max(1, int(self.sim2real.ir_dropout_sticky_steps))
            fresh = (self._dropout_remaining <= 0) & (
                self.np_random.random(self.ec.n_ir_sensors) < self.sim2real.ir_dropout_probability
            )
            self._dropout_remaining[fresh] = sticky_steps
            mask = self._dropout_remaining > 0
            out[mask] = float(np.clip(self.sim2real.ir_dropout_value, clip_lo, clip_hi))
            self._dropout_remaining[mask] -= 1

        if self.sim2real.ir_adc_bits > 0:
            levels = (2 ** int(self.sim2real.ir_adc_bits)) - 1
            out = np.round(out * levels) / max(levels, 1)
        if self.sim2real.ir_digital_output:
            out = np.where(out >= float(self.sim2real.ir_comparator_level), 1.0, 0.0)
        return np.clip(out.astype(np.float64), 0.0, 1.0)

    def _lateral_norm(self, ir: np.ndarray) -> float:
        n = len(ir)
        if n == 1:
            return float(np.clip(1.0 - ir[0], 0.0, 1.0))
        darkness = np.maximum(0.0, 1.0 - ir)
        strength = float(np.sum(darkness))
        if strength < float(self.ec.ir_strength_eps):
            return 0.0
        idx = np.arange(n, dtype=np.float64)
        centroid = float(np.dot(darkness, idx) / strength)
        mid = 0.5 * float(n - 1)
        half = max(mid, 1e-6)
        return float((centroid - mid) / half)

    def _line_strength(self, ir: np.ndarray) -> float:
        return float(np.sum(np.maximum(0.0, 1.0 - ir)))

    def _motor_cmd(self, action: np.ndarray) -> np.ndarray:
        delay = max(0, int(self.sim2real.action_delay_steps))
        self._action_history.append(np.clip(action.astype(np.float64), -1.0, 1.0))
        idx = max(0, len(self._action_history) - 1 - delay)
        return self._action_history[idx]

    def _slew(self, u_prev: float, u_tgt: float) -> float:
        max_du = self.sim2real.motor_slew_rate_per_s / CONTROL_FREQUENCY
        return float(np.clip(u_prev + np.clip(u_tgt - u_prev, -max_du, max_du), -1.0, 1.0))

    def _deadband(self, u: float) -> float:
        return 0.0 if abs(u) < self.sim2real.motor_deadband else u

    def _pwm_quantize(self, u: float) -> float:
        levels = int(self.sim2real.pwm_discrete_levels)
        if levels <= 0:
            return u
        sign = 1.0 if u >= 0 else -1.0
        return sign * round(abs(u) * (levels - 1)) / max(levels - 1, 1)

    def _first_order(self, omega: float, omega_star: float) -> float:
        tau = max(self.sim2real.motor_time_constant_s, 1e-4)
        alpha = min(1.0, (1.0 / CONTROL_FREQUENCY) / tau)
        return float(omega + alpha * (omega_star - omega))

    def _apply_drive_command(self, cmd: np.ndarray) -> None:
        wmx = self._wheel_vel_max
        if self.sim2real.motor_dynamics == "accel":
            dt = 1.0 / CONTROL_FREQUENCY
            max_accel = 48.0
            self._omega_left = float(np.clip(self._omega_left + float(cmd[0]) * max_accel * dt, -wmx, wmx))
            self._omega_right = float(np.clip(self._omega_right + float(cmd[1]) * max_accel * dt, -wmx, wmx))
        else:
            u_l = self._pwm_quantize(self._slew(self._motor_u_left, self._deadband(float(cmd[0]))))
            u_r = self._pwm_quantize(self._slew(self._motor_u_right, self._deadband(float(cmd[1]))))
            self._motor_u_left = u_l
            self._motor_u_right = u_r
            self._omega_left = float(np.clip(self._first_order(self._omega_left, u_l * wmx), -wmx, wmx))
            self._omega_right = float(np.clip(self._first_order(self._omega_right, u_r * wmx), -wmx, wmx))

        self._data.ctrl[self._act_left] = float(np.clip(
            self._omega_left * self._motor_force_scale,
            self._ctrl_range_left[0],
            self._ctrl_range_left[1],
        ))
        self._data.ctrl[self._act_right] = float(np.clip(
            self._omega_right * self._motor_force_scale,
            self._ctrl_range_right[0],
            self._ctrl_range_right[1],
        ))

    def _sample_scene(self) -> None:
        cfg = self.sim2real
        rng = self.np_random
        n = self.ec.n_ir_sensors
        if not cfg.scene_randomization:
            self._ep_ir_black = cfg.ir_reflectance_black
            self._ep_ir_white = cfg.ir_reflectance_white
            self._ep_line_hw = self.ec.line_half_width
            self._ep_edge_scale = cfg.ir_edge_scale
            self._floor_bias_amp = 0.0
            self._floor_fine_amp = 0.0
            self._line_wear_amp = 0.0
            self._edge_wav_amp = 0.0
            self._sensor_pose_offsets = np.zeros((n, 2), dtype=np.float64)
            self._sensor_gain = np.ones(n, dtype=np.float64)
            self._sensor_offset = np.zeros(n, dtype=np.float64)
            self._sensor_bias = np.zeros(n, dtype=np.float64)
            self._global_light_bias = 0.0
            self._global_light_gain = 1.0
            self._dropout_remaining = np.zeros(n, dtype=np.int32)
        else:
            self._ep_ir_black = float(rng.uniform(*cfg.ir_reflectance_black_range))
            self._ep_ir_white = float(rng.uniform(*cfg.ir_reflectance_white_range))
            self._ep_line_hw = float(rng.uniform(*cfg.ir_line_half_width_range))
            self._ep_edge_scale = float(rng.uniform(*cfg.ir_edge_scale_range))
            self._floor_bias_amp = float(rng.uniform(*cfg.ir_floor_bias_amp_range))
            self._floor_bias_wl = float(rng.uniform(*cfg.ir_floor_bias_wavelength_range))
            self._floor_bias_phase = rng.uniform(0.0, 2.0 * math.pi, 3)
            self._floor_fine_amp = float(rng.uniform(*cfg.ir_floor_fine_noise_range))
            self._floor_fine_phase = rng.uniform(0.0, 2.0 * math.pi, 2)
            self._line_wear_amp = float(rng.uniform(*cfg.ir_line_wear_amp_range))
            self._line_wear_wl = float(rng.uniform(*cfg.ir_line_wear_wavelength_range))
            self._line_wear_phase = rng.uniform(0.0, 2.0 * math.pi, 2)
            self._edge_wav_amp = float(rng.uniform(*cfg.ir_edge_waviness_range))
            self._edge_wav_wl = float(rng.uniform(*cfg.ir_edge_waviness_wavelength_range))
            self._edge_wav_phase = rng.uniform(0.0, 2.0 * math.pi, 2)
            self._sensor_pose_offsets = np.column_stack([
                rng.normal(0.0, cfg.ir_sensor_x_jitter_std_m, size=n),
                rng.normal(0.0, cfg.ir_sensor_y_jitter_std_m, size=n),
            ])
            self._sensor_gain = rng.uniform(*cfg.ir_sensor_gain_range, size=n)
            self._sensor_offset = rng.uniform(*cfg.ir_sensor_offset_range, size=n)
            self._sensor_bias = np.zeros(n, dtype=np.float64)
            self._global_light_bias = 0.0
            self._global_light_gain = float(rng.uniform(*cfg.ir_global_light_gain_range))
            self._dropout_remaining = np.zeros(n, dtype=np.int32)

        self._scene_params = {
            "scene_randomization": bool(cfg.scene_randomization),
            "ir_reflectance_black": float(self._ep_ir_black),
            "ir_reflectance_white": float(self._ep_ir_white),
            "ir_line_half_width": float(self._ep_line_hw),
            "ir_edge_scale": float(self._ep_edge_scale),
            "ir_patch_width_m": float(cfg.ir_patch_width_m),
            "ir_patch_length_m": float(cfg.ir_patch_length_m),
            "ir_patch_samples_x": int(cfg.ir_patch_samples_x),
            "ir_patch_samples_y": int(cfg.ir_patch_samples_y),
            "sensor_pose_offsets": self._sensor_pose_offsets.tolist(),
            "sensor_gain": self._sensor_gain.tolist(),
            "sensor_offset": self._sensor_offset.tolist(),
            "initial_global_light_gain": float(self._global_light_gain),
        }

    def _sample_track(self) -> None:
        rng = self.np_random
        ec = self.ec
        theta = float(rng.uniform(*ec.path_theta_range)) if ec.randomize_path else 0.0
        offset = float(rng.uniform(*ec.path_offset_range)) if ec.randomize_path else 0.0
        step = max(ec.track_sample_step_m, 0.01)

        if ec.track_type == "straight":
            self._track = TrackGeometry.straight(ec.track_length_m, step, theta, offset)
        elif ec.track_type == "curve":
            amp = float(rng.uniform(*ec.curve_amplitude_range))
            wl = float(rng.uniform(*ec.curve_wavelength_range))
            phase = float(rng.uniform(*ec.curve_phase_range))
            self._track = TrackGeometry.curve(ec.track_length_m, step, theta, offset, amp, wl, phase)
        elif ec.track_type == "arc":
            radius = float(rng.uniform(*ec.arc_radius_range))
            sweep = float(rng.uniform(*ec.arc_sweep_range))
            sign = int(rng.choice([-1, 1]))
            self._track = TrackGeometry.arc(step, theta, offset, radius, sign, sweep)
        elif ec.track_type == "s_curve":
            amp = float(rng.uniform(*ec.s_curve_amplitude_range))
            length_m = float(rng.uniform(*ec.s_curve_length_range))
            self._track = TrackGeometry.s_curve(length_m, step, theta, offset, amp)
        else:
            count_min = min(ec.turn_sequence_turn_count_min, ec.turn_sequence_turn_count_max)
            count_max = max(ec.turn_sequence_turn_count_min, ec.turn_sequence_turn_count_max)
            turn_count = int(rng.integers(count_min, count_max + 1))
            self._track = TrackGeometry.turn_sequence(
                rng,
                step,
                theta,
                offset,
                turn_count,
                ec.turn_sequence_straight_range,
                ec.turn_sequence_radius_range,
                ec.turn_sequence_angle_range,
            )

    def _episode_track_summary(self) -> str:
        md = self._track.metadata
        t = self._track.track_type
        if t == "straight":
            return f"track=straight theta={md['theta']:.3f} offset={md['offset']:.3f}"
        if t == "curve":
            return (
                f"track=curve amp={md['curve_amplitude']:.3f} "
                f"wl={md['curve_wavelength']:.3f} theta={md['theta']:.3f}"
            )
        if t == "arc":
            return f"track=arc radius={md['arc_radius']:.3f} turn={md['arc_turn']} sweep={md['arc_sweep_angle']:.3f}"
        if t == "s_curve":
            return f"track=s_curve amp={md['s_curve_amplitude']:.3f} len={md['s_curve_length_m']:.3f}"
        return f"track=turn_sequence turns={md['turn_count']}"

    def _apply_track_visuals(self) -> None:
        floor_level = float(np.clip(self._ep_ir_white, 0.0, 1.0))
        line_level = float(np.clip(self._ep_ir_black, 0.0, 1.0))
        self._model.geom_rgba[self._geom_floor] = np.array([floor_level, floor_level, floor_level, 1.0])

        if self._line_geom_ids:
            if self._geom_line >= 0:
                self._model.geom_pos[self._geom_line] = [0.0, 0.0, -5.0]
                self._model.geom_rgba[self._geom_line] = [line_level, line_level, line_level, 0.0]
            pairs = self._track.resample_segment_pairs(min(len(self._line_geom_ids), TRACK_RENDER_SEGMENTS))
            for gid, (p0, p1) in zip(self._line_geom_ids, pairs):
                delta = p1 - p0
                seg_len = float(np.linalg.norm(delta))
                if seg_len < 1e-6:
                    self._model.geom_pos[gid] = [0.0, 0.0, -5.0]
                    self._model.geom_rgba[gid] = [line_level, line_level, line_level, 0.0]
                    continue
                mid = 0.5 * (p0 + p1)
                yaw = math.atan2(delta[1], delta[0])
                self._model.geom_size[gid] = [0.5 * seg_len, self._ep_line_hw, TRACK_RENDER_THICKNESS]
                self._model.geom_pos[gid] = [float(mid[0]), float(mid[1]), TRACK_RENDER_THICKNESS]
                self._model.geom_quat[gid] = _yaw_quat(yaw)
                self._model.geom_rgba[gid] = [line_level, line_level, line_level, 1.0]
            for gid in self._line_geom_ids[len(pairs):]:
                self._model.geom_pos[gid] = [0.0, 0.0, -5.0]
                self._model.geom_rgba[gid] = [line_level, line_level, line_level, 0.0]
        elif self._geom_line >= 0:
            pairs = self._track.resample_segment_pairs(1)
            if pairs:
                p0, p1 = pairs[0]
                delta = p1 - p0
                seg_len = float(np.linalg.norm(delta))
                mid = 0.5 * (p0 + p1)
                yaw = math.atan2(delta[1], delta[0])
                self._model.geom_size[self._geom_line] = [0.5 * seg_len, self._ep_line_hw, TRACK_RENDER_THICKNESS]
                self._model.geom_pos[self._geom_line] = [float(mid[0]), float(mid[1]), TRACK_RENDER_THICKNESS]
                self._model.geom_quat[self._geom_line] = _yaw_quat(yaw)
                self._model.geom_rgba[self._geom_line] = [line_level, line_level, line_level, 1.0]

    def _pace_human_render(self) -> None:
        if self.render_mode != "human" or self._render_period_s is None:
            return
        now = time.perf_counter()
        if self._last_render_t is not None:
            delay = self._render_period_s - (now - self._last_render_t)
            if delay > 0.0:
                time.sleep(delay)
        self._last_render_t = time.perf_counter()

    def _robot_pose_world(self) -> tuple[float, float, float]:
        mujoco.mj_kinematics(self._model, self._data)
        xpos = self._data.xpos[self._body_id]
        xmat = self._data.xmat[self._body_id].reshape(3, 3)
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])
        return float(xpos[0]), float(xpos[1]), float(yaw)

    def _robot_world_velocity(self) -> np.ndarray:
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self._model, self._data, mujoco.mjtObj.mjOBJ_BODY, self._body_id, vel, 0)
        return vel[3:6].copy()

    def _set_robot_pose(self, x: float, y: float, yaw: float, z: float = 0.05) -> None:
        qaddr = self._model.joint("root").qposadr[0]
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        self._data.qpos[qaddr:qaddr + 7] = [x, y, z, qw, 0.0, 0.0, qz]
        self._data.qvel[:] = 0.0
        mujoco.mj_kinematics(self._model, self._data)

    def _reset_pose_valid(self, ir: np.ndarray) -> bool:
        darkness = np.maximum(0.0, 1.0 - ir)
        min_total = self.ec.ir_strength_eps * self.ec.n_ir_sensors * self.ec.reset_min_strength_mult
        min_single = self.ec.ir_strength_eps * self.ec.reset_min_single_mult
        return float(np.sum(darkness)) >= min_total and float(np.max(darkness)) >= min_single

    def _reference_track_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        s_margin = min(0.4, 0.15 * self._track.length)
        if self._track.length <= 2.0 * s_margin:
            s = 0.5 * self._track.length
        else:
            s = float(self.np_random.uniform(s_margin, self._track.length - s_margin))
        point, tangent = self._track.point_and_tangent_at_s(s)
        return point, tangent, s

    def _place_robot(self) -> None:
        point, tangent, ref_s = self._reference_track_pose()
        path_yaw = math.atan2(tangent[1], tangent[0])
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        for _ in range(self.ec.reset_tries):
            along = float(self.np_random.uniform(-self.ec.reset_along_m, self.ec.reset_along_m))
            perp = float(self.np_random.uniform(-self.ec.reset_perp_m, self.ec.reset_perp_m))
            yaw_err = float(self.np_random.uniform(-self.ec.reset_yaw_rad, self.ec.reset_yaw_rad))
            x = float(point[0] + along * tangent[0] + perp * normal[0])
            y = float(point[1] + along * tangent[1] + perp * normal[1])
            self._set_robot_pose(x, y, path_yaw + yaw_err)
            ir = self._compute_ir(add_noise=False)
            if self._reset_pose_valid(ir):
                return

        for perp in self.ec.reset_fallback_perp:
            x = float(point[0] + perp * normal[0])
            y = float(point[1] + perp * normal[1])
            self._set_robot_pose(x, y, path_yaw)
            ir = self._compute_ir(add_noise=False)
            if self._reset_pose_valid(ir):
                return
        self._set_robot_pose(float(point[0]), float(point[1]), path_yaw)

    def _apply_domain_rand(self) -> None:
        self._wheel_vel_max = float(self.ec.wheel_vel_max)
        self._motor_force_scale = 1.0
        self._model.opt.gravity[:] = self._gravity_nominal
        self._model.body_mass[self._body_id] = self._base_mass_nominal

        left_dof = self._model.joint("left_wheel_joint").dofadr[0]
        right_dof = self._model.joint("right_wheel_joint").dofadr[0]
        self._model.dof_damping[left_dof] = self._wheel_damping_nominal[0]
        self._model.dof_damping[right_dof] = self._wheel_damping_nominal[1]
        for idx, name in enumerate(("left_wheel", "right_wheel")):
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self._model.geom_friction[gid, 0] = self._wheel_friction_nominal[idx]

        if not self.sim2real.domain_randomization:
            self._domain_params = {
                "domain_randomization": False,
                "mass_scale": 1.0,
                "gravity_z": float(self._model.opt.gravity[2]),
                "wheel_friction": self._wheel_friction_nominal.tolist(),
                "wheel_damping": self._wheel_damping_nominal.tolist(),
                "motor_force_scale": 1.0,
                "wheel_vel_max": float(self._wheel_vel_max),
            }
            return

        rng = self.np_random
        cfg = self.sim2real
        mass_scale = float(rng.uniform(*cfg.mass_scale_range))
        gravity_z = float(rng.uniform(*cfg.gravity_z_range))
        wheel_friction = float(rng.uniform(*cfg.lateral_friction_range))
        wheel_damping = float(rng.uniform(*cfg.wheel_damping_range))
        self._motor_force_scale = float(rng.uniform(*cfg.motor_force_scale_range))
        self._wheel_vel_max = float(self.ec.wheel_vel_max) * float(rng.uniform(*cfg.wheel_vel_max_scale_range))

        self._model.body_mass[self._body_id] = self._base_mass_nominal * mass_scale
        self._model.opt.gravity[2] = gravity_z
        self._model.dof_damping[left_dof] = wheel_damping
        self._model.dof_damping[right_dof] = wheel_damping
        for name in ("left_wheel", "right_wheel"):
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self._model.geom_friction[gid, 0] = wheel_friction

        self._domain_params = {
            "domain_randomization": True,
            "mass_scale": mass_scale,
            "gravity_z": gravity_z,
            "wheel_friction": [wheel_friction, wheel_friction],
            "wheel_damping": [wheel_damping, wheel_damping],
            "motor_force_scale": float(self._motor_force_scale),
            "wheel_vel_max": float(self._wheel_vel_max),
        }

    def episode_metadata(self) -> dict[str, Any]:
        return {
            "episode_index": int(self._episode_idx),
            "config_digest": self._config_digest,
            "track": dict(self._track.metadata),
            "scene": dict(self._scene_params),
            "domain": dict(self._domain_params),
        }

    def config_snapshot(self) -> dict[str, Any]:
        return {
            "config_digest": self._config_digest,
            "env_config": asdict(self.ec),
            "sim2real": asdict(self.sim2real),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)

        self._action_history = []
        self._step_count = 0
        self._line_lost_ctr = 0
        self._prev_fwd_speed = 0.0
        self._last_cmd[:] = 0.0
        self._prev_cmd[:] = 0.0
        self._omega_left = 0.0
        self._omega_right = 0.0
        self._motor_u_left = 0.0
        self._motor_u_right = 0.0
        self._episode_idx += 1
        self._episode_return = 0.0
        self._last_render_t = None
        self._last_rendered_frame = None

        self._apply_domain_rand()
        self._sample_scene()
        self._sample_track()
        self._apply_track_visuals()
        self._place_robot()

        mujoco.mj_forward(self._model, self._data)
        self._last_ir = self._compute_ir(add_noise=False)
        self._prev_line_strength = self._line_strength(self._last_ir)
        obs = self._build_obs(self._last_ir)
        info = {
            "track_type": self._track.track_type,
            "episode_metadata": self.episode_metadata(),
            "config_digest": self._config_digest,
        }
        if self._verbose_episode:
            print(
                f"[LineFollowEnvMuJoCo] episode {self._episode_idx} START | "
                f"{self._episode_track_summary()} seed={seed!r} digest={self._config_digest}",
                flush=True,
            )
        return obs, info

    def step(self, action: np.ndarray):
        cmd = self._motor_cmd(np.asarray(action, np.float32))
        self._last_cmd = cmd.copy()
        cmd_delta = self._last_cmd - self._prev_cmd
        cmd_jitter = float(np.dot(cmd_delta, cmd_delta))
        self._prev_cmd = self._last_cmd.copy()

        self._apply_drive_command(cmd)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self._model, self._data)

        self._last_ir = self._compute_ir(add_noise=True)
        lat = self._lateral_norm(self._last_ir)
        line_str = self._line_strength(self._last_ir)

        rx, ry, _ = self._robot_pose_world()
        _, _, tx, ty, _, _ = self._track.nearest_point_and_tangent(rx, ry)
        world_vel = self._robot_world_velocity()
        v_progress = float(world_vel[0] * tx + world_vel[1] * ty)
        forward_accel = v_progress - self._prev_fwd_speed
        self._prev_fwd_speed = v_progress

        weak_thr = self.ec.ir_strength_eps * self.ec.n_ir_sensors * self.ec.ir_lost_strength_mult
        if line_str < weak_thr:
            self._line_lost_ctr += 1
        else:
            self._line_lost_ctr = 0

        lat_fail = abs(lat) > self.ec.lateral_term_thresh
        lost_fail = self._line_lost_ctr >= self.ec.line_lost_steps
        terminated = bool(lat_fail or lost_fail)

        vis_gate = float(np.clip(line_str / max(self.ec.line_visibility_target, 1e-6), 0.0, 1.0))
        align_gate = max(0.0, 1.0 - abs(lat))
        line_recovery = max(0.0, line_str - self._prev_line_strength)
        self._prev_line_strength = line_str

        r_track = -self.ec.lateral_weight * lat ** 2
        r_prog = self.ec.progress_weight * v_progress * vis_gate * align_gate
        r_accel = self.ec.accel_weight * max(0.0, forward_accel) * vis_gate * align_gate
        r_rec = self.ec.recovery_weight * line_recovery
        r_alive = self.ec.alive_bonus
        p_cmd = self.ec.cmd_penalty * float(cmd[0] ** 2 + cmd[1] ** 2)
        p_jit = self.ec.jitter_penalty * cmd_jitter
        p_bad = self.ec.bad_track_penalty if abs(lat) > self.ec.bad_track_threshold else 0.0
        p_term = (self.ec.line_lost_penalty if lost_fail else 0.0) + (
            self.ec.lateral_terminal_penalty if lat_fail else 0.0
        )
        reward = float(r_track + r_prog + r_accel + r_rec + r_alive - p_cmd - p_jit - p_bad - p_term)

        self._step_count += 1
        truncated = self._step_count >= self.ec.max_episode_steps
        self._episode_return += reward

        obs = self._build_obs(self._last_ir)
        if self.render_mode == "human":
            self.render()

        reason = None
        if truncated and not terminated:
            reason = "max_steps"
        elif lat_fail and lost_fail:
            reason = "lateral+line_lost"
        elif lat_fail:
            reason = "lateral"
        elif lost_fail:
            reason = "line_lost"

        info = {
            "lateral_norm": float(lat),
            "v_progress": float(v_progress),
            "forward_accel": float(forward_accel),
            "cmd_jitter": float(cmd_jitter),
            "line_strength": float(line_str),
            "line_recovery": float(line_recovery),
            "vis_gate": float(vis_gate),
            "align_gate": float(align_gate),
            "line_lost_count": int(self._line_lost_ctr),
            "termination_reason": reason,
            "episode_steps": int(self._step_count),
            "episode_return": float(self._episode_return),
            "episode_metadata": self.episode_metadata(),
            "config_digest": self._config_digest,
        }
        if self._verbose_episode and (terminated or truncated):
            print(
                f"[LineFollowEnvMuJoCo] episode {self._episode_idx} END | "
                f"steps={self._step_count} return={self._episode_return:.3f} "
                f"reason={reason} |lat|={abs(lat):.3f} line_strength={line_str:.3f} "
                f"line_lost_count={self._line_lost_ctr}",
                flush=True,
            )
        return obs, reward, terminated, truncated, info

    def _build_obs(self, ir: np.ndarray) -> np.ndarray:
        wl = float(self._data.qvel[self._model.joint("left_wheel_joint").dofadr[0]])
        wr = float(self._data.qvel[self._model.joint("right_wheel_joint").dofadr[0]])
        wmx = max(self._wheel_vel_max, 1e-6)
        return np.concatenate(
            [
                ir[:self.ec.n_ir_sensors].astype(np.float32),
                np.array(
                    [
                        float(np.clip(wl / wmx, -1.0, 1.0)),
                        float(np.clip(wr / wmx, -1.0, 1.0)),
                        float(self._last_cmd[0]),
                        float(self._last_cmd[1]),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            if self._human_render_backend is None:
                try:
                    import mujoco.viewer as mj_viewer

                    self._viewer = mj_viewer.launch_passive(
                        self._model,
                        self._data,
                        show_left_ui=False,
                        show_right_ui=False,
                    )
                    self._human_render_backend = "viewer"
                except Exception:
                    self._human_render_backend = "opencv"
            self._pace_human_render()
            if self._human_render_backend == "viewer" and self._viewer is not None:
                is_running = getattr(self._viewer, "is_running", None)
                if callable(is_running) and not self._viewer.is_running():
                    return None
                self._viewer.sync()

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        cam_names = [self._model.cam(i).name for i in range(self._model.ncam)]
        camera = "track" if "track" in cam_names else -1
        self._renderer.update_scene(self._data, camera=camera)
        img = self._renderer.render()
        self._last_rendered_frame = img.copy()
        if self.render_mode == "human":
            try:
                import cv2

                cv2.imshow("LineFollow-MuJoCo", img[:, :, ::-1])
                cv2.waitKey(1)
            except Exception:
                pass
        return img

    def last_rendered_frame(self) -> Optional[np.ndarray]:
        return None if self._last_rendered_frame is None else self._last_rendered_frame.copy()

    def close(self) -> None:
        if self._viewer is not None:
            close = getattr(self._viewer, "close", None)
            if callable(close):
                close()
            self._viewer = None
        self._human_render_backend = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass

    @staticmethod
    def obs_from_hardware(
        ir_readings: np.ndarray,
        omega_left: float,
        omega_right: float,
        last_cmd: np.ndarray,
        wheel_vel_max: float = WHEEL_VEL_MAX,
    ) -> np.ndarray:
        wmx = max(wheel_vel_max, 1e-6)
        return np.concatenate(
            [
                ir_readings.astype(np.float32),
                np.array(
                    [
                        float(np.clip(omega_left / wmx, -1.0, 1.0)),
                        float(np.clip(omega_right / wmx, -1.0, 1.0)),
                        float(last_cmd[0]),
                        float(last_cmd[1]),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
