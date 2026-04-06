"""
PyBullet line-following env (Gymnasium): infinite straight path in the XY plane.
Default: fixed world line along +x with y=0 (theta=0, offset=0).
Robot starts on/near the strip with small lateral/yaw deviation so IR always has partial line in view.
Set `randomize_path=True` to sample a new line heading and perpendicular offset each episode.

Robot (`robots/diff_drive.urdf`): differential drive — no IMU.
  Policy actions [-1,1]^2 drive left/right rear wheels. `Sim2RealConfig.motor_dynamics`:
  - "accel" (default): actions are per-wheel angular acceleration (legacy).
  - "pwm_first_order": actions as normalized PWM/duty (−1..1); deadband, slew, optional PWM
    quantization, then first-order lag to wheel ω (closer to DC motor + gearbox + H-bridge than
    instant velocity). Real Raspberry Pi + L298N uses PWM duty, not PyBullet velocity control.

Optional JSON overrides: merge with `Sim2RealConfig.merge_json(path)` (see field names on the dataclass).

Observations: IR reflectances + normalized left/right wheel speeds + normalized delayed commands.
IR readings use a simple real-sensor chain: ideal scene reflectance → photodiode → noise → ADC
→ optional comparator. Reward uses only sensor-side signals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent
URDF_PATH = ROOT / "robots" / "diff_drive.urdf"

# Physics
CONTROL_FREQUENCY = 50.0
TIMESTEP = 1.0 / 240.0
SUBSTEPS = max(1, int(round((1.0 / CONTROL_FREQUENCY) / TIMESTEP)))
WHEEL_VEL_MAX = 12.0
MAX_MOTOR_FORCE = 5.0
# Per-wheel angular acceleration (rad/s^2) when |action|=1; integrated at CONTROL_FREQUENCY
MAX_WHEEL_ANG_ACCEL = 32.0
FRONT_CASTER_ROLL_FORCE = 1.5

# Reward (IR + motor command penalty; no ground-truth path pose)
ALIVE_BONUS = 0.02
IR_SENSOR_LATERAL_WEIGHT = 1.2
IR_PROGRESS_WEIGHT = 0.35
WHEEL_CMD_PENALTY_WEIGHT = 0.015  # squared normalized delayed actions (smoothness)
# Lateral termination uses strict `abs(lat) > IR_TERMINATE_LATERAL_NORM` in step(). With default 1.0,
# lateral done is off: for 2 IR sensors, whenever line strength is high enough for reset, the
# centroid lateral signal is typically saturated at |lat|==1.0, so threshold 0.92 caused ep_len=1.
IR_TERMINATE_LATERAL_NORM = 1.0
# Line-lost termination: more tolerant = lower threshold for "weak" and/or more consecutive steps.
IR_LOST_CONSECUTIVE_STEPS = 8
IR_LINE_STRENGTH_EPS = 0.04
# step() counts a timestep toward line-lost only if line_strength < eps * n_sensors * this factor (<1 is more tolerant).
IR_LINE_LOST_STRENGTH_MULT = 0.5

# Reset: start on/near the strip with small pose noise; reject poses where IR sees no line.
# Perpendicular search must be wide enough that a narrow strip can intersect the IR row (sensor span > strip width).
RESET_ALONG_RANGE_M = 0.18
RESET_MAX_PERP_M = 0.14
RESET_MAX_YAW_ERR_RAD = 0.38
RESET_POSE_TRIES = 60
RESET_MIN_LINE_STRENGTH_MULT = 2.0
RESET_FALLBACK_PERP_M = (0.0, 0.06, -0.06, 0.1, -0.1, 0.12, -0.12)

# Visual line strip (world-aligned box; half-width perpendicular to path)
STRIP_HALF_LENGTH = 3.0
DEFAULT_LINE_HALF_WIDTH = 0.025

# IR: default row under front of chassis (body frame); y-span spaces sensors laterally (e.g. 2 = left/right)
DEFAULT_IR_SENSOR_X = 0.11
DEFAULT_IR_SENSOR_Y_SPAN = 0.18

# PyBullet GUI: fixed overview camera (world frame; does not chase the robot)
DEFAULT_FIXED_CAM_DISTANCE = 2.8
DEFAULT_FIXED_CAM_YAW_DEG = 52.0
DEFAULT_FIXED_CAM_PITCH_DEG = -30.0
DEFAULT_FIXED_CAM_TARGET = (0.0, 0.0, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


@dataclass
class Sim2RealConfig:
    domain_randomization: bool = False
    mass_scale_range: tuple[float, float] = (0.85, 1.15)
    lateral_friction_range: tuple[float, float] = (0.5, 1.2)
    gravity_z_range: tuple[float, float] = (-10.2, -9.4)
    # Extra domain randomization (used when domain_randomization=True)
    motor_force_scale_range: tuple[float, float] = (0.8, 1.2)
    wheel_vel_max_scale_range: tuple[float, float] = (0.85, 1.15)
    wheel_joint_damping_range: tuple[float, float] = (0.0, 0.3)
    action_delay_steps: int = 0
    # Motor dynamics: "accel" = legacy angular-accel integration; "pwm_first_order" = duty + lag
    motor_dynamics: Literal["accel", "pwm_first_order"] = "accel"
    motor_deadband: float = 0.0
    motor_slew_rate_per_s: float = 8.0
    motor_time_constant_s: float = 0.08
    pwm_discrete_levels: int = 0
    # IR: calibrated black/white reflectance (scene → linear “photocurrent” before front-end)
    ir_reflectance_black: float = 0.12
    ir_reflectance_white: float = 0.92
    ir_edge_scale: float = 0.004
    # IR front-end (mimics reflective sensor module: photodiode + amp + noise + ADC)
    ir_photodiode_gamma: float = 0.93
    ir_noise_std: float = 0.0
    ir_adc_bits: int = 10
    ir_digital_output: bool = False
    ir_comparator_level: float = 0.5

    @staticmethod
    def merge_json(path: Path | str, base: Optional["Sim2RealConfig"] = None) -> "Sim2RealConfig":
        """Load JSON object and merge into `base` (default: new Sim2RealConfig). Unknown keys ignored."""
        cfg = Sim2RealConfig() if base is None else base
        p = Path(path)
        with p.open(encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Sim2Real JSON must be an object: {p}")
        allowed = {f.name for f in fields(Sim2RealConfig)}
        kwargs: dict[str, Any] = {}
        for k, v in raw.items():
            if k.startswith("_") or k not in allowed:
                continue
            if k in (
                "mass_scale_range",
                "lateral_friction_range",
                "gravity_z_range",
                "motor_force_scale_range",
                "wheel_vel_max_scale_range",
                "wheel_joint_damping_range",
            ):
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    kwargs[k] = (float(v[0]), float(v[1]))
                else:
                    continue
            elif k == "motor_dynamics":
                if v in ("accel", "pwm_first_order"):
                    kwargs[k] = v
            elif k == "domain_randomization" or k == "ir_digital_output":
                kwargs[k] = bool(v)
            elif k == "action_delay_steps" or k == "pwm_discrete_levels" or k == "ir_adc_bits":
                kwargs[k] = int(v)
            else:
                kwargs[k] = float(v) if isinstance(v, (int, float)) else v
        return replace(cfg, **kwargs)


class LineFollowEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        sim2real: Optional[Sim2RealConfig] = None,
        *,
        randomize_path: bool = False,
        path_theta_range: tuple[float, float] = (-np.pi, np.pi),
        path_offset_range: tuple[float, float] = (-0.15, 0.15),
        n_ir_sensors: int = 2,
        line_half_width: float = DEFAULT_LINE_HALF_WIDTH,
        ir_sensor_x_body: float = DEFAULT_IR_SENSOR_X,
        ir_sensor_y_span: float = DEFAULT_IR_SENSOR_Y_SPAN,
        gui_camera_mode: Literal["fixed", "follow"] = "fixed",
        fixed_camera_distance: float = DEFAULT_FIXED_CAM_DISTANCE,
        fixed_camera_yaw_deg: float = DEFAULT_FIXED_CAM_YAW_DEG,
        fixed_camera_pitch_deg: float = DEFAULT_FIXED_CAM_PITCH_DEG,
        fixed_camera_target: tuple[float, float, float] = DEFAULT_FIXED_CAM_TARGET,
        show_ir_gui: bool = True,
        verbose_episode: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.sim2real = sim2real or Sim2RealConfig()
        self.randomize_path = randomize_path
        self.path_theta_range = path_theta_range
        self.path_offset_range = path_offset_range
        self.n_ir_sensors = max(1, int(n_ir_sensors))
        self.line_half_width = float(line_half_width)
        ys = np.linspace(
            -0.5 * ir_sensor_y_span,
            0.5 * ir_sensor_y_span,
            self.n_ir_sensors,
            dtype=np.float64,
        )
        self._ir_sensor_body = np.stack(
            [np.full(self.n_ir_sensors, ir_sensor_x_body, dtype=np.float64), ys, np.zeros(self.n_ir_sensors)],
            axis=1,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        # IR [0,1] + ω_left, ω_right, cmd_left, cmd_right in [-1, 1] (normalized; cmds = delayed actions)
        lo = np.concatenate(
            [
                np.zeros(self.n_ir_sensors, dtype=np.float32),
                np.full(4, -1.0, dtype=np.float32),
            ]
        )
        hi = np.concatenate(
            [
                np.ones(self.n_ir_sensors, dtype=np.float32),
                np.ones(4, dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(low=lo, high=hi, dtype=np.float32)

        self._client: Optional[int] = None
        self._robot: Optional[int] = None
        self._plane: Optional[int] = None
        self._rear_wheel_joints: list[int] = []
        self._front_caster_joint: int = -1
        self._omega_left = 0.0
        self._omega_right = 0.0
        self._last_cmd = np.zeros(2, dtype=np.float64)
        self._step_count = 0
        self._path_theta = 0.0
        self._path_offset = 0.0
        self._action_history: list[np.ndarray] = []
        self._debug_line_id: Optional[int] = None
        self._last_view_matrix: Optional[tuple] = None
        self._line_body_id: Optional[int] = None
        self._line_lost_counter = 0
        self.gui_camera_mode = gui_camera_mode
        self._fixed_cam_dist = float(fixed_camera_distance)
        self._fixed_cam_yaw = float(fixed_camera_yaw_deg)
        self._fixed_cam_pitch = float(fixed_camera_pitch_deg)
        self._fixed_cam_target = tuple(fixed_camera_target)
        self.show_ir_gui = bool(show_ir_gui)
        self._ir_gui_fig: Any = None
        self._verbose_episode = bool(verbose_episode)
        self._episode_idx = 0
        self._episode_return = 0.0
        self._wheel_vel_max = float(WHEEL_VEL_MAX)
        self._motor_force = float(MAX_MOTOR_FORCE)
        self._motor_u_left = 0.0
        self._motor_u_right = 0.0

        if not URDF_PATH.is_file():
            raise FileNotFoundError(f"Missing URDF: {URDF_PATH}")

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(TIMESTEP)

    def _configure_robot_joints(self, robot: int) -> None:
        """Cache joint indices for `diff_drive.urdf` (left/right rear driven; front caster passive)."""
        name_to_idx: dict[str, int] = {}
        for j in range(p.getNumJoints(robot)):
            name_to_idx[p.getJointInfo(robot, j)[1].decode("utf-8")] = j

        need = ("left_rear_wheel_joint", "right_rear_wheel_joint", "front_caster_joint")
        missing = [n for n in need if n not in name_to_idx]
        if missing:
            raise RuntimeError(
                f"{URDF_PATH.name} must define joints {need}; missing: {missing}"
            )
        self._rear_wheel_joints = [
            name_to_idx["left_rear_wheel_joint"],
            name_to_idx["right_rear_wheel_joint"],
        ]
        self._front_caster_joint = int(name_to_idx["front_caster_joint"])

    def _rear_wheel_velocities(self) -> tuple[float, float]:
        """Encoder-like: left/right rear wheel angular velocities (rad/s) from simulation."""
        if self._robot is None or len(self._rear_wheel_joints) < 2:
            return 0.0, 0.0
        jl, jr = self._rear_wheel_joints
        wl = float(p.getJointState(self._robot, jl)[1])
        wr = float(p.getJointState(self._robot, jr)[1])
        return wl, wr

    def _sample_path(self) -> None:
        r = self.np_random
        if self.randomize_path:
            lo, hi = self.path_theta_range
            self._path_theta = float(r.uniform(lo, hi))
            olo, ohi = self.path_offset_range
            self._path_offset = float(r.uniform(olo, ohi))
        else:
            self._path_theta = 0.0
            self._path_offset = 0.0

    def _path_normal(self) -> tuple[float, float]:
        t = self._path_theta
        return -float(np.sin(t)), float(np.cos(t))

    def _path_tangent(self) -> tuple[float, float]:
        t = self._path_theta
        return float(np.cos(t)), float(np.sin(t))

    def _rotation_matrix_body_to_world(self, orn: tuple[float, float, float, float]) -> np.ndarray:
        m = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float64).reshape(3, 3)
        return m

    def _reflectance_from_signed_dist(self, d: np.ndarray) -> np.ndarray:
        """Map signed perpendicular distance (m) to [0,1] reflectance; black strip near d=0."""
        cfg = self.sim2real
        w = self.line_half_width
        sigma = max(cfg.ir_edge_scale, 1e-6)
        u = np.abs(d)
        t = _sigmoid((w - u) / sigma)
        r_lo = cfg.ir_reflectance_black
        r_hi = cfg.ir_reflectance_white
        return r_hi * (1.0 - t) + r_lo * t

    def _compute_ir_reflectance(self, add_noise: bool) -> np.ndarray:
        """IR reflectance [0,1] per sensor; optional Gaussian sensor noise (simulates real IR)."""
        assert self._robot is not None
        pos, orn = p.getBasePositionAndOrientation(self._robot)
        r_mat = self._rotation_matrix_body_to_world(orn)
        p0 = np.array(pos, dtype=np.float64)
        nx, ny = self._path_normal()
        pts = (self._ir_sensor_body @ r_mat.T) + p0
        d = nx * pts[:, 0] + ny * pts[:, 1] - self._path_offset
        raw = self._reflectance_from_signed_dist(d.astype(np.float64))
        raw = self._apply_ir_sensor_frontend(raw, add_noise=add_noise)
        return raw.astype(np.float64)

    def _ir_adc_quantize_physical(self, x: np.ndarray, lo: float, hi: float, bits: int) -> np.ndarray:
        """Uniform ADC over the calibrated black–white swing (typical MCU read of sensor voltage)."""
        if bits <= 0:
            return x
        u = (np.clip(x, lo, hi) - lo) / max(hi - lo, 1e-9)
        levels = 2**bits
        idx = np.round(u * (levels - 1.0)).astype(np.int64)
        idx = np.clip(idx, 0, levels - 1)
        uq = idx.astype(np.float64) / (levels - 1.0)
        return lo + uq * (hi - lo)

    def _apply_ir_sensor_frontend(self, raw: np.ndarray, *, add_noise: bool) -> np.ndarray:
        """Scene linear reflectance → photodiode (gamma) → electronics noise → ADC → optional comparator."""
        cfg = self.sim2real
        lo, hi = cfg.ir_reflectance_black, cfg.ir_reflectance_white
        u = (np.clip(raw, lo, hi) - lo) / max(hi - lo, 1e-9)
        g = max(float(cfg.ir_photodiode_gamma), 1e-6)
        u = np.power(u, g)
        out = lo + u * (hi - lo)
        if add_noise and cfg.ir_noise_std > 0.0:
            out = out + self.np_random.normal(0.0, cfg.ir_noise_std, size=out.shape).astype(np.float64)
        out = np.clip(out, 0.0, 1.0)
        if cfg.ir_adc_bits > 0:
            out = self._ir_adc_quantize_physical(out, lo, hi, int(cfg.ir_adc_bits))
        if cfg.ir_digital_output:
            thr = lo + float(cfg.ir_comparator_level) * (hi - lo)
            out = np.where(out >= thr, hi, lo).astype(np.float64)
        return out

    def _lateral_norm_from_ir(self, r: np.ndarray) -> float:
        """Lateral error from IR only: multi-sensor uses line centroid vs array center (~[-1,1]).
        Single sensor: [0,1] off-track score (0 = line under sensor, 1 = white / lost)."""
        r = np.asarray(r, dtype=np.float64).ravel()
        n = len(r)
        r_hi = self.sim2real.ir_reflectance_white
        r_lo = self.sim2real.ir_reflectance_black
        if n == 1:
            denom = max(r_hi - r_lo, 1e-6)
            strength = float(np.clip((r_hi - r[0]) / denom, 0.0, 1.0))
            return 1.0 - strength
        if n < 2:
            return 0.0
        w = np.maximum(0.0, r_hi - r)
        s = float(np.sum(w))
        if s < IR_LINE_STRENGTH_EPS:
            return 0.0
        idx = np.arange(n, dtype=np.float64)
        c = float(np.sum(w * idx) / s)
        mid = 0.5 * float(n - 1)
        half = max(0.5 * float(n - 1), 1e-6)
        return float((c - mid) / half)

    def _body_linear_velocity(self, orn: tuple[float, float, float, float]) -> np.ndarray:
        lin_w, _ang_w = p.getBaseVelocity(self._robot)
        r_mat = self._rotation_matrix_body_to_world(orn)
        v_w = np.array(lin_w, dtype=np.float64)
        return r_mat.T @ v_w

    def _get_ir_obs(self) -> np.ndarray:
        """IR + normalized left/right ω + last applied normalized motor commands (after delay)."""
        assert self._robot is not None
        r = self._compute_ir_reflectance(add_noise=True)
        wl, wr = self._rear_wheel_velocities()
        n = max(self._wheel_vel_max, 1e-6)
        return np.concatenate(
            [
                r.astype(np.float32),
                np.array(
                    [
                        float(np.clip(wl / n, -1.0, 1.0)),
                        float(np.clip(wr / n, -1.0, 1.0)),
                        float(self._last_cmd[0]),
                        float(self._last_cmd[1]),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    def _remove_line_strip(self) -> None:
        if self._line_body_id is not None:
            try:
                p.removeBody(self._line_body_id)
            except Exception:
                pass
            self._line_body_id = None

    def _spawn_line_strip(self) -> None:
        self._remove_line_strip()
        nx, ny = self._path_normal()
        fx = self._path_offset * nx
        fy = self._path_offset * ny
        z = 0.001
        quat = p.getQuaternionFromEuler([0.0, 0.0, self._path_theta])
        hl = STRIP_HALF_LENGTH
        hw = self.line_half_width
        hz = 0.0015
        # Use a real collision shape (not -1): visual-only multibodies can fail on second
        # createMultiBody after resetSimulation() on some GUI/Metal builds; keep behavior
        # identical by disabling collisions for this decorative strip.
        he = [hl, hw, hz]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
        viz = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=he,
            rgbaColor=[0.02, 0.02, 0.02, 1.0],
        )
        self._line_body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=viz,
            basePosition=[fx, fy, z + hz],
            baseOrientation=quat,
        )
        p.setCollisionFilterGroupMask(self._line_body_id, -1, 0, 0)

    def _style_plane_white(self) -> None:
        if self._plane is None:
            return
        for link in (-1, 0):
            try:
                p.changeVisualShape(self._plane, link, rgbaColor=[0.98, 0.98, 0.98, 1.0])
            except Exception:
                continue

    def _update_debug_visualizer_camera(self) -> None:
        """Set PyBullet main 3D view: fixed world overview or follow-chase."""
        if self.render_mode != "human" or self._robot is None:
            return

        if self.gui_camera_mode == "fixed":
            tgt = list(self._fixed_cam_target)
            p.resetDebugVisualizerCamera(
                self._fixed_cam_dist,
                self._fixed_cam_yaw,
                self._fixed_cam_pitch,
                tgt,
            )
            return

        pos, orn = p.getBasePositionAndOrientation(self._robot)
        yaw = p.getEulerFromQuaternion(orn)[2]
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        dx, dy = float(np.cos(yaw)), float(np.sin(yaw))
        camera_eye = [px, py, pz + 0.2]
        camera_target = [px + dx, py + dy, pz]
        view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1])
        self._last_view_matrix = view_matrix
        delta = np.array(camera_eye, dtype=np.float64) - np.array(camera_target, dtype=np.float64)
        dist = float(np.linalg.norm(delta)) + 1e-9
        cam_yaw = float(np.degrees(np.arctan2(delta[1], delta[0])))
        cam_pitch = float(np.degrees(np.arctan2(delta[2], np.hypot(delta[0], delta[1]))))
        p.resetDebugVisualizerCamera(dist, cam_yaw, cam_pitch, camera_target)

    def _update_ir_gui(self, ir: np.ndarray) -> None:
        """Side window(s): matplotlib figure with bar chart + IR strip (policy input)."""
        if self.render_mode != "human" or not self.show_ir_gui:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        ir = np.asarray(ir, dtype=np.float32).ravel()[: self.n_ir_sensors]
        if self._ir_gui_fig is None:
            plt.ion()
            self._ir_gui_fig, axes = plt.subplots(
                1,
                2,
                figsize=(10, 3.2),
                gridspec_kw={"width_ratios": [1.1, 1.0]},
            )
            self._ir_gui_ax_bar, self._ir_gui_ax_strip = axes[0], axes[1]
            mgr = getattr(self._ir_gui_fig.canvas, "manager", None)
            if mgr is not None and hasattr(mgr, "set_window_title"):
                mgr.set_window_title("IR sensor input (policy observation)")
            self._ir_gui_fig.tight_layout()

        self._ir_gui_ax_bar.clear()
        n = len(ir)
        self._ir_gui_ax_bar.bar(np.arange(n), ir, color="steelblue", width=0.85)
        self._ir_gui_ax_bar.set_ylim(0.0, 1.0)
        self._ir_gui_ax_bar.set_xlabel("Sensor index")
        self._ir_gui_ax_bar.set_ylabel("Reflectance")
        self._ir_gui_ax_bar.set_title("IR array")

        self._ir_gui_ax_strip.clear()
        strip = ir.reshape(1, -1)
        self._ir_gui_ax_strip.imshow(
            strip,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            interpolation="nearest",
        )
        self._ir_gui_ax_strip.set_xticks(np.arange(n))
        self._ir_gui_ax_strip.set_yticks([])
        self._ir_gui_ax_strip.set_xlabel("Sensor (lateral)")
        self._ir_gui_ax_strip.set_title("IR strip")

        self._ir_gui_fig.canvas.draw_idle()
        self._ir_gui_fig.canvas.flush_events()
        plt.pause(0.001)

    def _rear_wheel_link_indices(self) -> tuple[int, int]:
        """Child link indices for rear wheels (URDF: base=0, links 1..J follow joint order)."""
        jl, jr = self._rear_wheel_joints
        return int(jl + 1), int(jr + 1)

    def _apply_domain_randomization(self) -> None:
        if self._robot is None:
            return
        r = self.np_random
        ll, lr = self._rear_wheel_link_indices()
        if not self.sim2real.domain_randomization:
            p.setGravity(0, 0, -9.81)
            self._wheel_vel_max = float(WHEEL_VEL_MAX)
            self._motor_force = float(MAX_MOTOR_FORCE)
            p.changeDynamics(self._robot, ll, jointDamping=0.0)
            p.changeDynamics(self._robot, lr, jointDamping=0.0)
            return
        mass_scale = r.uniform(*self.sim2real.mass_scale_range)
        lat_fric = r.uniform(*self.sim2real.lateral_friction_range)
        gz = r.uniform(*self.sim2real.gravity_z_range)
        mf_s = r.uniform(*self.sim2real.motor_force_scale_range)
        wv_s = r.uniform(*self.sim2real.wheel_vel_max_scale_range)
        damp = r.uniform(*self.sim2real.wheel_joint_damping_range)
        p.setGravity(0, 0, gz)
        self._motor_force = float(MAX_MOTOR_FORCE * mf_s)
        self._wheel_vel_max = float(WHEEL_VEL_MAX * wv_s)
        p.changeDynamics(self._robot, -1, mass=1.0 * mass_scale, lateralFriction=lat_fric)
        p.changeDynamics(self._robot, ll, jointDamping=float(damp))
        p.changeDynamics(self._robot, lr, jointDamping=float(damp))

    def _load_world(self) -> None:
        self._remove_line_strip()
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(TIMESTEP)
        self._plane = p.loadURDF("plane.urdf")
        self._style_plane_white()
        start_pos = [0.0, 0.0, 0.08]
        start_orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self._robot = p.loadURDF(
            str(URDF_PATH),
            start_pos,
            start_orn,
            useFixedBase=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        self._configure_robot_joints(self._robot)
        self._omega_left = 0.0
        self._omega_right = 0.0
        self._motor_u_left = 0.0
        self._motor_u_right = 0.0
        self._last_cmd = np.zeros(2, dtype=np.float64)
        self._apply_domain_randomization()
        caster_f = FRONT_CASTER_ROLL_FORCE * (self._motor_force / max(MAX_MOTOR_FORCE, 1e-9))
        wheel_js = self._rear_wheel_joints + [self._front_caster_joint]
        forces = [self._motor_force, self._motor_force, caster_f]
        for j, f in zip(wheel_js, forces):
            p.setJointMotorControl2(
                self._robot,
                j,
                p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=float(f),
            )

    def _line_strength_from_reflectance(self, r: np.ndarray) -> float:
        """Sum of (white - reading); same notion as in step() for 'line visible'."""
        r_hi = self.sim2real.ir_reflectance_white
        return float(np.sum(np.maximum(0.0, r_hi - np.asarray(r, dtype=np.float64))))

    def _place_robot_on_path(self) -> None:
        """Place base on/near the line with small random error; ensure IR sees part of the strip."""
        assert self._robot is not None
        rng = self.np_random
        nx, ny = self._path_normal()
        ux, uy = self._path_tangent()
        fx = self._path_offset * nx
        fy = self._path_offset * ny
        min_strength = IR_LINE_STRENGTH_EPS * max(1, self.n_ir_sensors) * RESET_MIN_LINE_STRENGTH_MULT
        z = 0.08

        def try_pose(s0: float, eps: float, yaw: float) -> bool:
            x = fx + s0 * ux + eps * nx
            y = fy + s0 * uy + eps * ny
            orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
            p.resetBasePositionAndOrientation(self._robot, [x, y, z], orn)
            p.resetBaseVelocity(self._robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            refl = self._compute_ir_reflectance(add_noise=False)
            return self._line_strength_from_reflectance(refl) >= min_strength

        for _ in range(RESET_POSE_TRIES):
            s0 = float(rng.uniform(-RESET_ALONG_RANGE_M, RESET_ALONG_RANGE_M))
            eps = float(rng.uniform(-RESET_MAX_PERP_M, RESET_MAX_PERP_M))
            yaw = float(self._path_theta + rng.uniform(-RESET_MAX_YAW_ERR_RAD, RESET_MAX_YAW_ERR_RAD))
            if try_pose(s0, eps, yaw):
                return

        yaw0 = float(self._path_theta)
        for eps in RESET_FALLBACK_PERP_M:
            if try_pose(0.0, float(eps), yaw0):
                return

        p.resetBasePositionAndOrientation(self._robot, [fx, fy, z], p.getQuaternionFromEuler([0.0, 0.0, yaw0]))
        p.resetBaseVelocity(self._robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def _draw_line_debug(self) -> None:
        if self.render_mode != "human":
            return
        if self._debug_line_id is not None:
            p.removeUserDebugItem(self._debug_line_id)
        nx, ny = self._path_normal()
        ux, uy = self._path_tangent()
        fx = self._path_offset * nx
        fy = self._path_offset * ny
        half_len = 1.5
        z = 0.02
        p0 = [fx - half_len * ux, fy - half_len * uy, z]
        p1 = [fx + half_len * ux, fy + half_len * uy, z]
        self._debug_line_id = p.addUserDebugLine(
            p0,
            p1,
            lineColorRGB=[0.2, 0.2, 0.2],
            lineWidth=2.0,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._ensure_client()
        self._action_history = []
        self._load_world()
        self._sample_path()
        self._spawn_line_strip()

        self._place_robot_on_path()
        self._omega_left = 0.0
        self._omega_right = 0.0
        self._last_cmd = np.zeros(2, dtype=np.float64)

        self._step_count = 0
        self._line_lost_counter = 0
        self._episode_return = 0.0
        self._episode_idx += 1
        self._draw_line_debug()
        self._update_debug_visualizer_camera()

        obs = self._get_ir_obs()
        self._update_ir_gui(obs)
        info = {
            "path_theta": self._path_theta,
            "path_offset": self._path_offset,
        }
        if self._verbose_episode:
            print(
                f"[LineFollowEnv] episode {self._episode_idx} START | "
                f"path_theta={self._path_theta:.3f} rad  path_offset={self._path_offset:.3f} m  "
                f"randomize_path={self.randomize_path}  seed={seed!r}",
                flush=True,
            )
        return obs, info

    def _motor_command(self, action: np.ndarray) -> np.ndarray:
        delay = self.sim2real.action_delay_steps
        self._action_history.append(np.clip(action.astype(np.float64), -1.0, 1.0))
        t = len(self._action_history) - 1
        if t < delay:
            cmd = np.zeros(2, dtype=np.float64)
        else:
            cmd = self._action_history[t - delay]
        return cmd

    def _motor_deadband_u(self, u: float) -> float:
        db = float(self.sim2real.motor_deadband)
        if abs(u) < db:
            return 0.0
        return float(u)

    def _motor_slew_u(self, u_prev: float, u_target: float) -> float:
        dt = 1.0 / CONTROL_FREQUENCY
        max_du = float(self.sim2real.motor_slew_rate_per_s) * dt
        du = float(np.clip(u_target - u_prev, -max_du, max_du))
        return float(np.clip(u_prev + du, -1.0, 1.0))

    def _quantize_pwm_u(self, u: float) -> float:
        n = int(self.sim2real.pwm_discrete_levels)
        if n <= 0:
            return float(u)
        u = float(np.clip(u, -1.0, 1.0))
        if n == 1:
            return 0.0
        s = 1.0 if u >= 0.0 else -1.0
        mag = abs(u)
        q = round(mag * (n - 1)) / max(n - 1, 1)
        return s * float(q)

    def _first_order_omega_toward(self, omega: float, omega_star: float) -> float:
        tau = max(float(self.sim2real.motor_time_constant_s), 1e-4)
        dt = 1.0 / CONTROL_FREQUENCY
        alpha = min(1.0, dt / tau)
        return float(omega + alpha * (omega_star - omega))

    def _apply_drive_command(self, cmd: np.ndarray) -> None:
        """Map actions in [-1,1]^2 to rear wheel velocities (accel integration or pwm_first_order)."""
        assert self._robot is not None
        wmx = self._wheel_vel_max
        mf = self._motor_force
        if self.sim2real.motor_dynamics == "accel":
            aL, aR = float(cmd[0]), float(cmd[1])
            dt = 1.0 / CONTROL_FREQUENCY
            self._omega_left += aL * MAX_WHEEL_ANG_ACCEL * dt
            self._omega_right += aR * MAX_WHEEL_ANG_ACCEL * dt
            self._omega_left = float(np.clip(self._omega_left, -wmx, wmx))
            self._omega_right = float(np.clip(self._omega_right, -wmx, wmx))
        else:
            uL_raw = float(np.clip(cmd[0], -1.0, 1.0))
            uR_raw = float(np.clip(cmd[1], -1.0, 1.0))
            uLd = self._motor_deadband_u(uL_raw)
            uRd = self._motor_deadband_u(uR_raw)
            self._motor_u_left = self._motor_slew_u(self._motor_u_left, uLd)
            self._motor_u_right = self._motor_slew_u(self._motor_u_right, uRd)
            uL = self._quantize_pwm_u(self._motor_u_left)
            uR = self._quantize_pwm_u(self._motor_u_right)
            oL_star = uL * wmx
            oR_star = uR * wmx
            self._omega_left = self._first_order_omega_toward(self._omega_left, oL_star)
            self._omega_right = self._first_order_omega_toward(self._omega_right, oR_star)
            self._omega_left = float(np.clip(self._omega_left, -wmx, wmx))
            self._omega_right = float(np.clip(self._omega_right, -wmx, wmx))

        jl, jr = self._rear_wheel_joints
        p.setJointMotorControl2(
            self._robot,
            jl,
            p.VELOCITY_CONTROL,
            targetVelocity=self._omega_left,
            force=mf,
        )
        p.setJointMotorControl2(
            self._robot,
            jr,
            p.VELOCITY_CONTROL,
            targetVelocity=self._omega_right,
            force=mf,
        )
        omega_front = 0.5 * (self._omega_left + self._omega_right)
        caster_f = FRONT_CASTER_ROLL_FORCE * (mf / max(MAX_MOTOR_FORCE, 1e-9))
        p.setJointMotorControl2(
            self._robot,
            self._front_caster_joint,
            p.VELOCITY_CONTROL,
            targetVelocity=omega_front,
            force=float(caster_f),
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._robot is not None
        cmd = self._motor_command(np.asarray(action, dtype=np.float32))
        self._last_cmd = cmd.astype(np.float64).copy()
        self._apply_drive_command(cmd)
        for _ in range(SUBSTEPS):
            p.stepSimulation()

        _, orn = p.getBasePositionAndOrientation(self._robot)

        v_body = self._body_linear_velocity(orn)

        r = self._compute_ir_reflectance(add_noise=True)
        lat = self._lateral_norm_from_ir(r)
        r_hi = self.sim2real.ir_reflectance_white
        line_strength = float(np.sum(np.maximum(0.0, r_hi - r)))
        weak_thr = (
            IR_LINE_STRENGTH_EPS
            * max(1, self.n_ir_sensors)
            * float(IR_LINE_LOST_STRENGTH_MULT)
        )
        if line_strength < weak_thr:
            self._line_lost_counter += 1
        else:
            self._line_lost_counter = 0

        reward = float(
            -IR_SENSOR_LATERAL_WEIGHT * abs(lat)
            + IR_PROGRESS_WEIGHT * float(v_body[0])
            - WHEEL_CMD_PENALTY_WEIGHT * float(cmd[0] ** 2 + cmd[1] ** 2)
            + ALIVE_BONUS
        )
        lat_fail = abs(lat) > IR_TERMINATE_LATERAL_NORM
        line_lost_fail = self._line_lost_counter >= IR_LOST_CONSECUTIVE_STEPS
        terminated = bool(lat_fail or line_lost_fail)

        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        self._episode_return += float(reward)

        obs = self._get_ir_obs()

        if self.render_mode == "human":
            self._update_debug_visualizer_camera()
            self._update_ir_gui(obs)

        if self._verbose_episode and (terminated or truncated):
            if truncated and not terminated:
                reason = "max_steps"
            elif lat_fail and line_lost_fail:
                reason = "lateral+line_lost"
            elif lat_fail:
                reason = "lateral"
            else:
                reason = "line_lost"
            print(
                f"[LineFollowEnv] episode {self._episode_idx} END   | "
                f"steps={self._step_count}  return={self._episode_return:.3f}  "
                f"terminated={terminated}  truncated={truncated}  reason={reason}  "
                f"|lat|={abs(lat):.3f}  line_lost_count={self._line_lost_counter}  "
                f"last_reward={float(reward):.4f}",
                flush=True,
            )

        return obs, float(reward), terminated, truncated, {}

    def render(self) -> None:
        if self.render_mode == "human":
            self._ensure_client()

    def close(self) -> None:
        if self._ir_gui_fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._ir_gui_fig)
            except Exception:
                pass
            self._ir_gui_fig = None
        if self._client is not None:
            p.disconnect(self._client)
            self._client = None
        self._robot = None
        self._plane = None
        self._line_body_id = None

