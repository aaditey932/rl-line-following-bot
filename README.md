# RL Line Following Bot

This repository trains a PPO policy for a differential-drive robot to follow a black line using IR-style reflectance observations. Training runs in simulation (MuJoCo by default, PyBullet as an alternative); you can deploy the saved policy on a Raspberry Pi with the wiring patterns documented below (Hack Lab–style hardware).

## Sim-to-real

### Platforms

The project spans **simulated and physical embodiment**. Training uses **MuJoCo** (primary) or **PyBullet** (alternative) to model a differential-drive robot whose observations stack IR-style reflectance with wheel speeds and delayed motor commands. The **physical** stack is a Raspberry Pi with L298N drive and reflective IR sensors, run via `pi_deploy.py`. The trained PPO policy is meant to transfer to that hardware when observations and low-level dynamics match what the agent saw in sim.

### Task: navigation vs grasping and balance

Line following belongs to **navigation**: stay on a marked path using onboard sensing, with rewards and termination tied to tracking the strip. **Grasping** in contrast emphasizes contact-rich manipulation—gripper pose, forces, and object variation—in action and observation spaces that look nothing like a 1-D line cue. **Balance** tasks (e.g. pole or posture stabilization) stress underactuated, high-bandwidth stabilization; here the difficulty is rather **where the line is** and how steering commands propagate through motors at a fixed control rate under PPO. The prompt’s examples frame the breadth of embodied RL; this repository instantiates the **mobile, perception-driven** slice.

### Sim-to-real challenges here

- **Observation shift**: Real tape, lighting, sensor noise, and comparator or ADC thresholds rarely match simulated reflectance. Mismatched **IR count**, spacing, or calibration versus training breaks the policy input distribution. Mitigations in-repo include scene randomization (and presets), noise/ADC/comparator options in `Sim2RealConfig`, and building the on-Pi observation the same way as in sim (`pi_deploy.py`, including `obs_from_hardware` alignment).
- **Dynamics and actuation**: Voltage drop, friction spread, deadband, PWM resolution, and **latency** are simplified or absent in sim. The environments model **action delay** and **PWM-first-order** motor dynamics so policies are not tuned only to ideal, instantaneous wheel commands.
- **Geometry and contact**: Curves, slip, and chassis details (e.g. casters) may disagree with rigid-body assumptions; track diversity (e.g. MuJoCo `track_type`) reduces straight-line overfitting.
- **Protocol mismatch**: Different `line_half_width`, `motor_dynamics`, or JSON defaults between train, eval, and the Pi cause brittle failure. `EnvConfig` and `Sim2RealConfig` (and CLI merges) should be carried consistently end to end.

For a concrete **alignment checklist** (sensors, motors, delay, randomization, config paths), skip ahead to **Sim-to-real and train/eval alignment** at the end of this README.

## Features

- **MuJoCo simulation** (recommended, matches `requirements.txt`): curved and multi-segment tracks, scene and physics randomization presets, PWM-oriented motor dynamics, JSON task and sim-to-real configs.
- **PyBullet simulation**: same observation layout idea (IR channels plus wheel state and delayed commands), separate `train.py` / `evaluate.py` and JSON configs for experiments or comparison.
- **Structured configs**: `mujoco_env_config.json` / `mujoco_sim2real_config.json` (MuJoCo) and `environment_config.json` / `sim2real_config.json` (PyBullet); CLI flags merge on top of defaults.
- **Deployment**: `pi_deploy.py` runs the policy on-device with digital comparator IR modules or MCP3008 ADC inputs and L298N drive via `gpiozero`.

## Installation

```bash
cd rl-line-following-bot
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You need a working **MuJoCo** install (`mujoco` Python package loads native libraries; follow [MuJoCo](https://mujoco.readthedocs.io/) setup for your OS). **PyBullet** is listed for the optional Bullet environment; you can skip it if you only use MuJoCo.

## Repository layout

| Path | Role |
|------|------|
| `line_follow_env_mujoco.py` | MuJoCo Gymnasium environment |
| `train_mujoco.py` | PPO training (MuJoCo) |
| `evaluate_mujoco.py` | Rollout, optional viewer / video |
| `mujoco_env_config.json` | Default MuJoCo task config (`EnvConfig`) |
| `mujoco_sim2real_config.json` | Default MuJoCo sim-to-real config (`Sim2RealConfig`) |
| `robots/diff_drive.xml` | MuJoCo robot / scene |
| `line_follow_env.py` | PyBullet environment |
| `train.py` | PPO training (PyBullet) |
| `evaluate.py` | PyBullet evaluation |
| `environment_config.json` | Default PyBullet task config |
| `sim2real_config.json` | Default PyBullet sim-to-real config |
| `robots/diff_drive.urdf` | PyBullet robot model |
| `pi_deploy.py` | Raspberry Pi deployment |
| `requirements.txt` | Python dependencies |
| `scripts/watch_model_snapshots.py` | Copy changing `*.zip` saves to numbered snapshots |
| `scripts/diagnose_reset_lateral.py` | Debug reset lateral error vs IR layout (PyBullet) |
| `scripts/vision_train_eval.sh` | Example PyBullet train then eval |

## Simulation backends

Run training and evaluation from the repository root so default JSON config paths resolve. Use `--help` on each script for the full flag list.

### MuJoCo (recommended)

- **Scripts**: `train_mujoco.py`, `evaluate_mujoco.py`
- **Defaults**: `train_mujoco.py` uses `--timesteps` **1_000_000** and `--save` **`models/ppo_mujoco`** unless overridden. Config files `mujoco_env_config.json` and `mujoco_sim2real_config.json` are loaded when present.

Example training:

```bash
python3 train_mujoco.py --timesteps 1000000 --save models/ppo_mujoco
```

Heavier sim-to-real-style training (curved track, randomization, quiet logs):

```bash
python3 train_mujoco.py \
  --timesteps 2000000 \
  --save models/ppo_mujoco_s2r \
  --track-type curve \
  --scene-rand \
  --domain-rand \
  --entropy-start 0.02 \
  --entropy-end 0.002 \
  --quiet
```

Example evaluation:

```bash
python3 evaluate_mujoco.py --model models/ppo_mujoco --deterministic --episodes 10
```

With rendering and optional video:

```bash
python3 evaluate_mujoco.py --model models/ppo_mujoco --deterministic --render --episodes 5
python3 evaluate_mujoco.py --model models/ppo_mujoco --deterministic --save-video-root eval_videos
```

Notable options (see `--help` for the rest): `--track-type`, `--ir-sensors`, `--motor-dynamics`, `--scene-rand`, `--domain-rand`, `--scene-preset`, `--physics-preset`, `--action-delay`, `--randomize-path`, `--ir-noise`, `--ir-digital`, `--quiet`.

### PyBullet (alternative)

- **Scripts**: `train.py`, `evaluate.py`
- **Defaults**: `train.py` uses `--timesteps` **100_000** and `--save` **`models/ppo_line_follow`**. Defaults load `environment_config.json` and `sim2real_config.json` when present.

Example:

```bash
python3 train.py --timesteps 100000 --save models/ppo_line_follow
python3 evaluate.py --model models/ppo_line_follow --deterministic
```

**Reward** in `line_follow_env.py` combines several shaped terms (not a single forward-speed product): lateral tracking, forward progress gated by line visibility, a bonus for positive forward acceleration, line-recovery and alive terms, minus command smoothness / bad-tracking / terminal penalties. Weights come from `EnvConfig` (and JSON), e.g. `ir_progress_weight`, `forward_accel_reward_weight`, `ir_sensor_lateral_weight`.

**Sim-to-real flags**: `--domain-rand` randomizes mass, friction, gravity, motor scaling, wheel velocity limits, and rear joint damping. `--scene-rand` randomizes IR scene appearance (reflectance, line width, floor texture, sensor drift). `--action-delay` adds buffered previous commands in steps.

**IR count**: CLI and `environment_config.json` default **`n_ir_sensors` = 2** for PyBullet, versus **5** in the default MuJoCo JSON. Observation dimension is **`n_ir_sensors + 4`** for both backends (see below).

## Observation and action

The observation is **`n_ir_sensors` IR channels in [0, 1]** (black line vs white floor), plus **four** values in **[-1, 1]**: normalized left and right wheel angular velocities and the previous normalized motor commands (after any action delay). With default **`N` IR sensors**, the vector length is **`N + 4`** (e.g. **9** when `N = 5` in MuJoCo, **6** when `N = 2` in PyBullet).

Actions are **`[left_cmd, right_cmd]`** in **`[-1, 1]`**, mapped to wheel velocity targets according to each environment’s `wheel_vel_max` (and related sim-to-real motor dynamics).

## Raspberry Pi deployment

### 1. Copy files to the Pi

```bash
scp models/ppo_mujoco.zip pi@raspberrypi.local:~/line_follower/
scp pi_deploy.py line_follow_env_mujoco.py pi@raspberrypi.local:~/line_follower/
```

### 2. Install dependencies on the Pi

```bash
pip install stable-baselines3 torch numpy gpiozero RPi.GPIO
# For MCP3008 ADC analogue IR:
pip install spidev
```

### 3. Run the policy

Digital IR (onboard comparator, Hack Lab–style):

```bash
python3 pi_deploy.py \
  --model models/ppo_mujoco.zip \
  --duration 120 \
  --verbose
```

Analogue IR (calibrate, then run):

```bash
python3 pi_deploy.py --model models/ppo_mujoco.zip --calibrate
python3 pi_deploy.py \
  --model models/ppo_mujoco.zip \
  --ir-mode adc \
  --ir-black-raw 45 \
  --ir-white-raw 980 \
  --duration 120 \
  --verbose
```

Dry-run without hardware:

```bash
python3 pi_deploy.py --model models/ppo_mujoco.zip --dry-run --verbose
```

### 4. Wiring (Hack Lab kit — BCM GPIO)

Motors via `gpiozero.Robot` (ENA/ENB assumed high; only direction pins listed):

| Signal | GPIO (BCM) |
|--------|------------|
| Left IN1 | 4 |
| Left IN2 | 14 |
| Right IN1 | 17 |
| Right IN2 | 18 |

IR sensors (digital, left to right):

| Sensor | GPIO (BCM) |
|--------|------------|
| IR 0 | 5 |
| IR 1 | 6 |
| IR 2 | 13 |
| IR 3 | 19 |
| IR 4 | 26 |

ADC chip select (SPI0, analogue mode): GPIO 8 (CE0). Override pins with `--left-in1`, `--left-in2`, `--right-in1`, `--right-in2`, `--ir-pins`.

## Sim-to-real and train/eval alignment

The checklist below operationalizes the transfer challenges summarized in **Embodiment, task domain, and sim-to-real** above. Use it for both “make sim realistic” and “avoid train/eval drift”:

- Match **`n_ir_sensors`** / `--ir-sensors` and **line geometry** (`--line-width-m`, `line_half_width`).
- Match **motor dynamics** (`motor_dynamics`, deadband, time constants) and **`action_delay_steps`** / `--action-delay` to the robot’s control loop.
- Match **track** settings (`track_type`, `randomize_path`) and **randomization** (`--scene-rand`, `--domain-rand`, or presets / JSON) between training and evaluation.
- Keep **sim2real** and **env** JSON paths consistent, or pass the same CLI overrides in both phases; MuJoCo loads `mujoco_env_config.json` and `mujoco_sim2real_config.json` by default when run from the repo root.

Mitigations in brief: PWM-first motor models and delay for actuator lag; scene randomization for lighting and tape variation; domain randomization for friction and inertia spread; curved tracks so policies do not only see straight lines; MuJoCo’s contact-rich actuation vs older Bullet control modes if you compare backends.
