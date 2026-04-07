# RL Line Following Bot

Train a PPO policy to follow a black line using IR reflectance sensors, then deploy it on a Raspberry Pi.

Two simulation backends are provided:
- **MuJoCo** (recommended) — `line_follow_env_mujoco.py`, `train_mujoco.py`, `evaluate_mujoco.py`
- **PyBullet** (legacy) — `line_follow_env.py`, `train.py`, `evaluate.py`

---

## Bugs Found & Fixed (PyBullet → MuJoCo migration)

### 1. Motor dynamics mismatch — the biggest sim2real gap
**Problem**: `sim2real_config.json` had `"motor_dynamics": "accel"`, meaning the policy learned to command *angular accelerations* in simulation. A real Raspberry Pi + L298N H-bridge receives *PWM duty cycles*, not acceleration commands.

**Fix**: `Sim2RealConfig` now defaults to `"motor_dynamics": "pwm_first_order"`, which implements a slew-rate-limited, first-order-lag duty-cycle model matching the L298N directly.

### 2. Only 2 IR sensors — saturating lateral error signal
**Problem**: With 2 sensors, the centroid error saturates at `±1.0` whenever the line is not squarely between the sensors. The policy gets zero gradient signal about *how far off-track* it is.

**Fix**: 5 IR sensors, spaced 2 cm apart (8 cm total span). The centroid error is now smooth and informative across the full ±1 range.

### 3. Progress reward in wrong frame
**Problem**: `forward_speed = v_body[0]` — the robot's body-frame x-velocity. This rewards moving in the robot's *nose direction*, not along the *line*.

**Fix**: `v_progress = dot(world_velocity, path_tangent)` — true projection onto the track tangent, correct for straight, curved, and arc paths.

### 4. Straight-line-only training
**Problem**: `randomize_path=False` and no curves — the robot only ever saw a straight, fixed-heading line.

**Fix**: `EnvConfig.track_type` supports `"straight"`, `"curve"` (sinusoidal), `"arc"` (circular), `"s_curve"`, and `"turn_sequence"`.

### 5. No Raspberry Pi deployment code
**Problem**: No script to actually run the policy on hardware.

**Fix**: `pi_deploy.py` — full control loop for RPi with digital GPIO comparator IR sensors or MCP3008 SPI ADC, L298N H-bridge via `gpiozero.Robot`, and the same observation pipeline used in training.

### 6. `ir_lost_consecutive_steps` constant vs JSON mismatch
**Problem**: Hard-coded constant in `line_follow_env.py` differed from the JSON config value.

**Fix**: New environment uses only the `EnvConfig` dataclass; no duplicate constants.

### 7. MuJoCo physics instability (`Nan/Inf in QACC`)
**Problem**: Original XML used the `RK4` integrator with stiff contacts on the lightweight (0.35 kg) chassis, causing numerical blow-up.

**Fix**: Changed to `implicitfast` integrator, `timestep=0.004` (250 Hz), softened contacts (`solref="0.04 1"`, `solimp="0.95 0.99 0.001 0.5 2"`). Also set explicitly in `__init__` via `self._model.opt.integrator = 3`.

### 8. PyBullet import in MuJoCo environment
**Problem**: `_apply_drive_command` imported `MAX_WHEEL_ANG_ACCEL` from `line_follow_env.py`, triggering `ModuleNotFoundError: No module named 'pybullet'` during MuJoCo training.

**Fix**: Constant inlined directly as `MAX_WHEEL_ANG_ACCEL = 48.0`.

---

## Setup

```bash
cd rl-line-following-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## MuJoCo Training

Basic training run (straight line):
```bash
python3 train_mujoco.py --timesteps 1000000 --save models/ppo_kit
```

Full sim-to-real training (curved track, all randomisation, Pi-compatible motor model):
```bash
python3 train_mujoco.py \
  --timesteps 2000000 \
  --save models/ppo_kit_s2r \
  --track-type curve \
  --scene-rand \
  --domain-rand \
  --action-delay 1 \
  --motor-dynamics pwm_first_order \
  --entropy-start 0.02 \
  --entropy-end 0.002 \
  --quiet
```

Key options:
- `--track-type straight|curve|arc|s_curve|turn_sequence` — path shape
- `--ir-sensors N` — number of IR sensors (default 5; must match eval)
- `--motor-dynamics pwm_first_order` — matches real L298N (default)
- `--scene-rand` — randomise IR reflectance, line width, sensor gain, floor bias, wear
- `--domain-rand` — randomise physics (mass, friction, gravity, wheel damping)
- `--scene-preset none|balanced|aggressive` — preset scene randomisation level
- `--physics-preset none|balanced|aggressive` — preset domain randomisation level
- `--action-delay N` — simulate control latency in steps (1 step = 20 ms)
- `--randomize-path` — randomise track heading and lateral offset each episode
- `--ir-noise STD` — Gaussian noise on IR readings
- `--ir-gamma G` — photodiode gamma correction
- `--ir-digital` — binary comparator output instead of analogue
- `--quiet` — suppress per-episode output (much faster training throughput)

---

## MuJoCo Evaluation

```bash
python3 evaluate_mujoco.py --model models/ppo_kit --deterministic --episodes 10
```

With GUI rendering (MuJoCo passive viewer + OpenCV window):
```bash
python3 evaluate_mujoco.py --model models/ppo_kit --deterministic --render --episodes 5
```

On a curved track:
```bash
python3 evaluate_mujoco.py --model models/ppo_kit --deterministic --track-type curve --episodes 10
```

Save evaluation videos:
```bash
python3 evaluate_mujoco.py --model models/ppo_kit --deterministic --save-video-root eval_videos
```

**Important**: Always run from the `rl-line-following-bot/` directory so the default config files (`mujoco_env_config.json`, `mujoco_sim2real_config.json`) are found. If you see an observation space mismatch, add `--ir-sensors 5` explicitly.

---

## Observation Space

Each observation is a **9-dimensional** vector (with default 5 IR sensors):

| Index | Description |
|-------|-------------|
| 0–4   | IR sensor readings — 0 = black (on line), 1 = white (off line) |
| 5     | Left wheel velocity, normalised to [-1, 1] |
| 6     | Right wheel velocity, normalised to [-1, 1] |
| 7     | Previous left motor command [-1, 1] |
| 8     | Previous right motor command [-1, 1] |

Action space: `[left_cmd, right_cmd]` in `[-1, 1]`, mapped to wheel velocity targets `[-18, 18] rad/s`.

---

## Raspberry Pi Deployment

### 1. Copy files to the Pi

```bash
scp models/ppo_kit.zip pi@raspberrypi.local:~/line_follower/
scp pi_deploy.py line_follow_env_mujoco.py pi@raspberrypi.local:~/line_follower/
```

### 2. Install dependencies on Pi

```bash
pip install stable-baselines3 torch numpy gpiozero RPi.GPIO
# For analogue IR sensors via MCP3008 ADC:
pip install spidev
```

### 3. Run the policy (digital IR mode — Hack Lab kit default)

The standard Hack Lab kit IR modules have an onboard comparator (digital output). No calibration needed; sensitivity is set by the trimpot on each module.

```bash
python3 pi_deploy.py \
  --model models/ppo_kit.zip \
  --duration 120 \
  --verbose
```

Analogue IR mode (MCP3008 ADC):
```bash
# Calibrate first:
python3 pi_deploy.py --model models/ppo_kit.zip --calibrate

# Then run:
python3 pi_deploy.py \
  --model models/ppo_kit.zip \
  --ir-mode adc \
  --ir-black-raw 45 \
  --ir-white-raw 980 \
  --duration 120 \
  --verbose
```

Dry-run (no hardware):
```bash
python3 pi_deploy.py --model models/ppo_kit.zip --dry-run --verbose
```

### 4. Wiring (Hack Lab kit — BCM pin numbers)

Motor driver uses `gpiozero.Robot`. ENA/ENB jumpers are hardwired HIGH on this kit, so only IN1/IN2/IN3/IN4 are controlled.

| Signal    | GPIO (BCM) |
|-----------|------------|
| Left  IN1 | 4          |
| Left  IN2 | 14         |
| Right IN1 | 17         |
| Right IN2 | 18         |

IR sensor GPIO pins (digital comparator output, left to right):

| Sensor | GPIO (BCM) |
|--------|------------|
| IR 0   | 5          |
| IR 1   | 6          |
| IR 2   | 13         |
| IR 3   | 19         |
| IR 4   | 26         |

ADC CS (SPI0, analogue mode only): GPIO 8 (CE0)

Override any pin: `--left-in1 PIN`, `--left-in2 PIN`, `--right-in1 PIN`, `--right-in2 PIN`, `--ir-pins P0 P1 P2 P3 P4`.

---

## Sim2Real Transfer: Challenges and Mitigations

### Challenge 1: Motor lag and deadband
Real DC motors with gearboxes and H-bridges have a *deadband* (stiction) and a first-order *lag* in velocity response.

**Mitigation**: `motor_dynamics = "pwm_first_order"` models deadband, slew rate, and first-order lag. Tune `--motor-deadband`, `--motor-tau`, and `--motor-slew` to match your motor by watching `--verbose` output.

### Challenge 2: IR sensor variation
Real sensors vary in ambient light sensitivity, tape reflectance, and mounting tolerances.

**Mitigation**: `--scene-rand` randomises black/white reflectance, edge sharpness, floor bias, line wear, and per-sensor gain/offset each episode.

### Challenge 3: Control latency
The Pi introduces latency between sensing and actuation (Python interpreter, SPI/GPIO).

**Mitigation**: `--action-delay 1` during training adds a 1-step (20 ms) delay, matching real system lag.

### Challenge 4: Curved tracks
A policy trained only on straight lines fails on turns.

**Mitigation**: Use `--track-type curve` or `--track-type arc` during training. `TrackGeometry` provides analytically correct path-tangent and lateral-error for all track shapes. Training from scratch on curves with full randomisation is difficult — start with straight or minimal randomisation, then fine-tune on curves.

### Challenge 5: Physics model mismatch (PyBullet-specific)
The original PyBullet implementation used `VELOCITY_CONTROL` which bypasses physics entirely — wheel slip, contact stiffness, and inertia were ignored.

**Mitigation**: MuJoCo's velocity actuator applies forces through the constraint solver, so wheel slip, contact stiffness, and inertia are properly simulated.

---

## Keep Train and Eval Aligned

The most common source of confusing results is a mismatch between training and evaluation. Always keep these consistent:
- `--ir-sensors` / `n_ir_sensors` (default 5; determines obs space dimension)
- `--line-width-m` / `line_half_width`
- `--motor-dynamics`
- `--action-delay`
- `--track-type`
- `--scene-rand` / `--scene-preset`
- `--domain-rand` / `--physics-preset`

Config files (`mujoco_env_config.json`, `mujoco_sim2real_config.json`) are loaded automatically if present in the same directory. CLI flags override config file values.

---

## Files

| File | Purpose |
|------|---------|
| `line_follow_env_mujoco.py` | MuJoCo Gymnasium environment (~1550 lines) |
| `train_mujoco.py` | PPO training with MuJoCo |
| `evaluate_mujoco.py` | Policy evaluation with optional rendering and video save |
| `pi_deploy.py` | Raspberry Pi deployment (gpiozero + optional MCP3008) |
| `robots/diff_drive.xml` | MuJoCo robot model (Hack Lab kit dimensions) |
| `mujoco_env_config.json` | Task config (loaded by default, overrides dataclass defaults) |
| `mujoco_sim2real_config.json` | Sim2Real config (loaded by default) |
| `line_follow_env.py` | Legacy PyBullet environment |
| `train.py` | Legacy PyBullet training |
| `evaluate.py` | Legacy PyBullet evaluation |
| `robots/diff_drive.urdf` | Legacy PyBullet robot model |
| `sim2real_config.json` | Legacy PyBullet sim2real config |
| `environment_config.json` | Legacy PyBullet task config |
| `requirements.txt` | Python dependencies |
