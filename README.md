# RL Line Following Bot

Train and evaluate a PPO policy for a PyBullet line-following robot with IR-style observations.

## Files

- `train.py`: train a PPO policy
- `evaluate.py`: run a saved policy in the PyBullet GUI
- `line_follow_env.py`: Gymnasium environment and reward logic
- `sim2real.example.json`: example sim-to-real config overrides
- `models/`: default output directory for saved policies

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Basic training run:

```bash
python3 train.py --timesteps 1000000 --save models/ppo_line_follow
```

Useful training options:

- `--gui`: watch training in PyBullet
- `--quiet`: suppress per-episode env logs
- `--seed 0`: set training seed
- `--max-episode-steps 500`: episode horizon
- `--ir-sensors 2`: number of IR sensors
- `--line-width-m 0.05`: line width in meters
- `--scene-rand`: randomize line and floor appearance
- `--domain-rand`: randomize physical properties
- `--action-delay N`: add control delay
- `--ir-model analytic|ray_bundle`: choose sensor model
- `--sim2real-config sim2real.example.json`: load config overrides

Example sim-to-real style training run:

```bash
python3 train.py \
  --timesteps 1000000 \
  --save models/ppo_line_follow_sim2real \
  --scene-rand \
  --domain-rand \
  --action-delay 1 \
  --ir-model ray_bundle \
  --sim2real-config sim2real.example.json
```

Saved models are written as `.zip` files. For example, `--save models/ppo_line_follow` produces `models/ppo_line_follow.zip`.

## Evaluate

Run a trained model in the PyBullet GUI:

```bash
python3 evaluate.py --model models/ppo_line_follow --deterministic
```

Useful evaluation options:

- `--episodes 10`: number of rollouts
- `--deterministic`: use greedy actions instead of sampling
- `--gui-camera fixed|follow`: choose camera mode
- `--no-ir-gui`: disable the matplotlib IR window
- `--seed 0`: base seed for rollouts
- `--max-episode-steps 500`: episode horizon
- `--sim2real-config sim2real.example.json`: load config overrides

Example evaluation run matching the sim-to-real training setup:

```bash
python3 evaluate.py \
  --model models/ppo_line_follow_sim2real \
  --deterministic \
  --scene-rand \
  --ir-model ray_bundle \
  --sim2real-config sim2real.example.json
```

Note: `evaluate.py` does not currently expose `--action-delay` or `--domain-rand` directly on the CLI. If those were used during training, pass the same settings through `--sim2real-config`.

Example:

```bash
python3 evaluate.py \
  --model models/ppo_line_follow_sim2real \
  --deterministic \
  --sim2real-config sim2real.example.json
```

At the end of each episode, evaluation prints the return and termination diagnostics such as `reason`, `lat`, `line_strength`, and `line_lost_count`.

## Recommended Workflow

1. Train a model with `train.py`.
2. Evaluate the saved `.zip` model with `evaluate.py`.
3. If behavior is poor, adjust reward, motor dynamics, or sim-to-real settings in `line_follow_env.py` or `sim2real.example.json`.
4. Retrain after any environment or reward change.

## Keep Train And Eval Aligned

The most common source of confusing results is a mismatch between training and evaluation settings. Try to keep these aligned:

- `--ir-sensors`
- `--line-width-m`
- `--ir-model`
- `--ir-noise`
- `--ir-digital`
- `--ir-comparator-level`
- `--scene-rand`
- `--sim2real-config`

If you trained with a different environment configuration and evaluate with defaults, the robot may look unstable or terminate early even if training seemed fine.
