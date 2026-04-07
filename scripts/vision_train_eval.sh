#!/usr/bin/env bash
# Run with: bash scripts/vision_train_eval.sh
# (Avoids zsh paste issues with # comment lines — use bash or: setopt interactivecomments in zsh)
set -euo pipefail
cd "$(dirname "$0")/.."
python train_mujoco.py --timesteps 1000000 --save models/ppo_mujoco
python evaluate_mujoco.py --model models/ppo_mujoco --deterministic --episodes 5
