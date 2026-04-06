#!/usr/bin/env bash
# Run with: bash scripts/vision_train_eval.sh
# (Avoids zsh paste issues with # comment lines — use bash or: setopt interactivecomments in zsh)
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py --timesteps 100000 --save models/ppo_line_follow_ir
python evaluate.py --model models/ppo_line_follow_ir
