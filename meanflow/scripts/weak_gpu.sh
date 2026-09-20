#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
METHOD="${METHOD:-weak}"
SEED="${SEED:-0}"
BATCH="${BATCH:-64}"
CHANNELS="${CHANNELS:-32}"
EPOCHS="${EPOCHS:-100}"
WARMUP="${WARMUP:-5}"
LR="${LR:-0.0001}"
OUT="${OUT:-./runs/${METHOD}_seed${SEED}}"
python train.py \
    --method "$METHOD" --output_dir "$OUT" --dataset cifar10 \
    --data_path ./data --batch_size "$BATCH" --model_channels "$CHANNELS" \
    --epochs "$EPOCHS" --warmup_epochs "$WARMUP" --lr "$LR" \
    --optimizer_betas 0.9 0.999 --seed "$SEED" --iid_sampling \
    --dropout 0 --not_compile --ema_decay 0.9999 --ema_decays \
    --diag_probability 0.25 --weak_features 64 --weak_weight 1.0 \
    --weak_sigma_z 1.0 --weak_sigma_r 1.0 --weak_sigma_t 1.0 \
    --eval_frequency 10 --log_per_step 50 --num_workers 4 "$@"