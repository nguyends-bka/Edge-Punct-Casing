#!/bin/bash
# Train ViACaPu NGỮ ÂM (3.58M) trên VoxPopuli English — model chính ⭐.
# Chạy: bash 1_train_acoustic.sh
set -e
cd "$(dirname "$(readlink -f "$0")")"
source ./config.sh

echo "[$(date)] Bắt đầu train ViACaPu acoustic (d=$D, ac_gru=$AC_GRU, cross=$CROSS, heads=$HEADS)"
cd "$PROJECT_ROOT"
"$VENV" train_via_capu_full.py \
  --data_dir "$TRAIN_DIR" \
  --bpe_model "$BPE_MODEL" \
  --exp_dir "$EXP_DIR" \
  --tag en_ac \
  --use_acoustic 1 \
  --d "$D" --ac_gru_layers "$AC_GRU" --cross_layers "$CROSS" --n_heads "$HEADS" --adjacent_heads "$ADJACENT" \
  --epochs "$EPOCHS" --batch_size "$BATCH" --lr "$LR" --dropout "$DROPOUT" \
  --weight_decay "$WEIGHT_DECAY" --punct_weights "$PUNCT_WEIGHTS" --num_workers "$NUM_WORKERS"

echo "[$(date)] XONG acoustic -> $EXP_DIR/best_en_ac.pt"
