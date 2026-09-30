#!/bin/bash
# Train ViACaPu TEXT-ONLY (2.68M) trên VoxPopuli English — ablation.
# Cùng nhánh văn bản & đầu ra, tắt nhánh acoustic → chứng minh lợi ích ngữ âm.
# Chạy: bash 2_train_textonly.sh
set -e
cd "$(dirname "$(readlink -f "$0")")"
source ./config.sh

echo "[$(date)] Bắt đầu train ViACaPu text-only (ablation, tắt acoustic)"
cd "$PROJECT_ROOT"
"$VENV" train_via_capu_full.py \
  --data_dir "$TRAIN_DIR" \
  --bpe_model "$BPE_MODEL" \
  --exp_dir "$EXP_DIR" \
  --tag en_text \
  --use_acoustic 0 \
  --d "$D" --adjacent_heads "$ADJACENT" \
  --epochs "$EPOCHS" --batch_size "$BATCH" --lr "$LR" --dropout "$DROPOUT" \
  --weight_decay "$WEIGHT_DECAY" --punct_weights "$PUNCT_WEIGHTS" --num_workers "$NUM_WORKERS"

echo "[$(date)] XONG text-only -> $EXP_DIR/best_en_text.pt"
