#!/bin/bash
# LibriTTS-R (audiobook English) — mục tiêu vượt Dolly 0.921.
# Dữ liệu audiobook: người đọc DIỄN dấu câu bằng ngữ điệu => acoustic có tín hiệu thật.
# Recipe v2: aux-energy loss + gate bias + lr-const, model ⭐ 3.58M (d=256, cross=1).
# Chạy: setsid nohup bash run_libritts.sh > run_libritts.log 2>&1 &
set -e
cd "$(dirname "$(readlink -f "$0")")"
source ./config.sh

BPE_LT="$PROJECT_ROOT/bpe_model_libritts/bpe_lt.model"
TRAIN_LT="$PROJECT_ROOT/data_libritts_train_ssd"
TEST_LT="$PROJECT_ROOT/data_libritts_test_ssd"
EXP_LT="$PROJECT_ROOT/english_training/exp_libritts"
mkdir -p "$EXP_LT"

echo "========== [$(date)] LIBRITTS START =========="
cd "$PROJECT_ROOT"

echo ">>> 1/3: acoustic (model ⭐ + recipe v2)"
"$VENV" train_via_capu_full.py \
  --data_dir "$TRAIN_LT" --bpe_model "$BPE_LT" --exp_dir "$EXP_LT" --tag lt_ac \
  --use_acoustic 1 --d "$D" --ac_gru_layers "$AC_GRU" --cross_layers "$CROSS" \
  --n_heads "$HEADS" --adjacent_heads "$ADJACENT" \
  --epochs 16 --lr_const_epochs 4 --aux_energy_w 0.1 --gate_bias -0.5 \
  --batch_size "$BATCH" --lr "$LR" --dropout "$DROPOUT" \
  --weight_decay "$WEIGHT_DECAY" --punct_weights "$PUNCT_WEIGHTS" --num_workers "$NUM_WORKERS"

echo ">>> 2/3: text-only (ablation)"
"$VENV" train_via_capu_full.py \
  --data_dir "$TRAIN_LT" --bpe_model "$BPE_LT" --exp_dir "$EXP_LT" --tag lt_text \
  --use_acoustic 0 --d "$D" --adjacent_heads "$ADJACENT" \
  --epochs 16 --lr_const_epochs 4 \
  --batch_size "$BATCH" --lr "$LR" --dropout "$DROPOUT" \
  --weight_decay "$WEIGHT_DECAY" --punct_weights "$PUNCT_WEIGHTS" --num_workers "$NUM_WORKERS"

echo ">>> 3/3: eval test.clean"
"$VENV" english_training/3_evaluate_test.py --ckpt "$EXP_LT/best_lt_ac.pt" \
  --use_acoustic 1 --bpe_model "$BPE_LT" --test_dir "$TEST_LT"
"$VENV" english_training/3_evaluate_test.py --ckpt "$EXP_LT/best_lt_text.pt" \
  --use_acoustic 0 --bpe_model "$BPE_LT" --test_dir "$TEST_LT"

echo "========== [$(date)] LIBRITTS DONE =========="
