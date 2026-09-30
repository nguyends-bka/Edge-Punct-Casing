#!/bin/bash
# English recipe v2 — combo cải tiến A+B+C+D:
#   A. BPE tiếng Anh riêng (bpe_model_en, vocab 3500) + data retokenized (_enbpe)
#   B. Aux energy loss (--aux_energy_w 0.1): ép acoustic encoder học pause sớm
#   C. Gate bias -0.5 (--gate_bias): acoustic context chảy vào ngay từ đầu
#   D. Giữ lr cao 4 epoch đầu (--lr_const_epochs 4), tổng 16 epochs
# Chạy: setsid nohup bash run_v2.sh > run_v2.log 2>&1 &
set -e
cd "$(dirname "$(readlink -f "$0")")"
source ./config.sh

BPE_EN="$PROJECT_ROOT/bpe_model_en/bpe_en.model"
TRAIN_EN="$PROJECT_ROOT/data_voxpopuli_en_ssd_enbpe"
TEST_EN="$PROJECT_ROOT/data_voxpopuli_en_test_ssd_enbpe"

echo "========== [$(date)] ENGLISH V2 START =========="
cd "$PROJECT_ROOT"

echo ">>> V2 1/3: acoustic + aux-energy + gate-bias + lr-const"
"$VENV" train_via_capu_full.py \
  --data_dir "$TRAIN_EN" --bpe_model "$BPE_EN" --exp_dir "$EXP_DIR" --tag en2_ac \
  --use_acoustic 1 --d "$D" --ac_gru_layers "$AC_GRU" --cross_layers "$CROSS" \
  --n_heads "$HEADS" --adjacent_heads "$ADJACENT" \
  --epochs 16 --lr_const_epochs 4 --aux_energy_w 0.1 --gate_bias -0.5 \
  --batch_size "$BATCH" --lr "$LR" --dropout "$DROPOUT" \
  --weight_decay "$WEIGHT_DECAY" --punct_weights "$PUNCT_WEIGHTS" --num_workers "$NUM_WORKERS"

echo ">>> V2 2/3: text-only (ablation, cùng BPE mới)"
"$VENV" train_via_capu_full.py \
  --data_dir "$TRAIN_EN" --bpe_model "$BPE_EN" --exp_dir "$EXP_DIR" --tag en2_text \
  --use_acoustic 0 --d "$D" --adjacent_heads "$ADJACENT" \
  --epochs 16 --lr_const_epochs 4 \
  --batch_size "$BATCH" --lr "$LR" --dropout "$DROPOUT" \
  --weight_decay "$WEIGHT_DECAY" --punct_weights "$PUNCT_WEIGHTS" --num_workers "$NUM_WORKERS"

echo ">>> V2 3/3: eval test"
"$VENV" english_training/3_evaluate_test.py --ckpt "$EXP_DIR/best_en2_ac.pt" \
  --use_acoustic 1 --bpe_model "$BPE_EN" --test_dir "$TEST_EN"
"$VENV" english_training/3_evaluate_test.py --ckpt "$EXP_DIR/best_en2_text.pt" \
  --use_acoustic 0 --bpe_model "$BPE_EN" --test_dir "$TEST_EN"

echo "========== [$(date)] ENGLISH V2 DONE =========="
