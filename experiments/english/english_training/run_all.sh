#!/bin/bash
# Chạy TRỌN pipeline train tiếng Anh: acoustic -> text-only -> eval cả hai trên test.
# Chạy nền, tách phiên (sống qua reboot):
#   setsid nohup bash run_all.sh > run_all.log 2>&1 &
# Theo dõi:  tail -f run_all.log
set -e
cd "$(dirname "$(readlink -f "$0")")"
source ./config.sh

echo "========== [$(date)] PIPELINE ENGLISH START =========="

echo ">>> Bước 1/3: train acoustic (model chính)"
bash ./1_train_acoustic.sh

echo ">>> Bước 2/3: train text-only (ablation)"
bash ./2_train_textonly.sh

echo ">>> Bước 3/3: đánh giá trên test split"
cd "$PROJECT_ROOT"
"$VENV" english_training/3_evaluate_test.py --ckpt "$EXP_DIR/best_en_ac.pt"   --use_acoustic 1
"$VENV" english_training/3_evaluate_test.py --ckpt "$EXP_DIR/best_en_text.pt" --use_acoustic 0

echo "========== [$(date)] PIPELINE ENGLISH DONE =========="
