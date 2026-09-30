#!/bin/bash
# =============================================================================
# Cấu hình train ViACaPu trên VoxPopuli (English).
# Sửa mọi tham số ở ĐÂY — các script khác source file này.
# Data đọc từ thư mục cũ trên HDD (KHÔNG copy).
# =============================================================================

# --- Đường dẫn (data đã extract sẵn, chỉ đọc) ---
export PROJECT_ROOT="/home/ai/nguyends/Edge-Punct-Casing"
export VENV="$PROJECT_ROOT/.venv/bin/python"
export BPE_MODEL="$PROJECT_ROOT/bpe_model/bpe.model"       # dùng chung vocab 3500 với Dolly

# TẠM THỜI trên SSD (HDD bị build Android chiếm I/O). Gốc: /mnt/hdd_ngocmx/data_voxpopuli_en*
export TRAIN_DIR="$PROJECT_ROOT/data_voxpopuli_en_ssd"        # 162,082 clips (val 2% tách từ đây)
export TEST_DIR="$PROJECT_ROOT/data_voxpopuli_en_test_ssd"    # 1,663 clips (eval cuối)

export EXP_DIR="$PROJECT_ROOT/english_training/exp_vox_en"  # checkpoint + log ra đây

# --- Cấu hình MODEL TỐT NHẤT (ViACaPu ngữ âm 3.58M) ---
# Đây là config của model ⭐ đạt Punct F1 0.921 / COMMA 0.872 trên Dolly.
# GIỮ NGUYÊN — không đổi sang d=384 / cross=2 (các bản đã chứng minh thất bại).
export D=256                 # embedding / hidden width
export AC_GRU=1              # số lớp BiGRU nhánh acoustic
export CROSS=1              # số tầng cross-attention fusion
export HEADS=4             # số attention head
export ADJACENT=0         # KHÔNG dùng adjacent-concat heads

# --- Siêu tham số huấn luyện (giống bản Dolly thắng) ---
export EPOCHS=12
export BATCH=48
export LR=8e-4
export DROPOUT=0.3
export WEIGHT_DECAY=5e-5
export PUNCT_WEIGHTS="1.0,1.6,1.05,1.4"   # class-weight chống lệch COMMA/QUESTION
# Trên SSD dùng được nhiều worker (random-access nhanh). Nếu quay lại HDD giảm về 2.
export NUM_WORKERS=4

mkdir -p "$EXP_DIR"
