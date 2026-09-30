#!/bin/bash
# Tách biến aux_energy_w vs gate_bias.
#
# Bảng tab:seed hiện tại so sánh hai cấu hình khác nhau Ở HAI CHỖ:
#   "ngữ âm, không aux" : aux=0,   gate_bias=None   (run_multiseed.sh)
#   "ngữ âm + aux"      : aux=0.5, gate_bias=-2.0   (run_aux_seeds.sh)
# => không thể quy cải thiện cho aux loss. Script này chạy hai ô còn thiếu
# của lưới 2x2, mỗi ô 3 seed, mọi thứ khác giữ nguyên:
#   A) aux=0,   gate_bias=-2.0  -> gate bias một mình có đủ không?
#   B) aux=0.5, gate_bias=None  -> aux loss một mình có đủ không?
set -e
cd /home/ai/nguyends/Edge-Punct-Casing
DATA=/home/ai/ngocmx/Edge-Punct-Casing/data_audio_full
OUT=exp_disentangle
mkdir -p "$OUT"

for seed in 42 43 44; do
  # --- ô A: chỉ gate_bias, KHÔNG aux ---
  tag="gb_only_s${seed}"
  if [ -f "$OUT/best_${tag}.pt" ]; then
    echo "$(date) SKIP $tag (đã có)"
  else
    echo "$(date) === START $tag (aux=0, gate_bias=-2.0) ==="
    python3 -m src \
      --data_dir "$DATA" --exp_dir "$OUT" --use_acoustic 1 \
      --seed $seed --tag $tag --epochs 12 \
      --gate_bias -2.0 --aux_energy_w 0.0 \
      > "$OUT/run_$tag.out" 2>&1
    echo "$(date) === DONE $tag ==="
  fi

  # --- ô B: chỉ aux, KHÔNG gate_bias ---
  tag="aux_only_s${seed}"
  if [ -f "$OUT/best_${tag}.pt" ]; then
    echo "$(date) SKIP $tag (đã có)"
  else
    echo "$(date) === START $tag (aux=0.5, gate_bias mặc định) ==="
    python3 -m src \
      --data_dir "$DATA" --exp_dir "$OUT" --use_acoustic 1 \
      --seed $seed --tag $tag --epochs 12 \
      --aux_energy_w 0.5 \
      > "$OUT/run_$tag.out" 2>&1
    echo "$(date) === DONE $tag ==="
  fi
done
echo "$(date) DISENTANGLE GRID COMPLETE"
