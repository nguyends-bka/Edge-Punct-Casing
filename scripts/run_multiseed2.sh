#!/bin/bash
# Chạy nốt 4 lần còn lại (ac_s44 bị cắt giữa chừng -> chạy lại; text 3 seed).
set -e
cd /home/ai/nguyends/Edge-Punct-Casing
DATA=/home/ai/ngocmx/Edge-Punct-Casing/data_audio_full
SC=/tmp/claude-1000/-home-ai-nguyends-Edge-Punct-Casing/9f39f5e8-7573-4f9e-9665-bc36d864d7e1/scratchpad

runs=(
  "1 44 ac_s44"
  "0 42 text_s42"
  "0 43 text_s43"
  "0 44 text_s44"
)
for r in "${runs[@]}"; do
  set -- $r
  ua=$1; seed=$2; tag=$3
  echo "$(date) === START $tag (use_acoustic=$ua seed=$seed) ==="
  python3 -m src \
    --data_dir "$DATA" --exp_dir exp_seed --use_acoustic $ua \
    --seed $seed --tag $tag --epochs 12 \
    > "$SC/run_$tag.out" 2>&1
  echo "$(date) === DONE $tag ==="
done
echo "$(date) ALL REMAINING MULTISEED RUNS COMPLETE"
