#!/bin/bash
# Chạy tuần tự các seed còn lại cho multi-seed (mean±std).
# seed=42 acoustic đang chạy riêng; script này lo 5 lần còn lại.
# Val split cố định ở 42 trong train script -> mọi seed cùng tập val.
set -e
cd /home/ai/nguyends/Edge-Punct-Casing
DATA=/home/ai/ngocmx/Edge-Punct-Casing/data_audio_full
SC=/tmp/claude-1000/-home-ai-nguyends-Edge-Punct-Casing/9f39f5e8-7573-4f9e-9665-bc36d864d7e1/scratchpad

# đợi seed=42 acoustic (PID hiện tại) xong: chờ tới khi không còn tiến trình train nào
echo "$(date) chờ seed=42 acoustic xong..."
while pgrep -f "tag ac_s42" >/dev/null 2>&1; do sleep 30; done
echo "$(date) seed=42 xong, bắt đầu 5 lần còn lại"

# (use_acoustic, seed, tag)
runs=(
  "1 43 ac_s43"
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
echo "$(date) ALL MULTISEED RUNS COMPLETE"
