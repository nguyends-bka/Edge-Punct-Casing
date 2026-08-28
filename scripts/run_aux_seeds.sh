#!/bin/bash
# Xác nhận aux config ổn định qua 3 seed: đã có s43=0.930, chạy nốt s42, s44.
set -e
cd /home/ai/nguyends/Edge-Punct-Casing
DATA=/home/ai/ngocmx/Edge-Punct-Casing/data_audio_full
SC=/tmp/claude-1000/-home-ai-nguyends-Edge-Punct-Casing/9f39f5e8-7573-4f9e-9665-bc36d864d7e1/scratchpad
for seed in 42 44; do
  tag="ac_s${seed}_aux5"
  echo "$(date) === START $tag ==="
  python3 -m src \
    --data_dir "$DATA" --exp_dir exp_seed --use_acoustic 1 \
    --seed $seed --tag $tag --epochs 12 \
    --gate_bias -2.0 --aux_energy_w 0.5 \
    > "$SC/run_$tag.out" 2>&1
  echo "$(date) === DONE $tag ==="
done
echo "$(date) AUX 3-SEED COMPLETE"
