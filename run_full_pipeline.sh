#!/bin/bash
cd /home/ai/ngocmx/Edge-Punct-Casing
echo "waiting for extraction to finish..." > full_pipeline.log
while ! grep -q "^DONE:" data_audio_full_extract.log 2>/dev/null; do sleep 60; done
echo "extraction done at $(date)" >> full_pipeline.log
.venv/bin/python train_via_capu_full.py --use_acoustic 0 --epochs 12 --batch_size 64 \
  --exp_dir exp_via_full > exp_via_full_text.log 2>&1
echo "text-only done at $(date)" >> full_pipeline.log
.venv/bin/python train_via_capu_full.py --use_acoustic 1 --epochs 12 --batch_size 64 \
  --exp_dir exp_via_full > exp_via_full_ac.log 2>&1
echo "PIPELINE_DONE at $(date)" >> full_pipeline.log
