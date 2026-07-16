# Edge-Punct-Casing
This repo is the code implementation for the paper "A lightweight and efficient punctuation and word casing prediction model for on-device streaming ASR"

## Huấn luyện trên bộ dữ liệu Dolly 1000h

Xem [RESULTS.md](RESULTS.md) để biết chi tiết kiến trúc mô hình, quy trình dữ liệu,
các cải tiến (embedding, class-weight, bidirectional) và bảng số liệu đánh giá.

**Model khuyến nghị:** `exp_dolly_v2w/epoch-13.pt` (8.78M params) — Case F1 **0.875**, Punct F1 **0.725**.
