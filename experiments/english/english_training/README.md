# Train ViACaPu trên VoxPopuli (English)

Thư mục độc lập để huấn luyện **ViACaPu ngữ âm (3.58M)** — model tốt nhất hiện tại —
trên dữ liệu tiếng Anh VoxPopuli, phục vụ đánh giá đa ngôn ngữ cho paper.

Code chỉ **gọi** model/trainer có sẵn của project (`model_via_capu.py`,
`train_via_capu_full.py`), **không copy**. Dữ liệu đọc từ HDD (đã extract sẵn),
không nằm trong thư mục này.

## Dữ liệu (đọc từ thư mục cũ, chỉ đọc)

| Split | Clips | Đường dẫn |
|---|---|---|
| Train (val 2% tách ra) | 162,082 | `/mnt/hdd_ngocmx/data_voxpopuli_en` |
| Test | 1,663 | `/mnt/hdd_ngocmx/data_voxpopuli_en_test` |

Nguồn: `facebook/voxpopuli` config `en`, dùng cột `raw_text` (giữ dấu câu + viết hoa).
BPE dùng chung vocab 3500 với Dolly.

## Cấu hình model (⭐ tốt nhất — GIỮ NGUYÊN)

`d=256, ac_gru=1, cross=1, heads=4, adjacent=0` → **3.58M params**.
Đây là config đạt Punct F1 0.921 / COMMA F1 0.872 trên Dolly. **Không** đổi sang
d=384 hay cross=2 (các bản scale-up đã chứng minh thất bại — gate collapse).

Mọi tham số nằm trong [`config.sh`](config.sh).

## Cách chạy

**Trọn pipeline (khuyến nghị)** — acoustic → text-only → eval, chạy nền, sống qua reboot:
```bash
cd english_training
setsid nohup bash run_all.sh > run_all.log 2>&1 &
tail -f run_all.log        # theo dõi
```

**Hoặc từng bước:**
```bash
bash 1_train_acoustic.sh   # model chính  -> exp_vox_en/best_en_ac.pt
bash 2_train_textonly.sh   # ablation     -> exp_vox_en/best_en_text.pt
python 3_evaluate_test.py --ckpt exp_vox_en/best_en_ac.pt   --use_acoustic 1
python 3_evaluate_test.py --ckpt exp_vox_en/best_en_text.pt --use_acoustic 0
```

## Kết quả xuất ra

`exp_vox_en/`:
- `best_en_ac.pt` — ViACaPu ngữ âm (model chính)
- `best_en_text.pt` — ViACaPu text-only (ablation)
- `log-en_ac`, `log-en_text` — log huấn luyện từng epoch

## Thời gian ước tính (RTX 3060)

- Acoustic: ~3–4h · Text-only: ~1.5h · Tổng ~5–6h.
- Nếu đứt giữa chừng: `train_via_capu_full.py` có cờ `--init_from <checkpoint>`
  để warm-start (xem config, thêm vào script nếu cần resume).

## Kỳ vọng

So sánh `best_en_ac` vs `best_en_text` trên test — nếu acoustic nâng COMMA F1 mạnh
(như trên Dolly: 0.661 → 0.872), khẳng định ViACaPu tổng quát hóa qua ngôn ngữ.
