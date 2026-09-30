# Edge-Punct-Casing — ViACaPu

Mô hình **nhẹ, nhận biết ngữ âm** khôi phục dấu câu và viết hoa (CaPu) cho ASR
tiếng Việt trên thiết bị biên. Kết hợp nhánh văn bản gọn với nhánh ngữ âm log-Mel
qua gated cross-attention (alignment-free), **3.58M tham số**.

## Kết quả chính (Dolly-1000h, trung bình 3 seed)

| Cấu hình | Params | Case F1 | Punct F1 | COMMA F1 |
|---|---|---|---|---|
| Text-only | 2.68M | 0.937 ± .001 | 0.820 ± .002 | 0.660 ± .003 |
| **Acoustic + aux** | **3.58M** | **0.944 ± .002** | **0.927 ± .004** | **0.888 ± .007** |

Phát hiện cốt lõi: nhánh ngữ âm chỉ đóng góp **ổn định** khi bộ mã hóa ngữ âm được
giám sát bằng mất mát phụ dự đoán năng lượng (`--aux_energy_w`); nếu không, kết quả
phân thành hai chế độ tách biệt theo seed. Chi tiết: [docs/VIACAPU.md](docs/VIACAPU.md).

**Trước khi tiếp tục nghiên cứu, đọc
[docs/TRANG_THAI_VA_VIEC_CAN_LAM.md](docs/TRANG_THAI_VA_VIEC_CAN_LAM.md)** — ghi lại
các quyết định khoa học, giới hạn đã biết, và bốn việc cần làm theo thứ tự ưu tiên.

## Cấu trúc thư mục

```
├── src/                     # SOURCE ViACaPu (tách theo vai trò)
│   ├── model.py             #   kiến trúc (ViACaPu, AcousticEncoder)
│   ├── data.py              #   ĐỌC DATA: MemmapClipDS/ClipDS + collate
│   ├── metrics.py           #   ĐÁNH GIÁ: get_metrics/print_metrics + evaluate
│   ├── train.py             #   CHẠY train: vòng huấn luyện chính (Dolly memmap)
│   ├── train_prototype.py   #   train nhanh trên dataset .pt nhỏ
│   ├── __main__.py          #   entrypoint: `python3 -m src` gọi train.main()
│   └── utils.py             #   tiện ích (logger, scheduler...)
├── bpe_model/               # tokenizer SentencePiece tiếng Việt (bpe.model)
├── data/                    # data (SSD copies + symlink tới HDD/ngocmx)
│   └── data_audio_full/     #   → Dolly-1000h mel memmap (data chính)
├── exp_seed/                # checkpoint multi-seed của bài (ac/text × 3 seed + aux)
├── exp_via_full/            # checkpoint đơn (default --exp_dir)
├── experiments/             # thí nghiệm cũ (exp_dolly*, exp_via*, english/)
├── scripts/                 # script chạy multi-seed / pipeline
├── docs/                    # tài liệu ViACaPu
├── Paper_resource/          # bài báo IEEE (.tex) + hình
├── tools/                   # trích xuất/chuẩn bị dữ liệu
└── legacy/                  # code repo gốc Rud-nin (decode CLI, ONNX...) — không dùng cho ViACaPu
```

**Luồng dữ liệu:** `data.py` đọc mel memmap → `model.py` xử lý → `train.py` huấn luyện,
gọi `metrics.py` để đánh giá mỗi epoch. Import nội bộ dùng relative (`from .model ...`),
nên chạy bằng **`python3 -m src`** (qua `src/__main__.py`), **không** chạy trực tiếp
`python3 src/train.py`.

## Tái lập nhanh

```bash
# train acoustic + aux (cấu hình chính), 1 seed
python3 -m src --use_acoustic 1 --gate_bias -2.0 \
        --aux_energy_w 0.5 --seed 42 --tag ac_aux --exp_dir exp_seed

# multi-seed đầy đủ
bash scripts/run_aux_seeds.sh
```

Data mel Dolly nằm ở `data/data_audio_full` (symlink). Nếu mất, trích lại bằng
`tools/extract_dolly_audio_full.py`.

## Repo này chứa gì và KHÔNG chứa gì

Repo được dùng làm bản lưu đầy đủ của phần **không tái tạo được nếu không còn
máy/GPU gốc**. Tổng ~262 MB.

**Có trong repo:**

| Thư mục | Nội dung |
|---|---|
| `exp_seed/` | 10 checkpoint multi-seed: text/acoustic × seed 42-44, và biến thể `_aux5` |
| `exp_disentangle/` | 6 checkpoint lưới 2×2 tách biến `aux_energy_w` vs `gate_bias` |
| `exp_via_384_aux/` | checkpoint đối chứng d=384 kèm aux loss (7.34M tham số) |
| `exp_via_full/` | 2 checkpoint chạy đơn ban đầu |
| `experiments/` | **chỉ log** của các lần chạy cũ (`exp_dolly*`, `exp_via*`, `english/`) |
| `bpe_model/` | tokenizer SentencePiece — **bắt buộc** phải khớp thì checkpoint mới dùng được |
| `Paper_resource/` | bài báo `.tex`, hình `.pdf`/`.png`, script sinh hình |

**KHÔNG có trong repo, và cách lấy lại:**

| Thiếu | Dung lượng | Tái tạo bằng |
|---|---|---|
| `data/` (mel memmap, dataset) | ~51 GB | `tools/extract_dolly_audio_full.py` từ `dolly-vn/dolly-audio-1000h-vietnamese` trên HuggingFace |
| `experiments/**/*.pt` | ~18 GB | Chạy lại từ script trong `scripts/`; đây là các lần chạy cũ **đã bị thay thế**, log vẫn còn để đối chiếu |
| `.venv/` | ~5.6 GB | `python3 -m venv .venv && pip install torch numpy sentencepiece` |

Lý do loại: GitHub chặn file trên 100 MB, và `data/data_libritts_train_ssd/mel.f16`
riêng nó đã 28 GB. Các file này đều là **đặc trưng đã trích xuất** hoặc **checkpoint
của lần chạy cũ**, tính lại được từ code trong repo.

## Dùng lại checkpoint trên máy khác

```bash
git clone https://github.com/nguyends-bka/Edge-Punct-Casing.git
cd Edge-Punct-Casing

# checkpoint chính của bài (acoustic + aux, seed 43 — kết quả tốt nhất 0.930)
python3 -m tools.export_onnx --ckpt exp_seed/best_ac_s43_aux5.pt --out ac_aux.onnx
```

Checkpoint lưu dạng `{"model": state_dict}`; nạp bằng
`ViACaPu(vocab_size=3500, d=256, use_acoustic=True)` rồi `load_state_dict`.
Riêng `exp_via_384_aux/best_d384_aux_s42.pt` cần `d=384`.
