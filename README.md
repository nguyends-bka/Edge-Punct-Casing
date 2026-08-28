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
dao động mạnh theo seed. Chi tiết: [docs/VIACAPU.md](docs/VIACAPU.md).

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
