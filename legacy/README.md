# legacy/ — code repo gốc (Rud-nin), KHÔNG dùng cho ViACaPu

Đây là code của repo gốc "A lightweight ... punctuation and word casing prediction
model" (text-only, CNN-BiLSTM) mà ViACaPu kế thừa ý tưởng nhánh văn bản. Giữ lại để
tham chiếu, không nằm trong pipeline ViACaPu hiện tại.

- `model.py`, `model_v4.py` — kiến trúc text-only cũ (Model_new, v4)
- `train.py` — vòng huấn luyện cũ (get_model/get_params)
- `data_module.py` — DataModule cũ
- `eval_v4.py`, `eval_all.py`, `collect_preds.py`, `decode_sentence.py` — eval/decode cũ
- `onnx_decode*.py`, `export-onnx.py` — export/inference ONNX
- `docs/` — RESULTS.md, REPOSITORY_OVERVIEW.md, TRAINING_FROM_SCRATCH.md (tài liệu repo cũ)
- `figures_old/` — hình của các thí nghiệm cũ

Lưu ý: `decode.py` ở root (ViACaPu dùng cho get_metrics/print_metrics) lazy-import
`data_module` và `train` từ đây chỉ trong hàm CLI `main()` cũ. Để chạy CLI decode cũ,
cần thêm `legacy/` vào `PYTHONPATH`.
