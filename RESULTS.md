# Kết quả huấn luyện & Kiến trúc mô hình (Dolly 1000h)

Tài liệu này ghi lại quá trình huấn luyện mô hình dự đoán **dấu câu (punctuation)** và
**viết hoa (word casing)** dạng text-only trên bộ dữ liệu tiếng Việt
[`dolly-vn/dolly-audio-1000h-vietnamese`](https://huggingface.co/datasets/dolly-vn/dolly-audio-1000h-vietnamese),
cùng các cải tiến kiến trúc và số liệu đánh giá.

> Lưu ý phạm vi: đây là mô hình **hậu xử lý trên văn bản** (nhận chuỗi token chữ → gán nhãn
> case/punct cho từng từ). Nó **khác** với mô hình end-to-end acoustic trong bài báo JST
> (Zipformer + RNN-T, 68M params, micro-F1 0.94) vốn dự đoán dấu câu trực tiếp từ **ngữ âm**.

---

## 1. Kiến trúc mô hình

Mô hình `Model_new` ([model.py](model.py)) — chuỗi xử lý cho mỗi câu:

```
 token ids (BPE, vocab 3500)
        │
        ▼
 Embedding (dim = D)                         D = 100 (gốc) → 256 (cải tiến)
        │
        ▼
 Encoder: 3 × ConvLayer                      Conv1d(k=3, padding=same)
   ├─ Conv1d → ReLU                          + residual (Add) + LayerNorm
   └─ Add & LayerNorm
        │
        ▼
 BiLSTM  (2 lớp, hidden=384, bi)  → 768
        │
        ▼
 LSTM tầng trên (hidden=384)      → 384      (tùy chọn bidirectional → 768)
        │
        ├──────────────► ghép với token TRƯỚC ──► Dropout ──► Linear ──► case_logits (4)
        │
        └──────────────► ghép với token SAU   ──► Dropout ──► Linear ──► punct_logits (4)
```

**Ý tưởng then chốt — ghép token liền kề:**
- **Case** phụ thuộc ngữ cảnh phía **trước** (đầu câu → viết hoa) → mỗi vị trí ghép với vector token đứng trước.
- **Punct** phụ thuộc ngữ cảnh phía **sau** (từ tiếp theo báo hiệu ranh giới) → ghép với vector token đứng sau.
- Nhóm tác giả đã thử self-attention nhưng gây lỗi kiểu "How? are? you?" nên giữ cơ chế ghép cục bộ này.

**Nhãn (4 lớp mỗi đầu):**

| Case | Ý nghĩa | | Punct | Ý nghĩa |
|---|---|---|---|---|
| 0 | LOWER (thường) | | 0 | không dấu |
| 1 | UPPER (IN HOA) | | 1 | COMMA `,` (gồm `; :`) |
| 2 | CAP (Viết Hoa đầu) | | 2 | PERIOD `.` (gồm `!`) |
| 3 | MIX_CASE | | 3 | QUESTION `?` |

**Hàm mất mát:** `loss = case_loss + 0.7 × punct_loss` (CrossEntropy, tùy chọn có trọng số lớp).
**Tối ưu:** Adam (lr 0.002) + ReduceLROnPlateau. Chuỗi dài cố định 200 token (đóng gói nhiều câu ngắn vào 1 cửa sổ).

---

## 2. Quy trình dữ liệu (chỉ dùng Dolly)

| Bước | Công cụ | Kết quả |
|---|---|---|
| 1. Trích text (bỏ audio) | [tools/extract_dolly_text.py](tools/extract_dolly_text.py) | `raw/all_dolly.txt` — 664,125 dòng |
| 2. Làm sạch | [tools/clean_dolly.py](tools/clean_dolly.py) | `raw/all_dolly_clean.txt` — 648,916 dòng |
| 3. Tạo nhãn + chia tập | [tools/prepare_punct_case_data.py](tools/prepare_punct_case_data.py) | `data_dolly_clean/` |

- **Trích text:** chỉ đọc cột `text` từ 332 file parquet qua HuggingFace range-read (audio ~472MB/file được **bỏ qua**), 16 luồng, ~6 phút.
- **Làm sạch:** chuẩn hóa NFC, bỏ dòng có ký tự ngoại lai (14), bỏ dòng rác <3 từ (23), **khử 15,172 dòng trùng**. Giữ lại **97.7%**.
- **Chia tập** (seed 42, valid/test 2%): **622,960 train / 12,978 valid / 12,978 test**.
- **BPE:** tái dùng `bpe_model/bpe.model` (vocab 3500).

Phân bố nhãn (train): punct — không dấu 90.8%, COMMA 4.15%, PERIOD 4.81%, QUESTION 0.24%
(≈33K câu hỏi tuyệt đối); case — LOWER 91.7%, CAP 8.17%, UPPER 0.12%, MIX 0.01%.

---

## 3. Các phiên bản & cải tiến

| Phiên bản | Cải tiến so với trước | Params |
|---|---|---|
| **v1** | Cấu hình gốc: embedding 100, dropout 0.5, loss không trọng số | 7.26M |
| **v2** | Data đã làm sạch + embedding **256** + dropout **0.3** | 8.78M |
| **v2w** | v2 + **class-weight punct** `[1.0, 1.6, 1.05, 1.4]` + train dài hơn | 8.78M |
| **v3** | v2w + **bidirectional** LSTM tầng trên | 10.56M |

Tất cả cải tiến đều bật/tắt qua **tham số dòng lệnh**, mặc định giữ nguyên hành vi gốc
(`--embedding_dim`, `--dropout`, `--top_bidirectional`, `--punct_weights`, `--case_weights`).

---

## 4. Kết quả đánh giá (tập test Dolly, ~302K token)

### Tổng quan (micro-F1)

| Phiên bản | Params | Case F1 | Punct F1 | COMMA | PERIOD | QUESTION |
|---|---|---|---|---|---|---|
| v1 (dolly gốc)¹ | 7.26M | 0.856 | 0.692 | 0.558 | ~0.81 | ~0.67 |
| v2 | 8.78M | 0.858 | 0.704 | 0.589 | 0.812 | 0.642 |
| **v2w** ⭐ | 8.78M | **0.875** | **0.725** | 0.625 | 0.827 | 0.691 |
| v3 | 10.56M | **0.876** | **0.730** | 0.630 | 0.830 | 0.681 |

¹ v1 đánh giá trên tập test trước khi làm sạch (split hơi khác) → so sánh tương đối, không tuyệt đối.

### Chi tiết mô hình khuyến nghị — **v2w** (`exp_dolly_v2w/epoch-13.pt`)

**Case (Overall F1 0.875)**
| Lớp | Precision | Recall | F1 |
|---|---|---|---|
| LOWER | 0.986 | 0.992 | 0.989 |
| UPPER | 0.861 | 0.679 | 0.760 |
| CAP | 0.904 | 0.850 | 0.876 |
| MIX_CASE | 1.000 | 0.821 | 0.902 |

**Punct (Overall F1 0.725)**
| Lớp | Precision | Recall | F1 |
|---|---|---|---|
| COMMA | 0.606 | 0.645 | 0.625 |
| PERIOD | 0.848 | 0.806 | 0.827 |
| QUESTION | 0.720 | 0.664 | 0.691 |

### Nhận định
- **Bộ ba data-sạch + embedding 256 + class-weight (v2w)** là phần cải thiện đáng giá nhất:
  Punct **0.692 → 0.725**, COMMA recall **0.48 → 0.65**, chỉ tốn thêm ~1.5M params.
- **Bidirectional LSTM tầng trên (v3)** chỉ thêm ~+0.005 Punct / +0.001 Case nhưng **+20% params, +30% thời gian** → **không đáng** cho thiết bị biên.
- **Đánh đổi COMMA:** tăng recall làm precision COMMA giảm còn ~0.61 (thỉnh thoảng thừa phẩy). Chỉnh trọng số nếu cần cân lại.

---

## 5. Cách tái lập

Môi trường: venv `.venv/` (torch CUDA, sentencepiece, numpy, tensorboard).

**Chuẩn bị dữ liệu**
```bash
.venv/bin/python tools/extract_dolly_text.py --out raw/all_dolly.txt --workers 16
.venv/bin/python tools/clean_dolly.py --input raw/all_dolly.txt --output raw/all_dolly_clean.txt
.venv/bin/python tools/prepare_punct_case_data.py \
  --input raw/all_dolly_clean.txt --output-dir data_dolly_clean \
  --valid-ratio 0.02 --test-ratio 0.02 --seed 42
```

**Huấn luyện (v2w — khuyến nghị)**
```bash
.venv/bin/python train.py --world-size 1 \
  --data_dir data_dolly_clean --exp_dir exp_dolly_v2w --bpe_model bpe_model/bpe.model \
  --embedding_dim 256 --dropout 0.3 --punct_weights "1.0,1.6,1.05,1.4" \
  --base-lr 0.002 --epochs 14 --batch_size 128
```
Thêm `--top_bidirectional True` để train v3. (~57–85s/epoch trên RTX 3060.)

**Đánh giá toàn tập test**
```bash
.venv/bin/python eval_all.py --data_dir data_dolly_clean --exp_dir exp_dolly_v2w \
  --bpe_model bpe_model/bpe.model --ckpt exp_dolly_v2w/epoch-13.pt \
  --embedding_dim 256 --batch_size 256
```

**Chạy suy luận trên một câu** (chữ thường, không dấu câu, mỗi câu 1 dòng)
```bash
.venv/bin/python decode_sentence.py --text_file ./example/input.txt \
  --exp_dir ./exp_dolly_v2w --bpe_model ./bpe_model/bpe.model \
  --embedding_dim 256 --epoch 14 2>/dev/null | tail -1
```
> ⚠️ Model cải tiến **bắt buộc** truyền `--embedding_dim 256` (và `--top_bidirectional True` cho v3),
> nếu không sẽ lệch shape khi load checkpoint. `--epoch N` nạp `epoch-{N-1}.pt`.

---

## 6. Hạn chế đã biết
- **Dolly gần như không có chữ số** → text nhiều số / văn bản trang trọng bị lệch phân bố (viết hoa loạn).
- **Câu cực dài** bị cắt thành nhiều cửa sổ 200 token, chất lượng giảm ở các cửa sổ sau.
- COMMA là lớp khó nhất (đặt phẩy vốn mơ hồ về mặt cú pháp).
- Muốn vượt trần của mô hình text-only cần bổ sung **tín hiệu ngữ âm** (pause/F0/energy) — xem thảo luận về hướng lai nhẹ.
