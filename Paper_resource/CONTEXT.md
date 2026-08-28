# CONTEXT — Bàn giao ngữ cảnh cho phiên Claude Code mới (trên Windows)

> **Cách dùng:** Mở Claude Code trong thư mục chứa file này (ví dụ `E:\2026\paper`),
> nói: *"Đọc CONTEXT.md để nắm ngữ cảnh dự án, rồi giúp tôi tiếp tục."*
> File này tóm tắt toàn bộ công việc đã làm ở phiên trước (trên server Linux, nay đã mất).

---

## 1. Dự án là gì

Nghiên cứu + viết bài báo IEEE về **ViACaPu** — mô hình **khôi phục dấu câu (punctuation)
và viết hoa (casing)** cho ASR tiếng Việt, **nhẹ + nhận biết ngữ âm**, chạy trên thiết bị biên.

- Kế thừa từ paper gốc **You & Li 2024 (arXiv:2407.13142, Cisco)** — CNN+BiLSTM chỉ dùng văn bản.
- **Đóng góp chính:** mô hình CaPu vừa nhẹ vừa chất lượng cao. Ngữ âm (log-Mel) là **một
  phương pháp** để nâng chất lượng, không phải toàn bộ đóng góp.

## 2. Model chính (ViACaPu acoustic) — KẾT QUẢ CHỐT

Tất cả trên bộ **Dolly-1000h tiếng Việt** (658k clip, 973h, HuggingFace `dolly-vn/dolly-audio-1000h-vietnamese`):

| Model | Params | Case F1 | Punct F1 | COMMA F1 |
|---|---|---|---|---|
| Baseline You&Li [1] (train lại Dolly) | 7.26M | 0.856 | 0.703 | 0.577 |
| ViACaPu text-only | 2.68M | 0.938 | 0.822 | 0.661 |
| **ViACaPu acoustic** ⭐ | **3.58M** | **0.943** | **0.921** | **0.873** |

**Điểm bán hàng:** ViACaPu 3.58M NHỎ HƠN baseline 7.26M mà điểm CAO HƠN nhiều.
Thêm nhánh ngữ âm: Punct +0.099, COMMA +0.212, chỉ +0.91M params.

**Per-class ViACaPu acoustic:** COMMA 0.873 (P .856/R .892), PERIOD 0.969, QUESTION 0.778.
COMMA recall: text-only 0.659 → acoustic 0.892 (nhờ pause/khoảng lặng khi nói).

## 3. Kiến trúc ViACaPu (d=256, 3.58M)

- **Nhánh text:** Embedding 256 → 3×Conv1d residual → BiLSTM 2 lớp → gather đầu-từ.
- **Nhánh acoustic:** log-Mel 80 (16kHz, 25ms window, 10ms hop, n_fft 400) → 2×Conv stride-2
  (↓4×) → BiGRU. Có aux energy head.
- **Fusion:** gated cross-attention (query=từ, key/value=frame), alignment-free.
- **Heads:** BiLSTM word-level → case ghép từ TRƯỚC, punct ghép từ SAU (4 lớp mỗi đầu).
- **Loss:** CE(case) + 0.7·CE(punct), punct class-weight [1.0,1.6,1.05,1.4].

## 4. Phát hiện quan trọng (đã đưa vào paper)

- **d=256 là điểm ngọt:** tăng lên d=384 làm Punct F1 TỤT 0.921→0.821 ("gate collapse" —
  nhánh text rộng hơn hội tụ trước, gate đóng, acoustic ngừng đóng góp). → to hơn PHẢN tác dụng.
- **Vì sao baseline 7.26M kém hơn ViACaPu 3.58M:** (1) baseline thiếu ngữ âm; (2) cấu hình cũ
  (embedding 100, không class-weight); (3) params "sai chỗ". Chất lượng do LOẠI bằng chứng +
  thiết kế, KHÔNG do số params.

## 5. Thí nghiệm tiếng Anh (KHÔNG vào paper — chỉ tham khảo)

Đã thử VoxPopuli + LibriTTS (English). Kết luận: acoustic KHÔNG engage trên tiếng Anh vì
transcript là văn bản biên tập (dấu câu theo ngữ pháp, không theo pause). Củng cố giả thuyết
"acoustic gain phụ thuộc prosodic fidelity của data". Paper CHỐT chỉ dùng Dolly tiếng Việt.

## 6. Trạng thái PAPER (thư mục Paper_resource/)

- `paper_ieee.tex` — bản tiếng Anh (IEEEtran, ~455 dòng)
- `paper_ieee_vi.tex` — bản tiếng Việt (IEEEtran + gói vietnam, ~550 dòng)
- `figures/` — 6 ảnh màu chuẩn IEEE (PNG + PDF vector): fig0 kiến trúc, fig1 so sánh,
  fig2 F1 per-class, fig3/4 confusion punct/case, fig6 domain shift
- `PAPER.md` — bản Markdown xem nhanh

**Đã hoàn thành:**
- Cấu trúc IEEE 7 mục, framing "nhẹ + chất lượng cao"
- 14 tài liệu tham khảo (đã đọc PDF: You&Li, UniPunc, STPT-Samsung, Cho2022; còn lại từ search)
- Mô tả data chi tiết + bảng thống kê Dolly
- Lý giải mọi siêu tham số (0.7, d=256, class-weight)
- Công thức có chú thích đầy đủ ký hiệu
- Tác giả: Nguyen Doan Sy, Ngoc Mai Xuan, Vinh La The, Hiep Hoang Van, Hung Pham Ngoc,
  Thuan Nguyen Dinh — Hanoi University of Science and Technology
- Bảng đã sửa hết lỗi chồng (dùng \small + \setlength{\tabcolsep})

## 7. Việc CÒN LẠI có thể làm tiếp

**Bổ sung (tùy chọn, làm paper mạnh hơn):**
- [ ] Bảng ablation d=256 vs d=384 (số "gate collapse" — đã có kết quả, chỉ cần lập bảng)
- [ ] Bảng kích thước ONNX + tốc độ inference (ms) — nhấn "on-device", CHƯA đo
- [ ] Ví dụ định tính input→output→ground-truth (có sẵn trong test_sentence/ cũ)
- [ ] Verify chính xác tên tác giả + venue của các citation lấy từ web search (chưa đọc full)
- [ ] Điền email tác giả liên hệ nếu cần

**Trước khi nộp:**
- [ ] Compile trên Overleaf kiểm tra (máy không có LaTeX local)
- [ ] Kiểm tra ảnh không bị chồng/tràn sau compile

## 8. Lưu ý kỹ thuật

- **Data mel (mel.f16) ĐÃ MẤT** (HDD server bị xóa). Muốn train lại: tải lại từ HuggingFace
  bằng `tools/extract_dolly_audio_full.py` (có trong gói).
- Checkpoint còn: `exp_via_full/best_ac.pt` (model chính), `best_text.pt`, + English trong
  `english_training/exp_*/`. Baseline checkpoint ĐÃ MẤT (train lại 30ph bằng repo Rud-nin nếu cần).
- BPE: `bpe_model/bpe.model` (Việt, vocab 3500), `bpe_model_en/`, `bpe_model_libritts/`.
- Load checkpoint ViACaPu: `ViACaPu(vocab_size, d=256, use_acoustic=True)` từ `model_via_capu.py`.

## 9. Cách làm việc user mong muốn (quan trọng)

- Trả lời tiếng Việt.
- Trung thực về giới hạn: nếu chưa đọc full paper/chưa verify số → nói rõ, không bịa.
- Mọi số liệu trong paper phải từ thực nghiệm thật (log/checkpoint), không phóng đại.
- So sánh phải cùng tập dữ liệu (đã bỏ các so sánh chéo tập không hợp lệ).
