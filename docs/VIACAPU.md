# ViACaPu — Acoustic-aware Casing & Punctuation (3.58M params)

Mô hình khôi phục **dấu câu** (punctuation) và **viết hoa** (casing) cho văn bản ASR
tiếng Việt, kết hợp **text + tín hiệu ngữ âm** (log-mel), tổng **3,583,369 tham số**.

Đây là model khuyến nghị của repo: checkpoint `exp_via_full/best_ac.pt`
(backup: `/mnt/hdd_ngocmx/Edge-Punct-Casing/checkpoints/exp_via_full/best_ac.pt`).

**Vì sao acoustic:** dấu phẩy đi kèm khoảng lặng (pause), câu hỏi đi kèm ngữ điệu
lên giọng — thông tin không tồn tại trong text. Model text-only cùng kiến trúc đạt
Punct F1 0.822; thêm nhánh acoustic (+0.9M params) đẩy lên **0.921**, riêng COMMA
F1 từ 0.661 lên **0.872**.

**Alignment-free:** không cần timestamp từng từ từ ASR — cross-attention tự học
"soft alignment" giữa từ và frame audio, nên gắn được sau **bất kỳ** ASR nào chỉ
với (text đầu ra, log-mel đầu vào).

---

## 1. Kiến trúc

Định nghĩa trong [model_via_capu.py](model_via_capu.py), class `ViACaPu`
(cấu hình mặc định: `d=256, ac_gru_layers=1, cross_layers=1, n_heads=4,
adjacent_heads=False`).

```
─────────────────────── NHÁNH TEXT ───────────────────────
 BPE token ids (vocab 3500)          [B, L]
     │
     ▼
 Embedding d=256                     [B, L, 256]
     │
     ▼
 3 × (Conv1d k=3 + ReLU + residual + LayerNorm)
     │
     ▼
 BiLSTM 2 lớp (hidden 128/hướng)     [B, L, 256]
     │
     ▼
 gather vị trí token đầu mỗi từ  →  text_w  [B, W, 256]

───────────────────── NHÁNH ACOUSTIC ─────────────────────
 log-mel 80 dải, 1 frame/10ms        [B, Ta, 80]
     │
     ▼
 Conv1d 80→128 (stride 2) ─ ReLU ─ LayerNorm
 Conv1d 128→256 (stride 2) ─ ReLU ─ LayerNorm    (↓4× thời gian)
     │
     ▼
 BiGRU 1 lớp (hidden 128/hướng)  →  af  [B, Ta/4, 256]
     │
     └─► energy head (Linear 256→1): aux, khuyến khích mã hóa
         khoảng lặng / cường độ theo frame

──────────────────────── FUSION ──────────────────────────
 ctx = MultiheadAttention(query=text_w, key=value=af, 4 heads)
 z   = sigmoid(Linear([text_w ; ctx]))        ← gate học được
 fused = z · text_w + (1−z) · ctx             [B, W, 256]

 (mỗi TỪ tự truy vấn các frame audio quanh nó:
  "sau từ này có pause không? ngữ điệu thế nào?")

────────────────── WORD-LEVEL + OUTPUT ───────────────────
 BiLSTM 1 lớp (hidden 128/hướng)     [B, W, 256]
     ├─► Linear 256→4  →  case_logits   (LOWER/UPPER/CAP/MIX)
     └─► Linear 256→4  →  punct_logits  (NONE/COMMA/PERIOD/QUES)
```

### Phân bổ tham số

| Khối | Params | % |
|---|---|---|
| Embedding (3500×256) | 896,000 | 25.0% |
| Text BiLSTM (2 lớp) | 790,528 | 22.1% |
| Text Conv ×3 + LayerNorm | 592,128 | 16.5% |
| Acoustic encoder (2 conv + BiGRU + energy head) | 512,897 | 14.3% |
| Word-level BiLSTM | 395,264 | 11.0% |
| Fusion (cross-attention + gate) | 394,496 | 11.0% |
| 2 output heads | 2,056 | 0.1% |
| **Tổng** | **3,583,369** | 100% |

### Nhãn (4 lớp mỗi đầu)

| Case | Ý nghĩa | | Punct | Ý nghĩa |
|---|---|---|---|---|
| 0 LOWER | chữ thường | | 0 NONE | không dấu sau từ |
| 1 UPPER | IN HOA cả từ | | 1 COMMA | `,` (gộp `;` `:`) |
| 2 CAP | Viết Hoa đầu từ | | 2 PERIOD | `.` (gộp `!`) |
| 3 MIX | hỗn hợp (iPhone) | | 3 QUESTION | `?` |

---

## 2. Huấn luyện

- **Data:** Dolly 1000h tiếng Việt — 658,128 clip (audio + transcript), tách
  train/val 98/2 (seed 42). Features: log-mel 80 (float16 memmap ~53GB) + BPE
  vocab 3500 (`bpe_model/bpe.model`).
- **Loss:** `CE(case) + 0.7 × CE(punct, weight=[1.0, 1.6, 1.05, 1.4])`
  — class-weight chống lệch phân bố (COMMA/QUESTION hiếm).
  Chỉ tính trên vị trí từ thật (bỏ `<s>`, `</s>`, padding).
- **Optimizer:** Adam lr 8e-4, weight decay 5e-5, CosineAnnealing 12 epochs,
  batch 64.
- **Chi phí:** 12 epochs ≈ **1h27ph trên 1× RTX 3060 12GB** (433s/epoch).

Lệnh tái lập:

```bash
.venv/bin/python train_via_capu_full.py --use_acoustic 1 --epochs 12 \
  --batch_size 64 --exp_dir exp_via_full
# ablation text-only (cùng kiến trúc, tắt nhánh acoustic):
.venv/bin/python train_via_capu_full.py --use_acoustic 0 --epochs 12 \
  --batch_size 64 --exp_dir exp_via_full
```

---

## 3. Kết quả (val Dolly, 13,162 clip / 278,990 từ)

### So sánh model

| Model | Params | Case F1 | Punct F1 |
|---|---|---|---|
| Text-only (cùng kiến trúc, tắt acoustic) | 2.68M | 0.938 | 0.822 |
| **ViACaPu acoustic (model này)** | **3.58M** | **0.943** | **0.921** |
| ViACaPu-L text (scale 2× + adjacent heads) | 5.34M | 0.937 | 0.820 |
| ViACaPu-L acoustic (fusion 2 tầng) | 8.90M | 0.937 | 0.818 |

### Per-class punctuation

| Lớp | Precision | Recall | F1 | So với text-only |
|---|---|---|---|---|
| COMMA | 0.856 | 0.889 | **0.872** | +0.211 |
| PERIOD | 0.982 | 0.956 | **0.969** | +0.007 |
| QUESTION | 0.802 | 0.755 | **0.778** | +0.037 |

Chi tiết trực quan: [figures/confusion_punct_4models.png](figures/confusion_punct_4models.png),
[figures/punct_f1_comparison.png](figures/punct_f1_comparison.png).

### Phát hiện chính từ confusion matrix

1. **COMMA hưởng lợi lớn nhất từ acoustic:** text-only bỏ sót 33.5% dấu phẩy
   (nhầm thành NONE); acoustic giảm còn 10.2% — đúng giả thuyết pause.
2. **QUESTION nhầm chủ yếu sang PERIOD** (~17%): model biết có kết câu, ngữ điệu
   giúp phân biệt loại (recall 70.3% → 75.2%).
3. **Casing gần như không cần audio** (0.938 → 0.943): viết hoa là bài toán
   thuần text.

### Bài học ablation (ViACaPu-L)

Scale kiến trúc lên 8.9M với fusion 2 tầng gate **thất bại**: gate đóng trước khi
cross-attention kịp học alignment ("gate collapse"), nhánh acoustic chết, model
hoạt động như text-only. Kết luận: **fusion phải nông (1 tầng), capacity vừa phải**
— 3.58M là điểm ngọt trên bộ data này.

---

## 4. Suy luận

Input cần: (1) chuỗi từ ASR đầu ra (chữ thường), (2) log-mel 80 của đoạn audio
tương ứng. Xem `evaluate()` trong [train_via_capu.py](train_via_capu.py) và
collate format (tokens, word_pos, mel, mel_len).

```python
from model_via_capu import ViACaPu
import torch, sentencepiece as spm

sp = spm.SentencePieceProcessor(); sp.load("bpe_model/bpe.model")
model = ViACaPu(sp.get_piece_size(), d=256, use_acoustic=True)
ck = torch.load("exp_via_full/best_ac.pt", map_location="cpu")
model.load_state_dict(ck["model"]); model.eval()
# batch: xem hàm collate trong train_via_capu.py
case_logits, punct_logits, _ = model(batch)
```
