# Trạng thái nghiên cứu và việc cần làm

*Cập nhật: 2026-09-30*

Ghi lại các quyết định khoa học và giới hạn đã biết, để người đọc sau (hoặc
chính tác giả trên máy khác) không phải suy luận lại từ đầu.

---

## 1. Kết quả hiện có

| Cấu hình | Params | Case F1 | Punct F1 | COMMA F1 |
|---|---|---|---|---|
| Chỉ văn bản | 2.68M | 0.937 ± .001 | 0.820 ± .002 | 0.660 ± .003 |
| Ngữ âm, không aux | 3.58M | 0.940 ± .003 | *lưỡng cực* | *lưỡng cực* |
| **Ngữ âm + aux** | **3.58M** | **0.944 ± .002** | **0.927 ± .004** | **0.888 ± .007** |

Kích thước triển khai: ONNX FP32 13.71 MB → **INT8 4.93 MB** (mô hình cơ sở
You & Li công bố 7 MB đã lượng tử hoá).

### Lưới tách biến (đóng góp phương pháp cốt lõi)

| Cấu hình | gate_bias | λ_aux | Punct F1 | seeds 42/43/44 |
|---|---|---|---|---|
| Chỉ văn bản | — | — | 0.820 ± .002 | .820 / .819 / .822 |
| Ngữ âm | mặc định | 0 | *0.854 ± .059* | **.922 / .819 / .820** |
| Ngữ âm + bias cổng | −2.0 | 0 | 0.818 ± .001 | .817 / .819 / .818 |
| Ngữ âm + aux | mặc định | 0.5 | 0.926 ± .003 | .924 / .929 / .924 |
| Ngữ âm + bias + aux | −2.0 | 0.5 | 0.927 ± .004 | .928 / .930 / .923 |

**Đọc bảng này cho đúng:** dòng "Ngữ âm" KHÔNG phải nhiễu quanh 0.854. Ba seed
rơi vào hai cụm tách biệt (0.922 hoặc ≈0.82), không seed nào ở giữa. Con số
0.854 không ứng với lần chạy nào — nó là trung bình của một phân bố lưỡng cực.

**Kết luận:** aux loss là *điều kiện đủ*; gate-bias là *không cần thiết*
(0.818 còn thấp hơn mức chỉ-văn-bản 0.820).

---

## 2. Bốn việc cần làm, theo thứ tự ưu tiên

### (1) Chấm ViACaPu trên tập valid của repo cơ sở — RẺ NHẤT, LÀM TRƯỚC

**Vấn đề:** cùng một checkpoint baseline cho 0.703 trên tập valid nội bộ của
repo cơ sở (12 978 đoạn) nhưng chỉ 0.569 trên tập kiểm định chung (13 162
đoạn). Khoảng cách 0.134 này **lớn hơn cả đóng góp của nhánh ngữ âm (+0.107)**
và hiện chưa lý giải được. Đã loại trừ: độ dài câu (21.6 vs 21.7 từ/câu), trùng
nội dung (hai tập rời nhau).

**Vì sao quan trọng:** nếu một mô hình dao động 0.134 chỉ vì đổi tập đánh giá
trong cùng corpus, thì độ tin cậy của MỌI con số tuyệt đối trong bài bị đặt dấu
hỏi — kể cả 0.927.

**Cách làm:** chấm ViACaPu trên chính tập 12 978 đoạn đó. Nếu ViACaPu cũng tăng
tương ứng (giữ nguyên khoảng cách với baseline) thì đây chỉ là đặc tính của
tập, vô hại. Nếu khoảng cách thu hẹp đáng kể thì có vấn đề thật cần biết.

**Chi phí:** một lần eval, vài phút. Cần `mel.f16`.

### (2) Huấn luyện lại baseline dưới đúng công thức của ViACaPu

**Vấn đề:** dòng baseline trong Bảng 3 KHÔNG phải so sánh có kiểm soát. Nó giữ
công thức huấn luyện gốc, khác ViACaPu ở ít nhất bốn biến:

- **(i)** baseline dùng `nn.CrossEntropyLoss()` **không trọng số lớp**, trong
  khi ViACaPu dùng `[1.0, 1.6, 1.05, 1.4]`. Với phân bố lệch 90.8% NONE, đây
  là can thiệp tác động trực tiếp lên xu hướng dám dự đoán lớp hiếm. F1 COMMA
  0.634 của baseline đi kèm recall thấp — đúng triệu chứng không bù mất cân bằng.
- **(ii)** baseline chấm điểm trên **toàn bộ** vị trí; ViACaPu loại hai từ biên
  khỏi cả loss lẫn phép chấm. Hai mô hình không được đánh giá trên cùng tập vị trí.
- **(iii)** dropout 0.5 so với 0.3.
- **(iv)** baseline cố ý tránh self-attention.

**(i)** và **(ii)** là khác biệt về *cách huấn luyện và cách chấm*, không phải
năng lực kiến trúc. Vì vậy bài **không** tuyên bố khoảng cách 0.569 → 0.820 đo
được ưu thế kiến trúc.

**Lưu ý quan trọng:** giới hạn này KHÔNG ảnh hưởng tới lưới tách biến và
ablation dung lượng — chúng chỉ so các biến thể ViACaPu dùng chung y hệt công
thức mất mát, mặt nạ chấm điểm và cách chia tập.

### (3) Ô đối chứng aux loss với mục tiêu phụ vô nghĩa

**Vấn đề:** dữ liệu cho thấy aux loss *có tác dụng*, nhưng chưa biết *vì sao*.
Hai giả thuyết chưa phân biệt được:

- **(a) Nội dung:** mục tiêu năng lượng gắn với khoảng lặng nên buộc encoder mã
  hoá đúng đặc trưng ngôn điệu mà tác vụ dấu câu cần.
- **(b) Tối ưu:** *bất kỳ* mục tiêu phụ nào cũng cho gradient trực tiếp và sớm,
  phá vòng luẩn quẩn "cổng chưa mở vì biểu diễn còn ngẫu nhiên, biểu diễn còn
  ngẫu nhiên vì cổng chưa mở" — nội dung là thứ yếu.

**Cách phân biệt:** chạy một ô với mục tiêu phụ KHÔNG mang thông tin ngôn điệu
(ví dụ hồi quy một phép chiếu ngẫu nhiên cố định của phổ đầu vào). Nếu vẫn đạt
≈0.926 thì cơ chế là (b); nếu quay về mức chỉ-văn-bản thì là (a).

Bài hiện phát biểu ở mức dữ liệu cho phép: *"giám sát trực tiếp bộ mã hoá ngữ
âm là thành phần tạo ra độ ổn định"*, không khẳng định lợi ích đến riêng từ
tính chất ngôn điệu.

### (4) Chấm lại F1 cho bản ONNX INT8

Đã xác nhận mô hình INT8 *nạp và chạy được* (4 input / 2 output), nhưng **chưa
chấm lại F1**. Lượng tử hoá có thể làm giảm chất lượng, nhất là với LSTM/GRU.
Cần `mel.f16`.

---

## 3. Giới hạn đã khai báo trong bài

Những điểm này đã nằm trong mục Hạn chế, liệt kê lại để tiện tra:

- **WER = 0.** Mọi thí nghiệm dùng bản ghi tham chiếu, không phải đầu ra ASR
  thật. Các con số là *chặn trên* của chất lượng triển khai.
- **Dolly là dữ liệu TTS.** Theo metadata chính thức (`tags: synthetic, tts`).
  Khoảng lặng mà mô hình khai thác có thể phản ánh quy luật của bộ tổng hợp chứ
  không phải ngôn điệu người nói thật.
- **Chia tập theo đoạn, không theo giọng.** Metadata không có định danh giọng
  hay văn bản nguồn nên không loại trừ được trùng lặp.
- **N = 3 seed.** Không đủ để kiểm định thống kê; bằng chứng mạnh nhất là chênh
  lệch ghép cặp theo seed nhất quán về dấu.
- **Ablation dung lượng d=384 chỉ 1 seed.** Kết luận đúng là "không quan sát
  được cải thiện", không phải "đã chứng minh không có".
- **Chưa đo latency/RTF/RAM** trên phần cứng biên. Kích thước ONNX INT8 là điểm
  dữ liệu triển khai duy nhất hiện có.
- **QUESTION không rút kết luận riêng** — lớp chỉ chiếm 0.24% token nên biến
  thiên giữa seed cùng bậc với hiệu ứng.

---

## 4. Tình trạng dữ liệu và backup

| Hạng mục | Dung lượng | Ở đâu |
|---|---|---|
| Code, paper, 19 checkpoint, log | 262 MB | GitHub + repo này |
| `meta.npz` (nhãn, offset, thống kê) | 129 MB | ngoài git |
| `mel.f16` (đặc trưng mel Dolly) | **52 GB** | **chỉ trên máy gốc** |
| `experiments/**/*.pt` (run cũ) | 18 GB | chỉ trên máy gốc, đã bị thay thế |

`mel.f16` chỉ cần khi **huấn luyện lại**, mà việc đó đòi GPU. Lấy lại bằng một
trong hai cách:

1. Copy từ máy gốc: `/mnt/hdd_ngocmx/Edge-Punct-Casing/data_audio_full/`
2. Trích lại từ HuggingFace: `python -m tools.extract_dolly_audio_full`
   (tải 147 GB parquet để sinh ra 52 GB mel — chậm hơn cách 1, và có rủi ro
   dataset upstream đã đổi nên kết quả không tái lập chính xác)

Kiểm tra project sau khi chuyển máy: `python tools/check_project.py`
