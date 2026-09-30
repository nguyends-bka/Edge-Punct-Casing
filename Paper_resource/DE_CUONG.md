# Đề cương nghiên cứu: ViACaPu — Khôi phục dấu câu & viết hoa cho ASR tiếng Việt, nhẹ và nhận biết ngữ âm

## 1. Giới thiệu

- **Bối cảnh:** hệ thống ASR (nhận dạng tiếng nói tự động) tiếng Việt thường trả về văn bản
  không dấu câu, không viết hoa. Khôi phục dấu câu và viết hoa (Capitalization and Punctuation
  restoration — CaPu) là bước hậu xử lý bắt buộc để văn bản dễ đọc và dùng được cho các tác vụ
  hạ nguồn (dịch máy, tóm tắt, trợ lý ảo).
- **Động lực đề tài:** các mô hình CaPu chất lượng cao hiện nay (BERT/wav2vec-based) quá nặng
  để chạy trên thiết bị biên (edge device — điện thoại, loa thông minh, thiết bị nhúng); trong
  khi các mô hình nhẹ hiện có (ví dụ You & Li 2024, CNN-BiLSTM chỉ dùng văn bản) hy sinh nhiều
  chất lượng, đặc biệt ở lớp dấu phẩy (COMMA) vốn phụ thuộc ngữ điệu/khoảng lặng khi nói mà văn
  bản thuần không thể hiện được.
- **Bài toán:** cho chuỗi từ chưa dấu câu/chưa viết hoa (đầu ra ASR) và tín hiệu âm thanh gốc
  tương ứng, dự đoán nhãn viết hoa và nhãn dấu câu cho từng từ.
- **Mục tiêu của nghiên cứu:**
  1. Thiết kế một mô hình CaPu vừa **nhẹ** (phù hợp ngân sách tham số cho thiết bị biên) vừa
     **chất lượng cao**, đặc biệt cải thiện lớp dấu phẩy.
  2. Khai thác **tín hiệu ngữ âm** (log-Mel) như một nguồn thông tin bổ sung cho nhánh văn bản,
     thay vì chỉ dựa vào ngữ pháp/từ vựng.
  3. Xác thực trên dữ liệu tiếng Việt thực tế quy mô lớn (Dolly ~1000 giờ) với điều kiện đầu
     vào giống thực tế triển khai.
- **Đóng góp chính (dự kiến trình bày):**
  - Kiến trúc hai nhánh (text + acoustic) hợp nhất bằng cross-attention có cổng (gated
    cross-attention), alignment-free, giữ tổng số tham số nhỏ hơn baseline nhưng chất lượng
    cao hơn đáng kể.
  - Phát hiện thực nghiệm quan trọng: **mất mát phụ (auxiliary energy loss)**, chứ không phải
    việc khởi tạo thiên lệch cổng (gate-bias init), mới là yếu tố loại bỏ tính bất ổn theo seed
    của nhánh ngữ âm — làm rõ cơ chế "cổng kiểm soát LƯỢNG thông tin, mất mát phụ đảm bảo
    CHẤT LƯỢNG thông tin".
  - Phát hiện "gate collapse": tăng dung lượng mô hình (d=256→384) làm chất lượng dấu câu TỤT,
    cho thấy chất lượng không đơn thuần đi theo số tham số.
  - Đánh giá đa seed (3 seeds), tách dev/test tránh rò rỉ, và khai báo minh bạch giả định đầu
    vào (transcript tham chiếu, WER=0) cùng giới hạn liên quan.

## 2. Các nghiên cứu gần đây

- **CaPu chỉ dùng văn bản:**
  - You & Li (2024, arXiv:2407.13142) — mô hình CNN-BiLSTM đa nhiệm, nhẹ (7 MB ONNX lượng tử
    hoá), cố tình KHÔNG dùng self-attention vì thấy không cải thiện (thậm chí giảm nhẹ F1 dấu
    câu) trên cấu hình của họ. Đây là baseline kiến trúc mà ViACaPu kế thừa và mở rộng.
  - Các mô hình dựa trên BERT/transformer lớn cho CaPu văn bản: chất lượng cao nhưng không phù
    hợp ngân sách biên, và không tận dụng được thông tin ngôn điệu (prosody) vì chỉ thấy văn bản.
- **CaPu khai thác tín hiệu tiếng nói (audio-aware):**
  - UniPunc — dùng text làm query, tín hiệu âm thanh (wav2vec 2.0) làm key/value trong
    cross-attention, hợp nhất bằng cộng residual cố định (không có cổng học được); BERT + wav2vec
    ở d=768, động lực chính là xử lý modality-missing (thiếu một trong hai luồng) chứ không phải
    ràng buộc tài nguyên biên.
  - Các hướng dùng đặc trưng ngôn điệu tường minh (pause duration, pitch, energy) kết hợp bộ
    phân loại nhẹ hơn — hiệu quả nhưng thường cần bộ trích đặc trưng riêng, khó tối ưu đầu-cuối.
- **CaPu cho tiếng Việt:**
  - JointCapPunc / ViCapPunc — huấn luyện và đánh giá trên dữ liệu **văn bản** (Vinmec,
    Alobacsi, ISOFHCARE — hỏi đáp y tế), không phải dữ liệu tiếng nói; do đó không so sánh trực
    tiếp được với các kết quả trên ngữ liệu ASR như Dolly.
  - Trinh và cộng sự (JST-SSAD, cùng nhóm nghiên cứu, cùng dữ liệu Dolly ~1000 giờ) — tích hợp
    CaPu vào một mô hình RNN-T 68M tham số (huấn luyện lại toàn bộ ASR), đạt punct micro-F1
    0.94, cao hơn kết quả hiện tại của ViACaPu (0.927). Đây là hướng "tích hợp sâu vào ASR",
    khác với hướng "hậu xử lý bolt-on" của ViACaPu (nhẹ hơn ASR khoảng 19 lần).
- **Khoảng trống chung rút ra từ khảo sát:** phần lớn công trình audio-aware nhắm vào chất
  lượng tối đa (mô hình lớn, không ràng buộc tài nguyên) hoặc phần lớn công trình nhẹ lại bỏ
  hoàn toàn tín hiệu âm thanh; hiếm công trình vừa giữ ngân sách tham số cho thiết bị biên vừa
  khai thác ngữ âm một cách có kiểm soát (gate) và có cơ chế huấn luyện ổn định qua nhiều seed.

## 3. Khoảng trống nghiên cứu

1. **Thiếu mô hình vừa nhẹ vừa audio-aware:** các mô hình nhẹ hiện có (You & Li) bỏ hoàn toàn
   tín hiệu ngữ âm; các mô hình audio-aware hiện có (UniPunc, RNN-T tích hợp) không ràng buộc
   ngân sách tham số cho thiết bị biên hoặc yêu cầu huấn luyện lại toàn bộ ASR.
2. **Cơ chế hợp nhất cố định, chưa kiểm soát được lượng/chất thông tin ngữ âm:** UniPunc hợp
   nhất bằng cộng residual cố định — không có cơ chế học "khi nào nên tin vào ngữ âm". Chưa có
   nghiên cứu tách bạch rõ vai trò của "cổng lưu lượng" (gate) và "chất lượng biểu diễn ngữ âm"
   (được đảm bảo qua mất mát phụ) trong việc ổn định huấn luyện.
3. **Thiếu đánh giá độ ổn định (đa seed) và kiểm soát rò rỉ dev/test** trong các công trình CaPu
   tiếng Việt hiện có — kết quả thường báo cáo một lần chạy, khó biết mức độ dao động do khởi
   tạo ngẫu nhiên, đặc biệt quan trọng khi nhánh ngữ âm dùng encoder chưa được giám sát trực
   tiếp.
4. **Giả định đầu vào không tường minh:** nhiều công trình không nói rõ liệu thực nghiệm dùng
   transcript tham chiếu (WER=0) hay đầu ra ASR thực tế (có lỗi nhận dạng) — ảnh hưởng lớn đến
   khả năng khái quát của kết quả khi triển khai thực tế.
5. **Thiếu hiểu biết về mối quan hệ giữa dung lượng mô hình và chất lượng khi thêm nhánh ngữ
   âm** — liệu tăng kích thước mô hình có luôn cải thiện chất lượng hay tồn tại hiện tượng
   "một nhánh lấn át nhánh còn lại" (gate collapse).

→ Nghiên cứu này (ViACaPu) nhắm lấp các khoảng trống (1)-(5): thiết kế kiến trúc nhẹ + có cổng
học được, xác định rõ thành phần nào (gate-bias vs. mất mát phụ) chịu trách nhiệm cho ổn định,
đánh giá đa seed với tách dev/test nghiêm ngặt, và khai báo minh bạch giả định WER=0 cùng ablation
dung lượng mô hình.

## 4. Đề xuất phương pháp

- **Phát biểu bài toán:** đầu vào là chuỗi từ (từ ASR, chưa dấu câu/viết hoa) + đặc trưng
  log-Mel của đoạn âm thanh tương ứng; đầu ra là nhãn viết hoa và nhãn dấu câu cho mỗi từ
  (gán ở cấp từ, không cần alignment tường minh giữa từ và khung âm thanh).
- **Kiến trúc tổng quan (ViACaPu, d=256, ~3.58M tham số):**
  - *Nhánh văn bản:* Embedding (d=256) → 3 lớp Conv1d dạng residual → BiLSTM 2 lớp → gộp biểu
    diễn theo từng từ.
  - *Nhánh ngữ âm:* log-Mel 80 chiều (16kHz, cửa sổ 25ms, bước nhảy 10ms, n_fft=400) → 2 lớp
    Conv1d stride-2 (giảm 4 lần độ dài chuỗi khung) → BiGRU; có thêm đầu phụ (auxiliary head)
    dự đoán năng lượng tín hiệu.
  - *Hợp nhất:* cross-attention có cổng (gated cross-attention) — query là biểu diễn từ, key/
    value là biểu diễn khung ngữ âm; alignment-free, cổng học được kiểm soát lượng thông tin
    ngữ âm chảy vào biểu diễn từ.
  - *Bộ giải mã & hai đầu ra:* BiLSTM cấp từ dùng chung, sau đó tách hai đầu tuyến tính (mỗi
    đầu 4 lớp): đầu viết hoa ghép trước dấu câu, đầu dấu câu ghép sau — phản ánh thứ tự phụ
    thuộc ngữ pháp giữa hai nhãn.
- **Hàm mục tiêu:** tổng có trọng số của CE(viết hoa) + 0.7·CE(dấu câu, có class-weight theo
  tần suất lớp: dấu phẩy được tăng trọng số) + mất mát phụ dự đoán năng lượng tín hiệu ngữ âm
  (nhân tố ổn định hoá nhánh ngữ âm — trọng số 0.5).
- **Cơ chế được đề xuất là điểm mới cốt lõi:**
  - Tách bạch vai trò của **gate-bias init** (kiểm soát *lượng* thông tin ngữ âm cho đi qua ban
    đầu) và **mất mát phụ** (đảm bảo *chất lượng*/tính hữu ích của biểu diễn ngữ âm) — chứng
    minh bằng lưới thí nghiệm 2×2 (bật/tắt từng can thiệp) rằng chỉ mất mát phụ mới loại bỏ
    được tính bất ổn theo seed của nhánh ngữ âm; gate-bias một mình không tạo cải thiện nào.
  - Kiểm soát ngân sách tham số: toàn bộ nhánh ngữ âm + cơ chế hợp nhất chỉ thêm ~0.91M tham
    số so với mô hình chỉ-văn-bản, giữ tổng số tham số nhỏ hơn nhiều so với baseline You & Li
    (7.26M) trong khi chất lượng cao hơn.
- **Giả định và giới hạn được khai báo tường minh:** tất cả thực nghiệm dùng transcript tham
  chiếu đã bỏ dấu/chữ hoa làm đầu vào văn bản (tương đương giả định ASR có WER=0); đây là điều
  kiện lý tưởng hoá cần nêu rõ trong phần giới hạn và hướng phát triển (đánh giá với đầu ra ASR
  thật có lỗi nhận dạng).

## 5. Thí nghiệm

- **Dữ liệu:** Dolly-1000h tiếng Việt (HuggingFace `dolly-vn/dolly-audio-1000h-vietnamese`),
  658.128 đoạn âm thanh, ~973 giờ, ~15,3 triệu nhãn từ case/punct. Tách tập validation cố định
  bằng seed 42 (2%, 13.162 đoạn); tập này được chia đôi lại (dev/test, 6.581/6.581) bằng cùng
  `RandomState(42)` để việc chọn checkpoint trên dev không làm phóng đại kết quả trên test.
- **Baseline & mô hình so sánh:**
  1. You & Li (CNN-BiLSTM, chỉ văn bản) — huấn luyện lại trên cùng dữ liệu Dolly, cùng bpe/vocab
     (bpe.model chung, vocab 3500) để đảm bảo so sánh công bằng.
  2. ViACaPu text-only (không nhánh ngữ âm) — kiểm soát để đo đóng góp riêng của phần ngữ âm.
  3. ViACaPu acoustic (mô hình đề xuất đầy đủ, d=256).
  4. Đối chiếu định tính (không so trực tiếp do khác điều kiện) với UniPunc, JointCapPunc/
     ViCapPunc, và Trinh et al. (RNN-T 68M cùng dữ liệu Dolly).
- **Chỉ số đánh giá:** micro-F1 cho viết hoa (case) và dấu câu (punct); F1 theo từng lớp dấu câu
  (đặc biệt COMMA, PERIOD, QUESTION) và độ chính xác/độ phủ (precision/recall) của COMMA — lớp
  khó nhất vì phụ thuộc ngôn điệu.
- **Thí nghiệm chính:**
  1. **So sánh tổng quan** baseline vs. text-only vs. acoustic trên cùng tập test.
  2. **Phân tích theo lớp** (per-class F1/precision/recall), tập trung vào lý do COMMA cải
     thiện mạnh khi thêm nhánh ngữ âm (khoảng lặng/pause khi nói).
  3. **Độ ổn định theo seed:** chạy 3 seed (42/43/44) cho từng cấu hình (text-only; acoustic
     không can thiệp; acoustic + chỉ gate-bias; acoustic + chỉ mất mát phụ; acoustic + cả hai),
     báo cáo mean ± độ lệch chuẩn mẫu (sample stdev), nhằm cô lập vai trò thật sự của mất mát
     phụ so với gate-bias.
  4. **Ablation dung lượng mô hình** (d=256 vs d=384) để chỉ ra hiện tượng "gate collapse" —
     dung lượng lớn hơn không đảm bảo chất lượng cao hơn.
  5. **Kiểm tra rò rỉ dev/test:** so sánh kết quả trên dev và test để xác nhận việc chọn
     checkpoint theo dev không làm sai lệch hệ thống kết quả test.
  6. **(Tuỳ chọn, mở rộng nếu kịp)** đo kích thước mô hình sau lượng tử hoá ONNX và độ trễ suy
     luận (ms) trên phần cứng đại diện thiết bị biên, để củng cố luận điểm "phù hợp triển khai
     biên"; ví dụ định tính đầu vào → đầu ra → nhãn thật để minh hoạ trực quan.
- **Phân tích bổ sung cần có trong bài:** thảo luận thẳng thắn về giả định WER=0 (giới hạn lớn
  nhất về tính tổng quát), so sánh trung thực với Trinh et al. (điểm số thấp hơn nhưng mô hình
  nhẹ hơn ~19 lần, không cần huấn luyện lại ASR) để định vị đóng góp là "đánh đổi hợp lý giữa
  chất lượng và chi phí triển khai" chứ không phải "vượt trội tuyệt đối".
