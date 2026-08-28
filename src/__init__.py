"""ViACaPu — mô hình nhẹ nhận biết ngữ âm cho khôi phục dấu câu & viết hoa.

Module:
  model    — kiến trúc (ViACaPu, AcousticEncoder)
  data     — dataset (MemmapClipDS/ClipDS) + collate  [ĐỌC DATA]
  metrics  — get_metrics/print_metrics + evaluate       [ĐÁNH GIÁ]
  train    — vòng huấn luyện chính trên Dolly memmap     [CHẠY]
  utils    — tiện ích (logger, scheduler...)
"""
from .model import ViACaPu, AcousticEncoder
