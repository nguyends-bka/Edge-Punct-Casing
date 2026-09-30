import numpy as np
from pathlib import Path

# đường dẫn tuyệt đối tính từ gốc project -> chạy từ đâu cũng được
ROOT = Path(__file__).resolve().parent.parent   # .../Edge-Punct-Casing
DATA_DIR = ROOT / "data/data_audio_full"

# --- 1. meta.npz: metadata (offset, độ dài, nhãn...) ---
meta = np.load(DATA_DIR / "meta.npz")
print("=== meta.npz ===")
print("Các field:", meta.files)
for k in meta.files:
    v = meta[k]
    print(f"  {k:10s} shape={v.shape} dtype={v.dtype}")

n_mels = int(meta["n_mels"])
n_clips = len(meta["mel_len"])
print(f"\nSố clip: {n_clips}")
print(f"n_mels (số chiều Mel): {n_mels}")

# --- 2. mel.f16: mảng nhị phân thô, KHÔNG có header ---
# Toàn bộ mel của mọi clip được nối liền theo trục thời gian thành 1 mảng
# [tổng_frame, n_mels], lưu dạng float16. Muốn biết clip thứ i nằm ở đâu
# phải dùng mel_off[i] (frame bắt đầu) và mel_len[i] (số frame) từ meta.npz.
mel_path = DATA_DIR / "mel.f16"
file_size = mel_path.stat().st_size
total_frames = file_size // (2 * n_mels)   # 2 byte/số (float16) * n_mels cột

print(f"\n=== mel.f16 ===")
print(f"Kích thước file: {file_size / 1e9:.2f} GB")
print(f"Tổng số frame (suy từ size): {total_frames:,}")
print(f"Tổng số frame (theo meta):   {int(meta['mel_off'][-1] + meta['mel_len'][-1]):,}")

# mở dạng memmap: KHÔNG load hết 56GB vào RAM, chỉ ánh xạ file -> mảng ảo
mel = np.memmap(mel_path, dtype=np.float16, mode="r").reshape(-1, n_mels)
print(f"Shape toàn bộ mel (memmap): {mel.shape}")

# --- 3. cắt thử clip đầu tiên để xem cấu trúc thật ---
i = 0
off, length = int(meta["mel_off"][i]), int(meta["mel_len"][i])
clip0 = mel[off: off + length]   # [length, n_mels]
print(f"\n=== Clip #{i} ===")
print(f"mel_off={off}, mel_len={length} -> clip shape: {clip0.shape}")
print(f"Giá trị min/max/mean: {clip0.min():.3f} / {clip0.max():.3f} / {clip0.mean():.3f}")
print(f"5 giá trị đầu của frame 0: {clip0[0, :5]}")
