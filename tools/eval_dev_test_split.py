"""Chấm lại checkpoint đã huấn luyện trên hai nửa TÁCH BIỆT của tập val hiện tại.

Động cơ khoa học: pipeline huấn luyện chọn checkpoint tốt nhất theo F1 trên tập
val 2%, rồi bài báo BÁO CÁO trên chính tập đó -> điểm bị lạc quan hóa (rò rỉ tập
kiểm định). Script này chia tập val 13162 đoạn thành hai nửa tất định:
  - val_dev (nửa đầu) : vai trò tập chọn mô hình
  - test   (nửa sau)  : vai trò tập báo cáo, KHÔNG tham gia chọn checkpoint
Vì phép chia dùng cùng RandomState(42) như train.py, không cần huấn luyện lại;
ta chỉ chấm lại các checkpoint đã có trên từng nửa.

Lưu ý: checkpoint được chọn bằng toàn bộ val (gồm cả nửa test), nên điểm test ở
đây vẫn chưa hoàn toàn sạch -- nó là CHẶN TRÊN của mức rò rỉ, dùng để ĐO xem rò
rỉ lớn cỡ nào. Nếu chênh lệch dev-test nhỏ, kết luận của bài không bị ảnh hưởng.
"""
import argparse, json
import numpy as np
import torch
import sentencepiece as spm
from torch.utils.data import DataLoader

from src.data import MemmapClipDS, collate
from src.metrics import evaluate
from src.model import ViACaPu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/ai/ngocmx/Edge-Punct-Casing/data_audio_full")
    ap.add_argument("--bpe_model", default="bpe_model/bpe.model")
    ap.add_argument("--ckpt", required=True, nargs="+", help="các file .pt cần chấm")
    ap.add_argument("--use_acoustic", type=int, default=1)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default=None, help="ghi kết quả JSON ra file")
    args = ap.parse_args()

    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    npz = np.load(f"{args.data_dir}/meta.npz")
    meta = {k: npz[k] for k in npz.files}
    n_mels = int(meta["n_mels"])
    mel = np.memmap(f"{args.data_dir}/mel.f16", dtype=np.float16, mode="r").reshape(-1, n_mels)
    N = len(meta["mel_len"])

    # CHÍNH XÁC như train.py để hai nửa khớp với tập val đã dùng khi huấn luyện
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_val = int(N * args.val_frac)
    val_idx = perm[:n_val]
    half = len(val_idx) // 2
    splits = {"val_dev": val_idx[:half], "test": val_idx[half:]}

    sp = spm.SentencePieceProcessor(); sp.load(args.bpe_model)
    loaders = {
        name: DataLoader(MemmapClipDS(meta, mel, idx), batch_size=args.batch_size,
                         shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)
        for name, idx in splits.items()
    }

    results = {}
    for ck in args.ckpt:
        model = ViACaPu(sp.get_piece_size(), d=args.d,
                        use_acoustic=bool(args.use_acoustic)).to(dev)
        sd = torch.load(ck, map_location=dev)
        model.load_state_dict(sd["model"])
        model.eval()
        row = {}
        for name, dl in loaders.items():
            cf, pf, _ = evaluate(model, dl, dev)
            row[name] = {"case_f1": round(float(cf), 4), "punct_f1": round(float(pf), 4)}
        row["delta_punct_f1"] = round(row["val_dev"]["punct_f1"] - row["test"]["punct_f1"], 4)
        results[ck] = row
        print(f"{ck}\n  val_dev: case {row['val_dev']['case_f1']:.4f} punct {row['val_dev']['punct_f1']:.4f}"
              f"\n  test   : case {row['test']['case_f1']:.4f} punct {row['test']['punct_f1']:.4f}"
              f"\n  delta punct (dev-test): {row['delta_punct_f1']:+.4f}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
