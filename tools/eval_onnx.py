#!/usr/bin/env python3
"""Chấm F1 cho mô hình ONNX trên tập kiểm định, để biết lượng tử hoá có làm giảm
chất lượng hay không.

Bài báo mới chỉ xác nhận mô hình INT8 *nạp và chạy được*, chưa chấm lại F1. Script
này lấp chỗ đó: dùng cùng phép chia tập (RandomState 42), cùng mặt nạ chấm điểm và
cùng hàm metric như `src/train.py`, nên con số ra so sánh trực tiếp được với các
bảng trong bài.

Cách dùng:
    python -m tools.eval_onnx --onnx ac_aux_int8.onnx
    python -m tools.eval_onnx --onnx ac_aux.onnx --limit 2000   # chạy thử nhanh
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import MemmapClipDS, collate   # noqa: E402
from src.metrics import get_metrics          # noqa: E402

CASE_LABELS = {0: "LOWER", 1: "UPPER", 2: "CAP", 3: "MIX_CASE"}
PUNCT_LABELS = {0: "NO_PUNCT", 1: "COMMA", 2: "PERIOD", 3: "QUESTION"}


def show(pred, true, label_map, title):
    precision, recall, f_scores, overall = get_metrics(pred, true)
    print(f"{title}:")
    for k in sorted(label_map):
        if k not in precision:
            continue
        f = f_scores.get(k)
        print(f"  {label_map[k]:10s} Prec [{precision[k]:.3f}]  "
              f"Rec [{recall.get(k, float('nan')):.3f}]  "
              f"F1 [{f:.3f}]" if f is not None else "")
    print(f"  {'Overall':10s} Prec [{overall[0]:.3f}]  "
          f"Rec [{overall[1]:.3f}]  F1 [{overall[2]:.3f}]")
    return overall[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--data_dir", default=str(ROOT / "data/data_audio_full"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0,
                    help="chỉ chấm N đoạn đầu (0 = toàn bộ tập kiểm định)")
    ap.add_argument("--threads", type=int, default=0, help="số luồng CPU (0 = mặc định)")
    args = ap.parse_args()

    import onnxruntime as ort

    data_dir = Path(args.data_dir)
    # Nạp hẳn vào dict như src/train.py: NpzFile đọc lười không an toàn với
    # DataLoader nhiều worker (lỗi BadZipFile khi nhiều tiến trình cùng đọc).
    npz = np.load(data_dir / "meta.npz")
    meta = {k: npz[k] for k in npz.files}
    n_mels = int(meta["n_mels"])
    mel = np.memmap(data_dir / "mel.f16", dtype=np.float16,
                    mode="r").reshape(-1, n_mels)

    # phép chia GIỐNG HỆT src/train.py
    n = len(meta["mel_len"])
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    n_val = int(n * 0.02)
    val_idx = perm[:n_val]
    if args.limit:
        val_idx = val_idx[: args.limit]
    print(f"tập kiểm định : {len(val_idx):,} đoạn")

    ds = MemmapClipDS(meta, mel, val_idx)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=2)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.threads:
        so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    print(f"ONNX          : {Path(args.onnx).name}  "
          f"({Path(args.onnx).stat().st_size / 1048576:.2f} MB)")
    print(f"inputs        : {sorted(names)}\n")

    cp, ct, pp, pt = [], [], [], []
    t0 = time.time()
    for bi, b in enumerate(dl):
        feed = {
            "tokens": b["tokens"].numpy().astype(np.int64),
            "tok_mask": b["tok_mask"].numpy(),
            "word_pos": b["word_pos"].numpy().astype(np.int64),
            "word_mask": b["word_mask"].numpy(),
            "mel": b["mel"].numpy().astype(np.float32),
            "mel_len": b["mel_len"].numpy().astype(np.int64),
        }
        feed = {k: v for k, v in feed.items() if k in names}
        cl, pl = sess.run(["case_logits", "punct_logits"], feed)

        m = b["score_mask"].numpy()
        cp.append(cl.argmax(-1)[m]); ct.append(b["case"].numpy()[m])
        pp.append(pl.argmax(-1)[m]); pt.append(b["punct"].numpy()[m])

        if bi % 10 == 0:
            done = min((bi + 1) * args.batch_size, len(val_idx))
            print(f"  {done:>6,}/{len(val_idx):,}  ({time.time() - t0:.0f}s)",
                  end="\r", flush=True)

    cp, ct = np.concatenate(cp), np.concatenate(ct)
    pp, pt = np.concatenate(pp), np.concatenate(pt)
    print(f"\nđã chấm {len(cp):,} vị trí trong {time.time() - t0:.0f}s\n")

    cf = show(cp, ct, CASE_LABELS, "CASE")
    print()
    pf = show(pp, pt, PUNCT_LABELS, "PUNCT")
    print(f"\n==> CaseF1 {cf:.4f}   PunctF1 {pf:.4f}")


if __name__ == "__main__":
    main()
