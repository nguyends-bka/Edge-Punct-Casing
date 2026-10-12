#!/usr/bin/env python3
"""Chấm ViACaPu trên TẬP VALID NỘI BỘ của repo cơ sở, để giải thích khoảng cách
0.703 / 0.569 mà bài báo chưa lý giải được.

Vấn đề: cùng một checkpoint mô hình cơ sở cho F1 dấu câu 0.703 trên tập valid nội
bộ của repo gốc (12 978 đoạn) nhưng chỉ 0.569 trên tập kiểm định chung của chúng
tôi (13 162 đoạn). Chênh 0.134 này LỚN HƠN cả đóng góp của nhánh ngữ âm (+0.107),
nên nếu không lý giải được thì độ tin cậy của mọi con số tuyệt đối trong bài bị
đặt dấu hỏi.

Cách kiểm chứng: chấm ViACaPu trên CHÍNH tập 12 978 đoạn đó.
  - Nếu ViACaPu cũng tăng tương ứng (giữ nguyên khoảng cách với mô hình cơ sở)
    -> đây chỉ là đặc tính của tập, vô hại.
  - Nếu khoảng cách thu hẹp đáng kể -> có vấn đề thật cần biết.

Tập valid của repo cơ sở chỉ có văn bản, không kèm âm thanh. Script khớp từng
dòng với clip Dolly tương ứng (99.2 % khớp được) để lấy mel.

Cách dùng:
    python -m tools.eval_on_base_valid --ckpt exp_seed/best_ac_s43_aux5.pt
    python -m tools.eval_on_base_valid --ckpt exp_seed/best_text_s43.pt --no-acoustic
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import MemmapClipDS, collate, to_dev   # noqa: E402
from src.metrics import get_metrics                  # noqa: E402
from src.model import ViACaPu                        # noqa: E402

CASE_LABELS = {0: "LOWER", 1: "UPPER", 2: "CAP", 3: "MIX_CASE"}
PUNCT_LABELS = {0: "NO_PUNCT", 1: "COMMA", 2: "PERIOD", 3: "QUESTION"}
BASE_VALID = "/mnt/hdd_ngocmx/Edge-Punct-Casing-base/data_dolly_clean/valid_text.txt"


def show(pred, true, label_map, title):
    precision, recall, f_scores, overall = get_metrics(pred, true)
    print(f"{title}:")
    for k in sorted(label_map):
        if k not in precision:
            continue
        f = f_scores.get(k)
        fs = f"{f:.3f}" if f is not None else "None"
        print(f"  {label_map[k]:10s} Prec [{precision[k]:.3f}]  "
              f"Rec [{recall.get(k, float('nan')):.3f}]  F1 [{fs}]")
    print(f"  {'Overall':10s} Prec [{overall[0]:.3f}]  "
          f"Rec [{overall[1]:.3f}]  F1 [{overall[2]:.3f}]")
    return overall[2]


def build_index(meta, sp, base_lines, cache: Path):
    """Khớp từng dòng văn bản của tập cơ sở với chỉ số clip trong Dolly."""
    if cache.exists():
        idx = np.load(cache)["idx"]
        print(f"dùng lại index đã khớp: {len(idx):,} clip")
        return idx
    tok, to, tl = meta["tok"], meta["tok_off"], meta["tok_len"]
    want = {s: j for j, s in enumerate(base_lines)}
    hit, t0 = {}, time.time()
    for i in range(len(tl)):
        ids = tok[int(to[i]):int(to[i]) + int(tl[i])]
        s = sp.decode([int(x) for x in ids]).strip()
        j = want.get(s)
        if j is not None and j not in hit:
            hit[j] = i
        if i % 100000 == 0:
            print(f"  khớp {len(hit):,}  ({i:,}/{len(tl):,}, {time.time()-t0:.0f}s)",
                  end="\r", flush=True)
    idx = np.array([hit[j] for j in sorted(hit)], dtype=np.int64)
    print(f"\nkhớp {len(idx):,}/{len(base_lines):,} = "
          f"{100*len(idx)/len(base_lines):.1f} %")
    np.savez(cache, idx=idx)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default=str(ROOT / "data/data_audio_full"))
    ap.add_argument("--base_valid", default=BASE_VALID)
    ap.add_argument("--bpe_model", default=str(ROOT / "bpe_model/bpe.model"))
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--no-acoustic", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--cache", default=str(ROOT / "exp_seed/base_valid_idx.npz"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(); sp.load(args.bpe_model)

    data_dir = Path(args.data_dir)
    npz = np.load(data_dir / "meta.npz")
    meta = {k: npz[k] for k in npz.files}
    n_mels = int(meta["n_mels"])
    mel = np.memmap(data_dir / "mel.f16", dtype=np.float16, mode="r").reshape(-1, n_mels)

    base_lines = [l.strip() for l in open(args.base_valid, encoding="utf-8")]
    print(f"tập valid repo cơ sở: {len(base_lines):,} dòng")
    idx = build_index(meta, sp, base_lines, Path(args.cache))

    # Cảnh báo rò rỉ: bao nhiêu clip trong tập này nằm trong TRAIN của chúng tôi?
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(meta["mel_len"]))
    n_val = int(len(meta["mel_len"]) * 0.02)
    our_val = set(perm[:n_val].tolist())
    overlap = sum(1 for i in idx if int(i) in our_val)
    print(f"trong đó nằm trong tập kiểm định của chúng tôi: {overlap:,} "
          f"({100*overlap/len(idx):.1f} %)")
    print(f"=> {len(idx)-overlap:,} clip ({100*(len(idx)-overlap)/len(idx):.1f} %) "
          f"nằm trong tập HUẤN LUYỆN -- điểm trên tập này do đó LẠC QUAN\n")

    use_ac = not args.no_acoustic
    model = ViACaPu(sp.get_piece_size(), d=args.d, use_acoustic=use_ac).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    print(f"checkpoint: {Path(args.ckpt).name}  "
          f"({'acoustic+text' if use_ac else 'text-only'}, {dev})")

    dl = DataLoader(MemmapClipDS(meta, mel, idx), batch_size=args.batch_size,
                    shuffle=False, collate_fn=collate, num_workers=2)
    cp, ct, pp, pt = [], [], [], []
    with torch.no_grad():
        for b in dl:
            b = to_dev(b, dev)
            cl, pl, _ = model(b)
            m = b["score_mask"]
            cp.append(cl.argmax(-1)[m].cpu().numpy()); ct.append(b["case"][m].cpu().numpy())
            pp.append(pl.argmax(-1)[m].cpu().numpy()); pt.append(b["punct"][m].cpu().numpy())
    cp, ct = np.concatenate(cp), np.concatenate(ct)
    pp, pt = np.concatenate(pp), np.concatenate(pt)
    print(f"đã chấm {len(cp):,} vị trí\n")

    cf = show(cp, ct, CASE_LABELS, "CASE")
    print()
    pf = show(pp, pt, PUNCT_LABELS, "PUNCT")
    print(f"\n==> CaseF1 {cf:.4f}   PunctF1 {pf:.4f}")


if __name__ == "__main__":
    main()
