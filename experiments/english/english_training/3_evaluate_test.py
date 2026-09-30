#!/usr/bin/env python3
"""Đánh giá checkpoint ViACaPu trên TEST split VoxPopuli English.

In precision/recall/F1 theo lớp (case + punct) và ma trận nhầm lẫn dấu câu.
Tái dùng model/collate/eval của project — không copy code.

Chạy:
  python 3_evaluate_test.py --ckpt exp_vox_en/best_en_ac.pt --use_acoustic 1
  python 3_evaluate_test.py --ckpt exp_vox_en/best_en_text.pt --use_acoustic 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = "/home/ai/nguyends/Edge-Punct-Casing"
sys.path.insert(0, PROJECT)
import sentencepiece as spm
from model_via_capu import ViACaPu
from train_via_capu import collate
from train_via_capu_full import MemmapClipDS
from decode import get_metrics, print_metrics, punct_id, case_id
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="/mnt/hdd_ngocmx/data_voxpopuli_en_test")
    ap.add_argument("--bpe_model", default=f"{PROJECT}/bpe_model/bpe.model")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--use_acoustic", type=int, default=1)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--ac_gru_layers", type=int, default=1)
    ap.add_argument("--cross_layers", type=int, default=1)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--adjacent_heads", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=48)
    args = ap.parse_args()

    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    npz = np.load(f"{args.test_dir}/meta.npz")
    meta = {k: npz[k] for k in npz.files}
    n_mels = int(meta["n_mels"])
    mel = np.memmap(f"{args.test_dir}/mel.f16", dtype=np.float16, mode="r").reshape(-1, n_mels)
    N = len(meta["mel_len"])
    logging.info(f"TEST: {N} clips")

    sp = spm.SentencePieceProcessor(); sp.load(args.bpe_model)
    dl = DataLoader(MemmapClipDS(meta, mel, np.arange(N)), batch_size=args.batch_size,
                    shuffle=False, collate_fn=collate, num_workers=4)

    model = ViACaPu(sp.get_piece_size(), d=args.d, use_acoustic=bool(args.use_acoustic),
                    ac_gru_layers=args.ac_gru_layers, cross_layers=args.cross_layers,
                    n_heads=args.n_heads, adjacent_heads=bool(args.adjacent_heads)).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()

    cp, ct, pp, pt = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            cl, pl, _ = model(batch)
            m = batch["score_mask"]
            cp.append(cl.argmax(-1)[m].cpu().numpy()); ct.append(batch["case"][m].cpu().numpy())
            pp.append(pl.argmax(-1)[m].cpu().numpy()); pt.append(batch["punct"][m].cpu().numpy())
    cp, ct = np.concatenate(cp), np.concatenate(ct)
    pp, pt = np.concatenate(pp), np.concatenate(pt)

    logging.info(f"\n=== {Path(args.ckpt).name} | use_acoustic={args.use_acoustic} | {len(pt):,} words ===")
    logging.info("CASE:");  print_metrics(logging, *get_metrics(cp, ct), case_id)
    logging.info("PUNCT:"); print_metrics(logging, *get_metrics(pp, pt), punct_id)

    # confusion matrix punct
    cm = np.zeros((4, 4), dtype=int)
    np.add.at(cm, (pt, pp), 1)
    logging.info("\nPunct confusion (hàng=true, cột=pred) [NONE,COMMA,PERIOD,QUES]:")
    for i, r in enumerate(cm):
        logging.info(f"  {['NONE','COMMA','PERIOD','QUES'][i]:7s} {r}")


if __name__ == "__main__":
    main()
