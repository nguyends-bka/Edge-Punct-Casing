"""Train ViA-CaPu on the FULL Dolly memmap dataset (data_audio_full/).

Uses a lazy memmap Dataset so the ~56GB of mel never enters RAM at once.
Reuses the collate / eval / model from the prototype. Ablation via --use_acoustic.
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sentencepiece as spm
from model_via_capu import ViACaPu
from train_via_capu import collate, evaluate
from decode import get_metrics, print_metrics, punct_id, case_id
from utils import setup_logger


class MemmapClipDS(Dataset):
    def __init__(self, meta, mel, indices):
        self.m = meta
        self.mel = mel
        self.idx = indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        m = self.m
        mo, ml = int(m["mel_off"][i]), int(m["mel_len"][i])
        to, tl = int(m["tok_off"][i]), int(m["tok_len"][i])
        lo, ll = int(m["lab_off"][i]), int(m["lab_len"][i])
        return {
            "mel": torch.from_numpy(np.array(self.mel[mo:mo + ml])),           # float16 [T,80]
            "tokens": torch.from_numpy(m["tok"][to:to + tl].astype(np.int64)),
            "valid": torch.from_numpy(m["valid"][to:to + tl].astype(np.int64)),
            "case": torch.from_numpy(m["case"][lo:lo + ll].astype(np.int64)),
            "punct": torch.from_numpy(m["punct"][lo:lo + ll].astype(np.int64)),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_audio_full")
    ap.add_argument("--bpe_model", default="bpe_model/bpe.model")
    ap.add_argument("--exp_dir", default="exp_via_full")
    ap.add_argument("--use_acoustic", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight_decay", type=float, default=5e-5)
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--punct_weights", type=str, default="1.0,1.6,1.05,1.4")
    # --- scaling knobs (defaults reproduce the 3.58M ablation model) ---
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--ac_gru_layers", type=int, default=1)
    ap.add_argument("--cross_layers", type=int, default=1)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--adjacent_heads", type=int, default=0)
    ap.add_argument("--tag", type=str, default=None,
                    help="checkpoint tag; default 'ac'/'text' by --use_acoustic")
    args = ap.parse_args()

    Path(args.exp_dir).mkdir(exist_ok=True)
    tag = args.tag or ("ac" if args.use_acoustic else "text")
    setup_logger(f"{args.exp_dir}/log-{tag}")
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(42)

    npz = np.load(f"{args.data_dir}/meta.npz")
    meta = {k: npz[k] for k in npz.files}
    n_mels = int(meta["n_mels"])
    mel = np.memmap(f"{args.data_dir}/mel.f16", dtype=np.float16, mode="r").reshape(-1, n_mels)
    N = len(meta["mel_len"])
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_val = int(N * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    logging.info(f"clips: {N} | train {len(tr_idx)} val {len(val_idx)} | mel frames {mel.shape[0]/1e6:.1f}M")

    sp = spm.SentencePieceProcessor(); sp.load(args.bpe_model)
    tr = DataLoader(MemmapClipDS(meta, mel, tr_idx), batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    va = DataLoader(MemmapClipDS(meta, mel, val_idx), batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=args.num_workers)

    model = ViACaPu(sp.get_piece_size(), d=args.d, use_acoustic=bool(args.use_acoustic),
                    dropout=args.dropout, ac_gru_layers=args.ac_gru_layers,
                    cross_layers=args.cross_layers, n_heads=args.n_heads,
                    adjacent_heads=bool(args.adjacent_heads)).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    logging.info(f"use_acoustic={args.use_acoustic} | d={args.d} ac_gru={args.ac_gru_layers} "
                 f"cross={args.cross_layers} heads={args.n_heads} adj={args.adjacent_heads} "
                 f"| params {npar/1e6:.2f}M")

    pw = torch.tensor([float(x) for x in args.punct_weights.split(",")], device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = (0.0, 0.0)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for batch in tr:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            cl, pl, _ = model(batch)
            m = batch["score_mask"].reshape(-1)
            closs = F.cross_entropy(cl.reshape(-1, cl.shape[-1])[m], batch["case"].reshape(-1)[m])
            ploss = F.cross_entropy(pl.reshape(-1, pl.shape[-1])[m], batch["punct"].reshape(-1)[m], weight=pw)
            loss = closs + 0.7 * ploss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        cf, pf, _ = evaluate(model, va, dev)
        if cf + pf > sum(best):
            best = (cf, pf); torch.save({"model": model.state_dict()}, f"{args.exp_dir}/best_{tag}.pt")
        logging.info(f"ep {ep:02d} | {time.time()-t0:.0f}s | loss {tot/nb:.3f} | val CaseF1 {cf:.3f} PunctF1 {pf:.3f}"
                     + ("  *best*" if best == (cf, pf) else ""))

    logging.info(f"BEST use_acoustic={args.use_acoustic}: CaseF1 {best[0]:.3f} PunctF1 {best[1]:.3f}")
    cf, pf, (cp, ct, pp, pt) = evaluate(model, va, dev)
    logging.info("CASE (final):"); print_metrics(logging, *get_metrics(cp, ct), case_id)
    logging.info("PUNCT (final):"); print_metrics(logging, *get_metrics(pp, pt), punct_id)


if __name__ == "__main__":
    main()
