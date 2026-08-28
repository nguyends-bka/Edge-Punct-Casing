"""Train / evaluate ViA-CaPu on the small audio prototype dataset.

Run twice for a clean ablation:
  --use_acoustic 0   text-only baseline (same text branch & heads)
  --use_acoustic 1   text + acoustic cross-attention fusion
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sentencepiece as spm

from .model import ViACaPu
from .data import ClipDS, collate, to_dev
from .metrics import evaluate, get_metrics, print_metrics, punct_id, case_id
import logging
from .utils import setup_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_audio")
    ap.add_argument("--bpe_model", default="bpe_model/bpe.model")
    ap.add_argument("--exp_dir", default="exp_via")
    ap.add_argument("--use_acoustic", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--punct_weights", type=str, default="1.0,1.6,1.05,1.4")
    args = ap.parse_args()

    Path(args.exp_dir).mkdir(exist_ok=True)
    setup_logger(f"{args.exp_dir}/log-via-{'ac' if args.use_acoustic else 'text'}")
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(42)

    sp = spm.SentencePieceProcessor(); sp.load(args.bpe_model)
    tr = DataLoader(ClipDS(f"{args.data_dir}/train.pt"), batch_size=args.batch_size,
                    shuffle=True, collate_fn=collate)
    va = DataLoader(ClipDS(f"{args.data_dir}/val.pt"), batch_size=args.batch_size,
                    shuffle=False, collate_fn=collate)

    model = ViACaPu(sp.get_piece_size(), d=256, use_acoustic=bool(args.use_acoustic),
                    dropout=args.dropout).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    logging.info(f"use_acoustic={args.use_acoustic} | params={npar:,} ({npar/1e6:.2f}M)")

    pw = torch.tensor([float(x) for x in args.punct_weights.split(",")], device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = (0, 0)
    for ep in range(args.epochs):
        model.train()
        tot = 0
        for batch in tr:
            batch = to_dev(batch, dev)
            cl, pl, _ = model(batch)
            m = batch["score_mask"].reshape(-1)
            closs = F.cross_entropy(cl.reshape(-1, cl.shape[-1])[m], batch["case"].reshape(-1)[m])
            ploss = F.cross_entropy(pl.reshape(-1, pl.shape[-1])[m], batch["punct"].reshape(-1)[m], weight=pw)
            loss = closs + 0.7 * ploss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        cf, pf, _ = evaluate(model, va, dev)
        if cf + pf > best[0] + best[1]:
            best = (cf, pf)
            torch.save({"model": model.state_dict()}, f"{args.exp_dir}/best_{'ac' if args.use_acoustic else 'text'}.pt")
        logging.info(f"ep {ep:02d} | train_loss {tot/len(tr):.3f} | val CaseF1 {cf:.3f} PunctF1 {pf:.3f}"
                     + ("  *best*" if best == (cf, pf) else ""))

    logging.info(f"\nBEST use_acoustic={args.use_acoustic}: CaseF1 {best[0]:.3f} PunctF1 {best[1]:.3f}")
    # detailed report of best-of-final
    cf, pf, (cp, ct, pp, pt) = evaluate(model, va, dev)
    logging.info("CASE (final):"); print_metrics(logging, *get_metrics(cp, ct), case_id)
    logging.info("PUNCT (final):"); print_metrics(logging, *get_metrics(pp, pt), punct_id)


if __name__ == "__main__":
    main()
