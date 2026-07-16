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
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from model_via_capu import ViACaPu
from decode import get_metrics, print_metrics, punct_id, case_id
import logging
from utils import setup_logger


class ClipDS(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    B = len(batch)
    Lmax = max(e["tokens"].shape[0] for e in batch)
    Wmax = max(e["case"].shape[0] for e in batch)
    Tmax = max(e["mel"].shape[0] for e in batch)

    tokens = torch.zeros(B, Lmax, dtype=torch.long)
    tok_mask = torch.zeros(B, Lmax, dtype=torch.bool)
    word_pos = torch.zeros(B, Wmax, dtype=torch.long)
    word_mask = torch.zeros(B, Wmax, dtype=torch.bool)
    score_mask = torch.zeros(B, Wmax, dtype=torch.bool)
    case = torch.zeros(B, Wmax, dtype=torch.long)
    punct = torch.zeros(B, Wmax, dtype=torch.long)
    mel = torch.zeros(B, Tmax, 80, dtype=torch.float32)
    mel_len = torch.zeros(B, dtype=torch.long)

    for b, e in enumerate(batch):
        L = e["tokens"].shape[0]; W = e["case"].shape[0]; T = e["mel"].shape[0]
        tokens[b, :L] = e["tokens"]; tok_mask[b, :L] = True
        wp = e["valid"].nonzero(as_tuple=False).squeeze(1)  # [W]
        word_pos[b, :W] = wp; word_mask[b, :W] = True
        score_mask[b, :W] = True
        score_mask[b, 0] = False; score_mask[b, W - 1] = False  # drop <s>/</s>
        case[b, :W] = e["case"]; punct[b, :W] = e["punct"]
        mel[b, :T] = e["mel"].float(); mel_len[b] = T
    return dict(tokens=tokens, tok_mask=tok_mask, word_pos=word_pos, word_mask=word_mask,
                score_mask=score_mask, case=case, punct=punct, mel=mel, mel_len=mel_len)


def to_dev(batch, dev):
    return {k: v.to(dev) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    cp, ct, pp, pt = [], [], [], []
    for batch in dl:
        batch = to_dev(batch, dev)
        cl, pl, _ = model(batch)
        m = batch["score_mask"]
        cp.append(cl.argmax(-1)[m].cpu().numpy()); ct.append(batch["case"][m].cpu().numpy())
        pp.append(pl.argmax(-1)[m].cpu().numpy()); pt.append(batch["punct"][m].cpu().numpy())
    cp, ct = np.concatenate(cp), np.concatenate(ct)
    pp, pt = np.concatenate(pp), np.concatenate(pt)

    def overall_f1(pred, true):
        _, _, _, ov = get_metrics(pred, true)
        return ov[2]
    return overall_f1(cp, ct), overall_f1(pp, pt), (cp, ct, pp, pt)


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
