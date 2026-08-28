"""
Aggregate evaluation over the whole test set.

Unlike decode.py (which prints metrics per-batch), this collects predictions
across all batches and computes a single Precision/Recall/F1 report for
casing and punctuation. Reuses the model / data pipeline and metric helpers.

Usage:
  .venv/bin/python eval_all.py \
      --data_dir dataset/data --exp_dir exp --bpe_model bpe_model/bpe.model \
      --epoch 11 --batch_size 256
(--epoch N loads epoch-{N-1}.pt, matching decode.py convention.)
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm

from train import get_model, get_params
from data_module import DataModule
from decode import get_metrics, print_metrics, punct_id, case_id
from utils import setup_logger


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--bpe_model", required=True)
    p.add_argument("--max_seq_length", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--epoch", type=int, default=-1, help="loads epoch-{epoch-1}.pt")
    p.add_argument("--batch", type=int, default=-1, help="loads checkpoint-{batch}.pt")
    p.add_argument("--ckpt", type=str, default=None, help="explicit .pt path (overrides epoch/batch)")
    # arch overrides must match the trained checkpoint's shapes
    p.add_argument("--embedding_dim", type=int, default=100)
    p.add_argument("--hidden_size1", type=int, default=384)
    p.add_argument("--hidden_size2", type=int, default=384)
    p.add_argument("--top_bidirectional", type=lambda s: s.lower() in ("1", "true", "yes"), default=False)
    return p


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)
    params = get_params()
    params.update(vars(args))

    torch.manual_seed(42)
    setup_logger(f"{params.exp_dir}/log-eval-all")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    params.vocab_size = sp.get_piece_size()

    model = get_model(params)
    num_param = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of model parameters: {num_param}")

    if args.ckpt is not None:
        ptfile = args.ckpt
    elif params.epoch > 0:
        ptfile = f"{params.exp_dir}/epoch-{params.epoch-1}.pt"
    elif params.batch > 0:
        ptfile = f"{params.exp_dir}/checkpoint-{params.batch}.pt"
    else:
        raise SystemExit("Provide --ckpt or --epoch or --batch")
    logging.info(f"Loading checkpoint from {ptfile}")
    checkpoint = torch.load(ptfile, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    model.to(device)
    model.eval()

    data_module = DataModule(args, sp)
    decode_dl, test_file = data_module.test_dataloader()
    logging.info(f"test_file:{test_file}, len(decode_dl):{len(decode_dl)}")

    case_preds, case_tgts, punct_preds, punct_tgts = [], [], [], []

    for batch in tqdm(decode_dl):
        batch = tuple(t.to(device) for t in batch)
        token_ids, label_ids, valid_ids, label_lens, label_masks = batch

        active_case_logits, active_punct_logits, mask = model(
            token_ids, valid_ids=valid_ids, label_lens=label_lens
        )

        # model internally sorts sequences by label_lens (descending, stable);
        # reproduce that ordering on the labels so they line up with the logits.
        label_lens, indx = torch.sort(label_lens, dim=0, descending=True, stable=True)
        label_ids = label_ids[indx]
        label_ids = label_ids[:, :, : mask.shape[1]]

        case_pred = torch.argmax(F.log_softmax(active_case_logits, dim=1), dim=1)
        punct_pred = torch.argmax(F.log_softmax(active_punct_logits, dim=1), dim=1)

        active_case_labels = label_ids[:, 0, :][mask]
        active_punct_labels = label_ids[:, 1, :][mask]

        case_preds.append(case_pred.cpu().numpy())
        case_tgts.append(active_case_labels.cpu().numpy())
        punct_preds.append(punct_pred.cpu().numpy())
        punct_tgts.append(active_punct_labels.cpu().numpy())

    case_preds = np.concatenate(case_preds)
    case_tgts = np.concatenate(case_tgts)
    punct_preds = np.concatenate(punct_preds)
    punct_tgts = np.concatenate(punct_tgts)

    logging.info(f"Total evaluated tokens: {len(case_preds)}")

    pc, rc, fc, oc = get_metrics(case_preds, case_tgts)
    pp, rp, fp, op = get_metrics(punct_preds, punct_tgts)

    logging.info("\nCASE metrics (aggregated over full test set):\n" + "-" * 70)
    print_metrics(logging, pc, rc, fc, oc, case_id)
    logging.info("\nPUNCT metrics (aggregated over full test set):\n" + "=" * 70)
    print_metrics(logging, pp, rp, fp, op, punct_id)


if __name__ == "__main__":
    main()
