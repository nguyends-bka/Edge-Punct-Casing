"""Aggregate evaluation for the v4 arch (Conformer-lite + CRF + Focal).

Handles the CRF branch where the model returns punctuation *predictions*
(Viterbi path) instead of logits. Labels are sorted with the exact same call
the model uses internally so predictions and labels stay aligned.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm

from train import get_params, get_model
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
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--hidden_size1", type=int, default=384)
    p.add_argument("--hidden_size2", type=int, default=384)
    p.add_argument("--conformer_layers", type=int, default=2)
    p.add_argument("--attn_window", type=int, default=8)
    p.add_argument("--use_conformer", type=int, default=1)
    p.add_argument("--use_crf", type=int, default=1)
    p.add_argument("--use_focal", type=int, default=1)
    p.add_argument("--dump", type=str, default=None, help="optional .npz to save preds/labels")
    return p


@torch.no_grad()
def run(args):
    args.exp_dir = Path(args.exp_dir)
    params = get_params()
    params.update(vars(args))
    params.arch = "v4"
    params.use_conformer = bool(args.use_conformer)
    params.use_crf = bool(args.use_crf)
    params.use_focal = bool(args.use_focal)
    torch.manual_seed(42)
    setup_logger(f"{params.exp_dir}/log-eval-v4")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    sp = spm.SentencePieceProcessor(); sp.load(args.bpe_model)
    params.vocab_size = sp.get_piece_size()

    model = get_model(params)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device); model.eval()
    logging.info(f"Loaded {args.ckpt} | params={sum(p.numel() for p in model.parameters())}")

    dm = DataModule(args, sp)
    dl, _ = dm.test_dataloader()

    cp, ct, pp, pt = [], [], [], []
    for batch in tqdm(dl):
        batch = tuple(t.to(device) for t in batch)
        token_ids, label_ids, valid_ids, label_lens, label_masks = batch
        out = model(token_ids, valid_ids=valid_ids, label_lens=label_lens.clone())
        case_logits, punct_out, final_mask, is_crf = out
        T = case_logits.shape[1]

        # replicate the model's internal (non-stable) sort so labels align
        _, indx = label_lens.sort(dim=0, descending=True)
        label_ids = label_ids[indx][:, :, :T]

        case_pred = case_logits.argmax(dim=-1)[final_mask]
        case_true = label_ids[:, 0, :][final_mask]
        if is_crf:
            punct_pred = punct_out[final_mask]
        else:
            punct_pred = punct_out.argmax(dim=-1)[final_mask]
        punct_true = label_ids[:, 1, :][final_mask]

        cp.append(case_pred.cpu().numpy()); ct.append(case_true.cpu().numpy())
        pp.append(punct_pred.cpu().numpy()); pt.append(punct_true.cpu().numpy())

    cp = np.concatenate(cp); ct = np.concatenate(ct)
    pp = np.concatenate(pp); pt = np.concatenate(pt)
    logging.info(f"Total tokens: {len(cp)}")

    if args.dump:
        np.savez_compressed(args.dump, case_pred=cp, case_true=ct, punct_pred=pp, punct_true=pt)
        logging.info(f"dumped preds to {args.dump}")

    logging.info("\nCASE metrics:\n" + "-" * 70)
    print_metrics(logging, *get_metrics(cp, ct), case_id)
    logging.info("\nPUNCT metrics:\n" + "=" * 70)
    print_metrics(logging, *get_metrics(pp, pt), punct_id)


if __name__ == "__main__":
    run(get_parser().parse_args())
