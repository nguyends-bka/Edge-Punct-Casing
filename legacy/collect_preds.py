"""Run a checkpoint over the test set and dump (pred,true) for case & punct
to an .npz, for confusion-matrix plotting. Works for both the base arch and v4.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm
from tqdm import tqdm

from train import get_params, get_model
from data_module import DataModule


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--bpe_model", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="output .npz")
    p.add_argument("--arch", default="base", choices=["base", "v4"])
    p.add_argument("--max_seq_length", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--embedding_dim", type=int, default=100)
    p.add_argument("--hidden_size1", type=int, default=384)
    p.add_argument("--hidden_size2", type=int, default=384)
    p.add_argument("--top_bidirectional", type=int, default=0)
    p.add_argument("--conformer_layers", type=int, default=2)
    p.add_argument("--attn_window", type=int, default=8)
    return p


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    params = get_params()
    params.update(vars(args))
    params.arch = args.arch
    params.top_bidirectional = bool(args.top_bidirectional)
    params.use_conformer = params.use_crf = params.use_focal = True

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    sp = spm.SentencePieceProcessor(); sp.load(args.bpe_model)
    params.vocab_size = sp.get_piece_size()

    model = get_model(params)
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()

    dm = DataModule(args, sp)
    dl, _ = dm.test_dataloader()

    cp, ct, pp, pt = [], [], [], []
    for batch in tqdm(dl):
        batch = tuple(t.to(device) for t in batch)
        token_ids, label_ids, valid_ids, label_lens, label_masks = batch
        out = model(token_ids, valid_ids=valid_ids, label_lens=label_lens.clone())
        # same (non-stable) sort the model uses internally -> labels stay aligned
        _, indx = label_lens.sort(dim=0, descending=True)

        if len(out) == 4:  # v4: logits are UNMASKED [B,T,K]; punct may be a CRF path
            case_logits, punct_out, final_mask, is_crf = out
            T = final_mask.shape[1]
            case_pred = case_logits.argmax(dim=-1)[final_mask]
            punct_pred = punct_out[final_mask] if is_crf else punct_out.argmax(dim=-1)[final_mask]
        else:  # base Model_new: logits already masked to [N,K]
            active_case_logits, active_punct_logits, final_mask = out
            T = final_mask.shape[1]
            case_pred = active_case_logits.argmax(dim=-1)
            punct_pred = active_punct_logits.argmax(dim=-1)

        lab = label_ids[indx][:, :, :T]
        cp.append(case_pred.cpu().numpy()); ct.append(lab[:, 0, :][final_mask].cpu().numpy())
        pp.append(punct_pred.cpu().numpy()); pt.append(lab[:, 1, :][final_mask].cpu().numpy())

    np.savez_compressed(
        args.out,
        case_pred=np.concatenate(cp), case_true=np.concatenate(ct),
        punct_pred=np.concatenate(pp), punct_true=np.concatenate(pt),
    )
    print(f"wrote {args.out}: {len(np.concatenate(cp))} tokens")


if __name__ == "__main__":
    main()
