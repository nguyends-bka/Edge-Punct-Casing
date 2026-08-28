"""Đọc dữ liệu ViACaPu: dataset + collate.

Hai dataset:
  - MemmapClipDS: đọc Dolly-1000h dạng memmap (mel.f16 + meta.npz), lazy nên ~56GB
    mel không vào RAM cùng lúc. Dùng cho train chính (src/train.py).
  - ClipDS: dataset prototype nhỏ (train.pt/val.pt đã tải sẵn vào RAM). Dùng cho
    thử nghiệm nhanh.

`collate` gộp batch (pad tokens/word/mel, dựng score_mask bỏ <s>/</s>).
`to_dev` chuyển batch lên thiết bị.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapClipDS(Dataset):
    """Dolly memmap: meta.npz (offsets/labels) + mel.f16 (np.memmap [N,80])."""
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
            "mel": torch.from_numpy(np.array(self.mel[mo:mo + ml])),        # float16 [T,80]
            "tokens": torch.from_numpy(m["tok"][to:to + tl].astype(np.int64)),
            "valid": torch.from_numpy(m["valid"][to:to + tl].astype(np.int64)),
            "case": torch.from_numpy(m["case"][lo:lo + ll].astype(np.int64)),
            "punct": torch.from_numpy(m["punct"][lo:lo + ll].astype(np.int64)),
        }


class ClipDS(Dataset):
    """Prototype dataset: một file .pt chứa list các sample đã tiền xử lý."""
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
        score_mask[b, 0] = False; score_mask[b, W - 1] = False  # bỏ <s>/</s>
        case[b, :W] = e["case"]; punct[b, :W] = e["punct"]
        mel[b, :T] = e["mel"].float(); mel_len[b] = T
    return dict(tokens=tokens, tok_mask=tok_mask, word_pos=word_pos, word_mask=word_mask,
                score_mask=score_mask, case=case, punct=punct, mel=mel, mel_len=mel_len)


def to_dev(batch, dev):
    return {k: v.to(dev) for k, v in batch.items()}
