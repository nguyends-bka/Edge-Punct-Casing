#!/usr/bin/env python3
"""Build a SMALL audio+text prototype dataset for ViA-CaPu.

For each clip: decode the WAV bytes -> 16kHz mono -> log-mel(80); and turn the
punctuated transcript into (bpe token ids, valid mask marking word starts,
per-word case & punct labels). One clip = one training example (NOT packed into
200-token windows like the text-only pipeline).

Saves data_audio/{train,val}.pt as lists of dicts with tensors:
  mel [T,80], tokens [L], valid [L], case [W+2], punct [W+2]
(W+2 = words + <s>/</s>; label 0 for the two boundary slots.)
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
import sentencepiece as spm
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_punct_case_data import convert_line  # (words, case_labels, punct_labels)

BASE = "datasets/dolly-vn/dolly-audio-1000h-vietnamese@refs/convert/parquet/default/train"
SR = 16000
melT = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=400, hop_length=160, win_length=400, n_mels=80
)


def wav_to_logmel(b):
    wav, sr = sf.read(io.BytesIO(b), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    mel = melT(wav.unsqueeze(0)).squeeze(0)            # [80, T]
    mel = torch.log(mel.clamp_min(1e-5)).transpose(0, 1)  # [T,80]
    # per-utterance CMVN
    mel = (mel - mel.mean(0, keepdim=True)) / (mel.std(0, keepdim=True) + 1e-5)
    return mel.to(torch.float16)


def encode_clip(text, sp):
    words, case, punct = convert_line(text)
    if not words:
        return None
    toks = [sp.piece_to_id("<s>")]
    valid = [1]
    for w in words:
        wt = sp.encode(w, out_type=int)
        if not wt:
            wt = [sp.unk_id()]
        toks.extend(wt)
        valid.extend([1] + [0] * (len(wt) - 1))
    toks.append(sp.piece_to_id("</s>"))
    valid.append(1)
    case = [0] + case + [0]     # <s> ... </s>
    punct = [0] + punct + [0]
    assert sum(valid) == len(case) == len(punct)
    return (torch.tensor(toks, dtype=torch.long),
            torch.tensor(valid, dtype=torch.long),
            torch.tensor(case, dtype=torch.long),
            torch.tensor(punct, dtype=torch.long))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe_model", default="bpe_model/bpe.model")
    ap.add_argument("--n_files", type=int, default=3, help="how many parquet files to pull")
    ap.add_argument("--max_clips", type=int, default=6000)
    ap.add_argument("--max_sec", type=float, default=18.0, help="skip clips longer than this")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--out_dir", default="data_audio")
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(); sp.load(args.bpe_model)
    fs = HfFileSystem()
    files = sorted(f["name"] for f in fs.ls(BASE, detail=True))[: args.n_files]

    examples = []
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(fs.open(f))
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["audio", "text"]).to_pylist()
            for row in tbl:
                if len(examples) >= args.max_clips:
                    break
                b = row["audio"]["bytes"]
                enc = encode_clip(row["text"], sp)
                if enc is None:
                    continue
                try:
                    mel = wav_to_logmel(b)
                except Exception:
                    continue
                if mel.shape[0] > args.max_sec * 100:  # 100 frames/sec
                    continue
                toks, valid, case, punct = enc
                examples.append({"mel": mel, "tokens": toks, "valid": valid,
                                 "case": case, "punct": punct})
            print(f"  file {fi} rg {rg}: total {len(examples)} clips", flush=True)
            if len(examples) >= args.max_clips:
                break
        if len(examples) >= args.max_clips:
            break

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(examples))
    n_val = int(len(examples) * args.val_frac)
    val = [examples[i] for i in idx[:n_val]]
    train = [examples[i] for i in idx[n_val:]]

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    torch.save(train, out / "train.pt")
    torch.save(val, out / "val.pt")
    tot_words = sum(int(e["valid"].sum()) - 2 for e in examples)
    print(f"\nSaved {len(train)} train / {len(val)} val clips ({tot_words} words) to {out}/")


if __name__ == "__main__":
    main()
