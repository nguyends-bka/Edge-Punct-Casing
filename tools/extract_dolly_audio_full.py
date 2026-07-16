#!/usr/bin/env python3
"""Scalable acoustic extraction for the FULL Dolly 1000h set.

Streams every parquet file, decodes WAV -> log-mel(80), and appends the mel
frames sequentially to one big float16 binary (`mel.f16`) while recording a
compact per-clip index + BPE tokens + word case/punct labels in `meta.npz`.
Nothing is held fully in RAM except the small meta arrays, so it scales to
664k clips (~40GB mel on disk). Meta is checkpointed every few files so a
crash loses at most one file of progress; re-run with --start_file to resume.

Output dir layout:
  data_audio_full/mel.f16     raw float16, all mels concatenated [total_frames,80]
  data_audio_full/meta.npz    offsets + tokens/valid/case/punct (concatenated)
"""
import argparse
import io
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
import sentencepiece as spm
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_punct_case_data import convert_line

BASE = "datasets/dolly-vn/dolly-audio-1000h-vietnamese@refs/convert/parquet/default/train"
SR = 16000
N_MELS = 80
melT = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=400, hop_length=160, win_length=400, n_mels=N_MELS)


def wav_to_logmel(b):
    wav, sr = sf.read(io.BytesIO(b), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    mel = melT(wav.unsqueeze(0)).squeeze(0)
    mel = torch.log(mel.clamp_min(1e-5)).transpose(0, 1)
    mel = (mel - mel.mean(0, keepdim=True)) / (mel.std(0, keepdim=True) + 1e-5)
    return mel.numpy().astype(np.float16)


def encode_clip(text, sp):
    words, case, punct = convert_line(text)
    if not words:
        return None
    toks = [sp.piece_to_id("<s>")]; valid = [1]
    for w in words:
        wt = sp.encode(w, out_type=int) or [sp.unk_id()]
        toks.extend(wt); valid.extend([1] + [0] * (len(wt) - 1))
    toks.append(sp.piece_to_id("</s>")); valid.append(1)
    case = [0] + case + [0]; punct = [0] + punct + [0]
    return (np.array(toks, np.int32), np.array(valid, np.int8),
            np.array(case, np.int8), np.array(punct, np.int8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe_model", default="bpe_model/bpe.model")
    ap.add_argument("--out_dir", default="data_audio_full")
    ap.add_argument("--n_files", type=int, default=332)
    ap.add_argument("--start_file", type=int, default=0)
    ap.add_argument("--max_sec", type=float, default=16.0)
    ap.add_argument("--ckpt_every", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    sp = spm.SentencePieceProcessor(); sp.load(args.bpe_model)
    fs = HfFileSystem()
    files = sorted(f["name"] for f in fs.ls(BASE, detail=True))[: args.n_files]

    mel_off, mel_len = [], []
    tok, tok_off, tok_len = [], [], []
    valid_all, case_all, punct_all = [], [], []
    lab_off, lab_len = [], []
    frame_cursor = tok_cursor = lab_cursor = 0

    mode = "ab" if args.start_file > 0 else "wb"
    melfh = open(out / "mel.f16", mode)
    t0 = time.time()
    n_clip = 0
    for fi in range(args.start_file, len(files)):
        pf = pq.ParquetFile(fs.open(files[fi]))
        for rg in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(rg, columns=["audio", "text"]).to_pylist():
                enc = encode_clip(row["text"], sp)
                if enc is None:
                    continue
                try:
                    mel = wav_to_logmel(row["audio"]["bytes"])
                except Exception:
                    continue
                if mel.shape[0] > args.max_sec * 100 or mel.shape[0] < 5:
                    continue
                melfh.write(mel.tobytes())
                T = mel.shape[0]
                mel_off.append(frame_cursor); mel_len.append(T); frame_cursor += T
                t, v, c, p = enc
                tok.append(t); tok_off.append(tok_cursor); tok_len.append(len(t)); tok_cursor += len(t)
                valid_all.append(v)
                case_all.append(c); punct_all.append(p)
                lab_off.append(lab_cursor); lab_len.append(len(c)); lab_cursor += len(c)
                n_clip += 1
        if (fi + 1) % args.ckpt_every == 0 or fi == len(files) - 1:
            melfh.flush()
            np.savez(out / "meta.npz",
                     mel_off=np.array(mel_off, np.int64), mel_len=np.array(mel_len, np.int32),
                     tok=np.concatenate(tok), tok_off=np.array(tok_off, np.int64), tok_len=np.array(tok_len, np.int32),
                     valid=np.concatenate(valid_all),
                     case=np.concatenate(case_all), punct=np.concatenate(punct_all),
                     lab_off=np.array(lab_off, np.int64), lab_len=np.array(lab_len, np.int32),
                     n_mels=N_MELS)
            rate = n_clip / (time.time() - t0)
            print(f"[file {fi+1}/{len(files)}] {n_clip} clips, {frame_cursor/1e6:.1f}M frames "
                  f"({frame_cursor*N_MELS*2/1e9:.1f}GB mel), {rate:.0f} clip/s", flush=True)

    melfh.close()
    print(f"DONE: {n_clip} clips -> {out}/mel.f16 + meta.npz")


if __name__ == "__main__":
    main()
