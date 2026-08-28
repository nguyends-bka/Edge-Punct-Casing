#!/usr/bin/env python3
"""Acoustic extraction for VoxPopuli (English) into ViACaPu memmap format.

Reads the LOCAL parquet files downloaded to --data_dir (facebook/voxpopuli, en),
decodes each clip's `bytes` -> log-mel(80), turns `raw_text` (which keeps real
punctuation + casing) into (bpe tokens, valid mask, per-word case/punct labels),
and appends to one big float16 `mel.f16` + a compact `meta.npz` — identical layout
to extract_dolly_audio_full.py so train_via_capu_full.py works unchanged.

Resume with --start_file. Reuses the same BPE model (vocab 3500) as Dolly.
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_punct_case_data import convert_line

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
    ap.add_argument("--data_dir", default="/mnt/hdd_ngocmx/voxpopuli_en/en",
                    help="dir with train-*.parquet")
    ap.add_argument("--out_dir", default="/mnt/hdd_ngocmx/data_voxpopuli_en")
    ap.add_argument("--split", default="train", help="train / validation / test")
    ap.add_argument("--start_file", type=int, default=0)
    ap.add_argument("--max_sec", type=float, default=20.0)
    ap.add_argument("--min_words", type=int, default=3)
    ap.add_argument("--ckpt_every", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(); sp.load(args.bpe_model)
    files = sorted(Path(args.data_dir).glob(f"{args.split}-*.parquet"))
    assert files, f"không tìm thấy {args.split}-*.parquet trong {args.data_dir}"
    print(f"{len(files)} file {args.split}; BPE vocab {sp.get_piece_size()}", flush=True)

    mel_off, mel_len = [], []
    tok, tok_off, tok_len = [], [], []
    valid_all, case_all, punct_all = [], [], []
    lab_off, lab_len = [], []
    frame_cursor = tok_cursor = lab_cursor = 0

    mode = "ab" if args.start_file > 0 else "wb"
    melfh = open(out / "mel.f16", mode)
    t0 = time.time(); n_clip = skipped = 0
    for fi in range(args.start_file, len(files)):
        pf = pq.ParquetFile(files[fi])
        for rg in range(pf.metadata.num_row_groups):
            rows = pf.read_row_group(rg, columns=["audio", "raw_text"]).to_pylist()
            for row in rows:
                text = (row.get("raw_text") or "").strip()
                if len(text.split()) < args.min_words:
                    skipped += 1; continue
                enc = encode_clip(text, sp)
                if enc is None:
                    skipped += 1; continue
                try:
                    mel = wav_to_logmel(row["audio"]["bytes"])
                except Exception:
                    skipped += 1; continue
                if mel.shape[0] > args.max_sec * 100 or mel.shape[0] < 5:
                    skipped += 1; continue
                melfh.write(mel.tobytes())
                T = mel.shape[0]
                mel_off.append(frame_cursor); mel_len.append(T); frame_cursor += T
                t, v, c, p = enc
                tok.append(t); tok_off.append(tok_cursor); tok_len.append(len(t)); tok_cursor += len(t)
                valid_all.append(v); case_all.append(c); punct_all.append(p)
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
            print(f"[file {fi+1}/{len(files)}] {n_clip} clips ({skipped} skip), "
                  f"{frame_cursor/1e6:.1f}M frames ({frame_cursor*N_MELS*2/1e9:.1f}GB mel), "
                  f"{rate:.0f} clip/s", flush=True)

    melfh.close()
    print(f"DONE: {n_clip} clips -> {out}/mel.f16 + meta.npz", flush=True)


if __name__ == "__main__":
    main()
