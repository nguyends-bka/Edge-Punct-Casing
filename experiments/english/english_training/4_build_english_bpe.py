#!/usr/bin/env python3
"""Cải tiến A: BPE tiếng Anh riêng + retokenize KHÔNG cần đọc lại audio.

1. Decode từng từ từ meta.npz hiện có (dùng BPE tiếng Việt + valid mask) -> corpus.
2. Train SentencePiece vocab 3500 trên corpus tiếng Anh.
3. Re-encode từng từ bằng BPE mới -> meta_v2.npz (nhãn case/punct GIỮ NGUYÊN,
   mel.f16 GIỮ NGUYÊN — symlink, không copy 22GB).

Chạy:  python 4_build_english_bpe.py
"""
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

PROJECT = Path("/home/ai/nguyends/Edge-Punct-Casing")
HERE = PROJECT / "english_training"
SRC_TRAIN = PROJECT / "data_voxpopuli_en_ssd"
SRC_TEST = PROJECT / "data_voxpopuli_en_test_ssd"
DST_TRAIN = PROJECT / "data_voxpopuli_en_ssd_v2"
DST_TEST = PROJECT / "data_voxpopuli_en_test_ssd_v2"
BPE_VI = PROJECT / "bpe_model/bpe.model"
BPE_EN_PREFIX = HERE / "bpe_en"
VOCAB = 3500


def clip_words(meta, sp_vi, i):
    """Tách token của clip i thành từng TỪ theo valid mask, decode ra chữ."""
    to, tl = int(meta["tok_off"][i]), int(meta["tok_len"][i])
    toks = meta["tok"][to:to + tl]
    valid = meta["valid"][to:to + tl]
    # bỏ <s> đầu và </s> cuối
    words, cur = [], []
    for t, v in zip(toks[1:-1], valid[1:-1]):
        if v == 1 and cur:
            words.append(cur); cur = []
        cur.append(int(t))
    if cur:
        words.append(cur)
    return [sp_vi.decode(w) for w in words]


def load_meta(d):
    npz = np.load(d / "meta.npz")
    return {k: npz[k] for k in npz.files}


def retokenize(meta, sp_vi, sp_en, name):
    tok, tok_off, tok_len, valid_all = [], [], [], []
    cursor = 0
    N = len(meta["mel_len"])
    bos, eos = sp_en.piece_to_id("<s>"), sp_en.piece_to_id("</s>")
    for i in range(N):
        words = clip_words(meta, sp_vi, i)
        lo, ll = int(meta["lab_off"][i]), int(meta["lab_len"][i])
        n_words_label = ll - 2
        assert len(words) == n_words_label, f"clip {i}: {len(words)} words vs {n_words_label} labels"
        toks = [bos]; valid = [1]
        for w in words:
            wt = sp_en.encode(w, out_type=int) or [sp_en.unk_id()]
            toks.extend(wt); valid.extend([1] + [0] * (len(wt) - 1))
        toks.append(eos); valid.append(1)
        tok.append(np.array(toks, np.int32))
        valid_all.append(np.array(valid, np.int8))
        tok_off.append(cursor); tok_len.append(len(toks)); cursor += len(toks)
        if (i + 1) % 20000 == 0:
            print(f"  [{name}] {i+1}/{N}", flush=True)
    out = dict(meta)  # giữ nguyên mel_*, lab_*, case, punct, n_mels
    out["tok"] = np.concatenate(tok)
    out["tok_off"] = np.array(tok_off, np.int64)
    out["tok_len"] = np.array(tok_len, np.int32)
    out["valid"] = np.concatenate(valid_all)
    return out


def main():
    sp_vi = spm.SentencePieceProcessor(); sp_vi.load(str(BPE_VI))
    meta_tr = load_meta(SRC_TRAIN)
    meta_te = load_meta(SRC_TEST)

    # --- 1. dump corpus (mỗi clip 1 dòng) ---
    corpus = HERE / "corpus_en.txt"
    if not corpus.exists():
        print("1) Dump corpus tiếng Anh từ meta...", flush=True)
        with open(corpus, "w") as f:
            for i in range(len(meta_tr["mel_len"])):
                f.write(" ".join(clip_words(meta_tr, sp_vi, i)) + "\n")
                if (i + 1) % 20000 == 0:
                    print(f"  {i+1} clips", flush=True)
    else:
        print("1) corpus_en.txt đã có, bỏ qua")

    # --- 2. train BPE tiếng Anh (cùng cấu hình vocab với bản Việt) ---
    model_file = Path(str(BPE_EN_PREFIX) + ".model")
    if not model_file.exists():
        print("2) Train SentencePiece EN vocab 3500...", flush=True)
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=str(BPE_EN_PREFIX),
            vocab_size=VOCAB, model_type="bpe", character_coverage=1.0,
            bos_id=1, eos_id=2, unk_id=0, pad_id=-1,
            bos_piece="<s>", eos_piece="</s>")
    else:
        print("2) bpe_en.model đã có, bỏ qua")
    sp_en = spm.SentencePieceProcessor(); sp_en.load(str(model_file))
    print(f"   vocab EN: {sp_en.get_piece_size()}")

    # --- 3. retokenize train + test ---
    for src, dst, meta, name in [(SRC_TRAIN, DST_TRAIN, meta_tr, "train"),
                                 (SRC_TEST, DST_TEST, meta_te, "test")]:
        dst.mkdir(exist_ok=True)
        mel_link = dst / "mel.f16"
        if not mel_link.exists():
            mel_link.symlink_to(src / "mel.f16")   # mel GIỮ NGUYÊN, không copy
        print(f"3) Retokenize {name}...", flush=True)
        out = retokenize(meta, sp_vi, sp_en, name)
        np.savez(dst / "meta.npz", **out)
        # thống kê độ dài token: BPE đúng ngôn ngữ phải cho ít token hơn
        old_t = meta["tok_len"].mean(); new_t = out["tok_len"].mean()
        print(f"   {name}: {old_t:.1f} -> {new_t:.1f} token/clip "
              f"({(1-new_t/old_t)*100:+.1f}% ngắn hơn)")
    print("DONE_BPE_V2")


if __name__ == "__main__":
    main()
