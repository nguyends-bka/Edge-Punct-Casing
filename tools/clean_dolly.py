#!/usr/bin/env python3
"""Clean raw/all_dolly.txt for the punct+casing task.

Filters applied (order matters):
  1. NFC-normalize, collapse internal whitespace.
  2. Drop lines containing non Vietnamese-Latin letters (CJK, Cyrillic, ...).
  3. Drop degenerate lines: fewer than MIN_WORDS tokens, or fewer than
     MIN_ALPHA_WORDS tokens that contain a letter (kills things like "Cột:.").
  4. Exact de-duplicate (keep first occurrence, preserves order).

Prints a summary of how many lines each stage removed.
"""
import argparse
import re
import unicodedata

# Any letter that is NOT basic Latin or Latin-1/Latin-Extended (covers Vietnamese
# diacritics) flags a foreign script we want to drop.
FOREIGN = re.compile(
    r"[Ѐ-ӿ一-鿿぀-ヿ가-힯؀-ۿ฀-๿]"
)
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

MIN_WORDS = 3
MIN_ALPHA_WORDS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-words", type=int, default=MIN_WORDS)
    ap.add_argument("--min-alpha-words", type=int, default=MIN_ALPHA_WORDS)
    args = ap.parse_args()

    n_total = n_foreign = n_short = n_dup = n_kept = 0
    seen = set()

    with open(args.input, encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            n_total += 1
            s = unicodedata.normalize("NFC", " ".join(line.split()))
            if not s:
                n_short += 1
                continue
            if FOREIGN.search(s):
                n_foreign += 1
                continue
            words = s.split()
            alpha_words = sum(1 for w in words if LETTER.search(w))
            if len(words) < args.min_words or alpha_words < args.min_alpha_words:
                n_short += 1
                continue
            if s in seen:
                n_dup += 1
                continue
            seen.add(s)
            fout.write(s + "\n")
            n_kept += 1

    print(f"total       : {n_total}")
    print(f"dropped foreign : {n_foreign}")
    print(f"dropped short   : {n_short}")
    print(f"dropped dup     : {n_dup}")
    print(f"kept        : {n_kept}  ({100*n_kept/n_total:.1f}%)")


if __name__ == "__main__":
    main()
