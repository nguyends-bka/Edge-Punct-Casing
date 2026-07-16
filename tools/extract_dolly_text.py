#!/usr/bin/env python3
"""Extract ONLY the `text` column from the dolly-audio-1000h-vietnamese
parquet files (audio column is ~472MB/file and is skipped entirely via
columnar range reads). Writes one sentence per line to the output file.

Usage:
  .venv/bin/python tools/extract_dolly_text.py --out raw/all_dolly.txt --workers 16
"""
import argparse
import sys
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

BASE = "datasets/dolly-vn/dolly-audio-1000h-vietnamese@refs/convert/parquet/default/train"

_local = threading.local()


def get_fs():
    if not hasattr(_local, "fs"):
        _local.fs = HfFileSystem()
    return _local.fs


def read_texts(path):
    fs = get_fs()
    pf = pq.ParquetFile(fs.open(path))
    tbl = pf.read(columns=["text"])
    out = []
    for v in tbl.column("text").to_pylist():
        if not v:
            continue
        # collapse any internal newlines so one utterance == one line
        s = unicodedata.normalize("NFC", " ".join(v.split()))
        if s:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    fs = HfFileSystem()
    files = sorted(f["name"] for f in fs.ls(BASE, detail=True))
    print(f"Found {len(files)} parquet files", flush=True)

    results = {}
    done = 0
    total = len(files)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(read_texts, p): p for p in files}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                results[p] = fut.result()
            except Exception as e:
                results[p] = []
                print(f"ERROR {p}: {e}", file=sys.stderr, flush=True)
            done += 1
            if done % 10 == 0 or done == total:
                got = sum(len(v) for v in results.values())
                print(f"  {done}/{total} files, {got} lines so far", flush=True)

    n = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for p in files:  # keep deterministic order
            for line in results.get(p, []):
                f.write(line + "\n")
                n += 1
    print(f"Wrote {n} lines to {args.out}", flush=True)


if __name__ == "__main__":
    main()
