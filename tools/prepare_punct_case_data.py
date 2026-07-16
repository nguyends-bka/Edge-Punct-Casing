#!/usr/bin/env python3
"""Prepare Edge-Punct-Casing text and label files from punctuated text.

Input files should contain one sentence or utterance per line, with normal
casing and punctuation. The script writes the fixed filenames expected by
data_module.py.
"""

import argparse
import random
import unicodedata
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


PUNCT_LABELS = {
    ",": 1,
    ";": 1,
    ":": 1,
    ".": 2,
    "!": 2,
    "?": 3,
}

WORD_JOINERS = {"'", "-"}


def is_word_char(ch: str) -> bool:
    return ch.isalpha() or ch.isdigit()


def tokenize_line(line: str) -> List[str]:
    pieces: List[str] = []
    token: List[str] = []
    line = unicodedata.normalize("NFC", line.strip())

    for i, ch in enumerate(line):
        if ch in PUNCT_LABELS:
            if token:
                pieces.append("".join(token))
                token = []
            pieces.append(ch)
            continue

        if is_word_char(ch):
            token.append(ch)
            continue

        next_ch = line[i + 1] if i + 1 < len(line) else ""
        if ch in WORD_JOINERS and token and is_word_char(next_ch):
            token.append(ch)
            continue

        if token:
            pieces.append("".join(token))
            token = []

    if token:
        pieces.append("".join(token))

    return pieces


def case_label(word: str) -> int:
    letters = "".join(ch for ch in word if ch.isalpha())
    if not letters:
        return 0
    if letters.islower():
        return 0
    if letters.isupper():
        return 1
    if letters[0].isupper() and letters[1:].islower():
        return 2
    return 3


def clean_word(word: str, label: int) -> str:
    # The current decoder leaves MIX_CASE words unchanged, so keep their form.
    if label == 3:
        return word
    return word.lower()


def convert_line(line: str) -> Tuple[List[str], List[int], List[int]]:
    words: List[str] = []
    case_labels: List[int] = []
    punct_labels: List[int] = []

    for piece in tokenize_line(line):
        if piece in PUNCT_LABELS:
            if punct_labels:
                punct_labels[-1] = PUNCT_LABELS[piece]
            continue

        label = case_label(piece)
        words.append(clean_word(piece, label))
        case_labels.append(label)
        punct_labels.append(0)

    return words, case_labels, punct_labels


def read_non_empty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def split_lines(
    lines: Sequence[str],
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    if not 0 <= valid_ratio < 1:
        raise ValueError("--valid-ratio must be greater than or equal to 0 and less than 1")
    if not 0 <= test_ratio < 1:
        raise ValueError("--test-ratio must be greater than or equal to 0 and less than 1")
    if valid_ratio + test_ratio >= 1:
        raise ValueError("--valid-ratio plus --test-ratio must be less than 1")

    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(round(n_total * test_ratio))
    n_valid = int(round(n_total * valid_ratio))

    test = shuffled[:n_test]
    valid = shuffled[n_test : n_test + n_valid]
    train = shuffled[n_test + n_valid :]

    if not train:
        raise ValueError("The split produced no training lines. Use more data or smaller ratios.")
    if not valid:
        raise ValueError("The split produced no validation lines. Use more data or a larger valid ratio.")
    if not test:
        raise ValueError("The split produced no test lines. Use more data or a larger test ratio.")

    return train, valid, test


def write_split(
    lines: Iterable[str],
    text_path: Path,
    label_path: Path,
) -> None:
    kept = 0
    skipped = 0

    with text_path.open("w", encoding="utf-8", newline="\n") as text_f, label_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as label_f:
        for line in lines:
            words, case_labels, punct_labels = convert_line(line)
            if not words:
                skipped += 1
                continue

            if not (len(words) == len(case_labels) == len(punct_labels)):
                raise RuntimeError(f"Label count mismatch for line: {line}")

            text_f.write(" ".join(words) + "\n")
            label_f.write(" ".join(str(x) for x in case_labels) + "\n")
            label_f.write(" ".join(str(x) for x in punct_labels) + "\n")
            kept += 1

    print(f"Wrote {kept} examples to {text_path} and {label_path}")
    if skipped:
        print(f"Skipped {skipped} empty or unsupported lines for {text_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Edge-Punct-Casing data files from punctuated text."
    )

    parser.add_argument("--output-dir", required=True, help="Directory to write data files into.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when splitting one input file.")

    one_file = parser.add_argument_group("One-file split mode")
    one_file.add_argument("--input", help="One raw text file to split into train, valid, and test.")
    one_file.add_argument("--valid-ratio", type=float, default=0.1, help="Validation ratio for one-file mode.")
    one_file.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio for one-file mode.")

    three_files = parser.add_argument_group("Pre-split mode")
    three_files.add_argument("--train-input", help="Raw training text file.")
    three_files.add_argument("--valid-input", help="Raw validation text file.")
    three_files.add_argument("--test-input", help="Raw test text file.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        if args.train_input or args.valid_input or args.test_input:
            raise ValueError("Use either --input or the three pre-split inputs, not both.")
        train, valid, test = split_lines(
            read_non_empty_lines(Path(args.input)),
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        missing = [
            name
            for name, value in [
                ("--train-input", args.train_input),
                ("--valid-input", args.valid_input),
                ("--test-input", args.test_input),
            ]
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required pre-split inputs: {', '.join(missing)}")

        train = read_non_empty_lines(Path(args.train_input))
        valid = read_non_empty_lines(Path(args.valid_input))
        test = read_non_empty_lines(Path(args.test_input))

    write_split(train, output_dir / "train_text.txt", output_dir / "train_label.txt")
    write_split(valid, output_dir / "valid_text.txt", output_dir / "valid_label.txt")
    write_split(
        test,
        output_dir / "0_IWSLT2011_asr_test_text.txt",
        output_dir / "0_IWSLT2011_asr_test_label.txt",
    )


if __name__ == "__main__":
    main()
