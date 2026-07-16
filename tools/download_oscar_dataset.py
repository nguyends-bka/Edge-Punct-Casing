from datasets import load_dataset
import os

TARGET_SIZE_GB = 1
TARGET_SIZE_BYTES = TARGET_SIZE_GB * 1024 * 1024 * 1024

output_file = "bpe_model_15000/spm_corpus_2.txt"

dataset = load_dataset(
    "oscar-corpus/OSCAR-2201",
    language="vi",
    split="train",
    streaming=True
)

written = 0
count = 0

with open(output_file, "w", encoding="utf-8") as f:
    for sample in dataset:
        text = sample["text"].strip()

        if not text:
            continue

        line = text.replace("\n", " ") + "\n"

        encoded = line.encode("utf-8")

        f.write(line)

        written += len(encoded)
        count += 1

        if count % 10000 == 0:
            print(
                f"{written / (1024**3):.2f} GB written "
                f"({count:,} documents)"
            )

        if written >= TARGET_SIZE_BYTES:
            break

print(f"\nDone!")
print(f"Documents: {count:,}")
print(f"Size: {written / (1024**3):.2f} GB")
print(f"Saved to: {output_file}")