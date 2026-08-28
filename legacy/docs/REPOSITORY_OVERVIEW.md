1# Edge-Punct-Casing Repository Overview

## Short Description

This repository implements a lightweight punctuation and word-casing restoration model for automatic speech recognition (ASR) text.

ASR systems often produce plain text such as:

```text
hello how are you i am fine
```

This model predicts two things for each word:

1. The word casing, for example lowercase, uppercase, capitalized, or mixed case.
2. The punctuation after the word, for example no punctuation, comma, period, or question mark.

The final output can become:

```text
Hello, how are you? I am fine.
```

The README says this is an implementation of the paper:

```text
A lightweight and efficient punctuation and word casing prediction model for on-device streaming ASR
```

## What Is Included

This repo contains Python source code only. It does not include:

- A dataset.
- A trained checkpoint.
- A SentencePiece/BPE model.
- A dependency file such as `requirements.txt`.
- A downloader for public data.
- An official dataset-specific preparation script for IWSLT or another public corpus.

You must provide the training data and the SentencePiece model yourself, or create them by following `TRAINING_FROM_SCRATCH.md`.

## Main Tasks Supported

The code supports these workflows:

- Train a PyTorch model from labeled text data: `train.py`
- Evaluate a PyTorch checkpoint on a labeled test set: `decode.py`
- Run a PyTorch checkpoint on plain text sentences: `decode_sentence.py`
- Export a PyTorch checkpoint to ONNX: `export-onnx.py`
- Evaluate an ONNX model on a labeled test set: `onnx_decode.py`
- Run an ONNX model on plain text sentences: `onnx_decode_sentence.py`

## Repository File Map

| File | Purpose |
| --- | --- |
| `README.md` | Very short project description. |
| `LICENSE` | Apache License 2.0. |
| `train.py` | Main training entry point. Builds the model, loads data, trains, validates, writes checkpoints and TensorBoard logs. |
| `data_module.py` | Reads text and labels, tokenizes with SentencePiece, creates cached feature files, and returns PyTorch DataLoaders. |
| `model.py` | Defines the active neural network class, `Model_new`. |
| `utils.py` | Logging, distributed training helpers, learning-rate scheduler helpers, convolution encoder blocks, attention blocks, and layer normalization. |
| `decode.py` | Evaluates a PyTorch checkpoint on the test files inside `data_dir`. Logs precision, recall, and F1. |
| `decode_sentence.py` | Applies a PyTorch checkpoint to a plain text file and prints restored casing and punctuation. |
| `export-onnx.py` | Exports a PyTorch checkpoint to ONNX, simplifies it, and creates an int8 quantized ONNX model. |
| `onnx_decode.py` | Evaluates an ONNX model on the test files inside `data_dir`. |
| `onnx_decode_sentence.py` | Applies an ONNX model to a plain text file and prints restored casing and punctuation. |
| `tools/prepare_punct_case_data.py` | Helper script that converts punctuated, cased raw text into the text and label files expected by `data_module.py`. |
| `__init__.py` | Package marker that imports the main modules. |

## Model Inputs

The model does not read raw audio. It reads text that has already been transcribed by ASR or prepared as ASR-like text.

The important input files are:

```text
data/
  train_text.txt
  train_label.txt
  valid_text.txt
  valid_label.txt
  0_IWSLT2011_asr_test_text.txt
  0_IWSLT2011_asr_test_label.txt

bpe_model/
  bpe.model
```

The `*_text.txt` files contain one sentence, utterance, or text segment per line.

The `*_label.txt` files contain two label lines for every text line:

1. One line of casing labels.
2. One line of punctuation labels.

For example, if `train_text.txt` has this line:

```text
hello how are you
```

Then `train_label.txt` needs two matching lines:

```text
2 0 0 0
1 0 0 3
```

That means:

- `hello` should become `Hello`, so case label `2`.
- punctuation after `hello` is comma, so punctuation label `1`.
- punctuation after `you` is question mark, so punctuation label `3`.

## Label Meanings

### Casing Labels

| ID | Name | Meaning |
| --- | --- | --- |
| `0` | `LOWER` | Keep the word lowercase. |
| `1` | `UPPER` | Make the whole word uppercase. |
| `2` | `CAP` | Capitalize the word, for example `hello` to `Hello`. |
| `3` | `MIX_CASE` | Mixed-case word. The current sentence decoder leaves the input word unchanged for this label. |

### Punctuation Labels

| ID | Name | Meaning |
| --- | --- | --- |
| `0` | `NO_PUNCT` | No punctuation after the word. |
| `1` | `COMMA` | Add `,` after the word. |
| `2` | `PERIOD` | Add `.` after the word. |
| `3` | `QUESTION` | Add `?` after the word. |

Only comma, period, and question mark are supported by the current decoder.

## Data Flow

The training pipeline works like this:

1. Read text and label files from `data_dir`.
2. Use `bpe_model/bpe.model` to split each word into SentencePiece token IDs.
3. Add beginning-of-sentence and end-of-sentence tokens.
4. Create `valid_ids` so the model knows which subword token is the first token of a real word.
5. Pad every sample to `max_seq_length`, which defaults to `200`.
6. Cache processed features to:

```text
data/train_features.txt
data/valid_features.txt
data/test_features.txt
```

7. Train the model with PyTorch.
8. Save checkpoints in `exp_dir`.

The cached feature files make later runs start faster. If you change the text files, label files, SentencePiece model, or `max_seq_length`, delete the cached feature files before training again.

## Model Architecture

The active model is `Model_new` in `model.py`.

At a high level, it uses:

- `nn.Embedding` to convert SentencePiece token IDs into vectors.
- A 3-layer convolution encoder from `utils.Encoder`.
- A 2-layer bidirectional LSTM.
- A 1-layer LSTM.
- Two output heads:
  - one classifier for word casing,
  - one classifier for punctuation.

The casing head looks at the current token plus previous context. The punctuation head looks at the current token plus next context. This is why the model is useful for restoring readable text after ASR.

Default model parameters from `train.py`:

| Parameter | Default |
| --- | --- |
| Vocabulary size | Read from the SentencePiece model |
| Embedding dimension | `100` |
| Max sequence length | `200` |
| First hidden size | `384` |
| Second hidden size | `384` |
| Casing classes | `4` |
| Punctuation classes | `4` |
| Dropout | `0.5` |

## Training Outputs

Training writes files into `exp_dir`, for example:

```text
exp/
  log-train-YYYY-MM-DD-HH-MM-SS
  tensorboard/
  epoch-0.pt
  epoch-1.pt
  checkpoint-200.pt
  best-valid-loss.pt
```

Important output types:

- `epoch-N.pt`: checkpoint saved at the end of epoch `N`.
- `checkpoint-N.pt`: checkpoint saved every `save_every_n` training batches.
- `best-valid-loss.pt`: checkpoint saved when validation loss improves.
- `log-train-*`: plain text training logs.
- `tensorboard/`: TensorBoard event files.

## Evaluation Outputs

`decode.py` logs metrics for case and punctuation labels:

- Precision
- Recall
- F1
- Overall precision, recall, and F1 for non-zero labels

`decode_sentence.py` prints restored sentences to the terminal.

## ONNX Outputs

`export-onnx.py` writes:

```text
exp/model.onnx
exp/model_sim.onnx
exp/model.int8.onnx
```

`model.onnx` is the direct export.

`model_sim.onnx` is the simplified model.

`model.int8.onnx` is the dynamically quantized int8 model intended for smaller and faster CPU inference.

The ONNX model metadata includes the label mapping:

```text
NO_PUNCT=0
COMMA=1
PERIOD=2
QUESTION=3
LOWER=0
UPPER=1
CAP=2
MIX_CASE=3
```

## Important Implementation Notes

- The repo expects a SentencePiece model path via `--bpe_model`.
- The repo pads token IDs with `0`, so it is best to train SentencePiece with a real pad token at ID `0`.
- `decode.py` and `decode_sentence.py` choose checkpoints by `--epoch` or `--batch`.
- In the current code, `--epoch 1` loads `epoch-0.pt`, `--epoch 2` loads `epoch-1.pt`, and so on.
- The parsed `--start-epoch` and `--start-batch` training options are not fully implemented as resume logic.
- The current sentence decoder does not reconstruct the exact original spelling for `MIX_CASE`; it leaves the input word unchanged.
- Multi-GPU training uses PyTorch distributed training with the `nccl` backend, so it is intended for NVIDIA GPU environments.

## Minimal Command Examples

Train:

```bash
python train.py --world-size 1 --data_dir data --exp_dir exp --bpe_model bpe_model/bpe.model --epochs 10
```

Evaluate a PyTorch batch checkpoint:

```bash
python decode.py --data_dir data --exp_dir exp --bpe_model bpe_model/bpe.model --batch 200
```

Restore punctuation and casing for a text file:

```bash
python decode_sentence.py --text_file examples/input.txt --exp_dir exp --bpe_model bpe_model/bpe.model --batch 200
```

Export to ONNX:

```bash
python export-onnx.py --exp_dir exp --batch 200
```

Run the ONNX model on a text file:

```bash
python onnx_decode_sentence.py --text_file examples/input.txt --model_filename exp/model.int8.onnx --bpe_model bpe_model/bpe.model
```

## License

This repository uses the Apache License 2.0, based on the included `LICENSE` file.
