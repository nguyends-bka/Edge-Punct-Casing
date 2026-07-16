# Training and Using Edge-Punct-Casing From Scratch

This guide explains how to go from a fresh copy of the project to a trained model, then use that model on new text.

It is written for a beginner. You can follow it step by step.

Windows commands in this guide use PowerShell. Linux commands use Bash.

## What This Model Needs

The model needs three main things:

1. The project code in this repository.
2. Text data with correct punctuation and casing.
3. A SentencePiece BPE tokenizer file named something like `bpe_model/bpe.model`.

The repository does not include a dataset or pretrained model. You must bring your own text resource.

## Final Folder Layout

When everything is ready, the project should look like this:

```text
ProjectII/
  train.py
  model.py
  data_module.py
  decode.py
  decode_sentence.py
  export-onnx.py
  onnx_decode.py
  onnx_decode_sentence.py
  tools/
    pull_wikipedia_text.py
    prepare_punct_case_data.py
  raw/
    all.txt
  data/
    train_text.txt
    train_label.txt
    valid_text.txt
    valid_label.txt
    0_IWSLT2011_asr_test_text.txt
    0_IWSLT2011_asr_test_label.txt
  bpe_model/
    bpe.model
    bpe.vocab
  exp/
    checkpoint-200.pt
    epoch-0.pt
    log-train-...
  examples/
    input.txt
```

## Step 1: Get The Project

If you already have this folder on your computer, skip this step.

If the project is in GitHub or another Git server, open PowerShell on Windows or a terminal on Linux and run:

```shell
git clone <REPO_URL> ProjectII
cd ProjectII
```

Replace `<REPO_URL>` with the real repository URL.

If you received the project as a ZIP file:

1. Extract the ZIP file.
2. Open PowerShell on Windows or a terminal on Linux.
3. Move into the extracted folder:

Windows:

```powershell
cd C:\Users\duyni\OneDrive\ProjectII
```

Linux:

```bash
cd ~/ProjectII
```

## Step 2: Install Python

Install Python 3.10 or 3.11.

On Linux, install Python and the virtual environment package with your distribution's package manager. For Ubuntu or Debian:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

Check that Python works:

Windows:

```powershell
python --version
```

Linux:

```bash
python3 --version
```

You should see something like:

```text
Python 3.10.13
```

If PowerShell says `python` is not recognized, reinstall Python and enable the option named "Add Python to PATH".

On Linux, use `python3` when creating the virtual environment if `python` is not available.

## Step 3: Create A Virtual Environment

A virtual environment keeps this project's packages separate from other Python projects.

Run these commands inside the project folder:

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

When activation works, your prompt usually starts with `(.venv)`.

## Step 4: Install Python Packages

Install the packages needed for training and normal PyTorch inference:

Windows or Linux:

```shell
pip install torch numpy sentencepiece tqdm tensorboard
```

Install the packages needed for ONNX export and ONNX inference:

Windows or Linux:

```shell
pip install onnx onnxruntime onnxsim onnxconverter-common
```

Check the imports:

Windows or Linux:

```shell
python -c "import torch, sentencepiece, numpy, tqdm; print('basic packages OK')"
python -c "import onnx, onnxruntime; print('onnx packages OK')"
```

If these commands print `OK`, the packages are installed.

For GPU training, install a PyTorch build that matches your NVIDIA driver and CUDA version. If you are not sure, start with CPU training using `--world-size 1`.

## Step 5: Create Working Folders

Run:

Windows:

```powershell
New-Item -ItemType Directory -Force raw,data,bpe_model,exp,examples
```

Linux:

```bash
mkdir -p raw data bpe_model exp examples
```

Meaning of each folder:

| Folder | Meaning |
| --- | --- |
| `raw/` | Your original text resource before conversion. |
| `data/` | Converted text and label files used by this repo. |
| `bpe_model/` | SentencePiece tokenizer files. |
| `exp/` | Training logs, checkpoints, TensorBoard files, and ONNX exports. |
| `examples/` | Small text files used for manual testing. |

## Step 6: Prepare A Raw Text Resource

You need text that already has correct casing and punctuation.

Good raw text:

```text
Hello, how are you?
I am fine.
This project restores punctuation and casing.
```

Bad raw text:

```text
hello how are you
i am fine
```

Bad raw text has no punctuation or casing, so the script cannot learn the correct answer.

Put your raw text into this file:

```text
raw/all.txt
```

Rules for `raw/all.txt`:

- Use one sentence or one utterance per line.
- Do not use empty lines.
- Use normal punctuation where it belongs.
- The current model supports comma, period, and question mark.
- Exclamation marks are converted to periods by the helper script.
- Semicolons and colons are converted to commas by the helper script.
- The more data you have, the better. A few lines are enough only for testing that the pipeline runs.

Optional: pull raw text from Wikipedia.

The repository includes a helper that can create `raw/all.txt` from Wikipedia article text. It pulls the start page, then follows outgoing article links breadth-first.

You must set at least one stopping condition:

- `--max-depth`: maximum link depth. Use `0` to pull only the start page.
- `--size`: stop after `raw/all.txt` first becomes larger than this many megabytes.
- `--max-pages`: stop after this many unique pages have been written.

By default, the helper uses Vietnamese Wikipedia, starts at `Việt Nam`, and overwrites `raw/all.txt`.

Windows:

```powershell
python tools\pull_wikipedia_text.py --size 50
```

Linux:

```bash
python tools/pull_wikipedia_text.py --size 50
```

Example with all stopping conditions:

Windows:

```powershell
python tools\pull_wikipedia_text.py --begin-at "Việt Nam" --lang vi --max-depth 2 --size 100 --max-pages 500
```

Linux:

```bash
python tools/pull_wikipedia_text.py --begin-at "Việt Nam" --lang vi --max-depth 2 --size 100 --max-pages 500
```

To append to an existing `raw/all.txt` instead of overwriting it:

Windows:

```powershell
python tools\pull_wikipedia_text.py --size 50 --overwritten False
```

Linux:

```bash
python tools/pull_wikipedia_text.py --size 50 --overwritten False
```

The helper logs progress in the console only. It caches page titles during one run, so one page is not written twice in the same crawl.

Example small test file:

```text
Hello, how are you?
I am fine.
What time is the meeting?
The meeting starts at nine.
This is a small training example.
```

## Step 7: Convert Raw Text Into Repo Data Files

The model does not train directly from `raw/all.txt`.

It expects these exact files:

```text
data/train_text.txt
data/train_label.txt
data/valid_text.txt
data/valid_label.txt
data/0_IWSLT2011_asr_test_text.txt
data/0_IWSLT2011_asr_test_label.txt
```

Use the included helper:

Windows:

```powershell
python tools\prepare_punct_case_data.py --input raw\all.txt --output-dir data --valid-ratio 0.1 --test-ratio 0.1 --seed 42
```

Linux:

```bash
python tools/prepare_punct_case_data.py --input raw/all.txt --output-dir data --valid-ratio 0.1 --test-ratio 0.1 --seed 42
```

This command:

- Reads `raw/all.txt`.
- Randomly splits it into training, validation, and test data.
- Writes the exact filenames required by `data_module.py`.

If you already have separate raw files, use this instead:

Windows:

```powershell
python tools\prepare_punct_case_data.py --train-input raw\train_raw.txt --valid-input raw\valid_raw.txt --test-input raw\test_raw.txt --output-dir data
```

Linux:

```bash
python tools/prepare_punct_case_data.py --train-input raw/train_raw.txt --valid-input raw/valid_raw.txt --test-input raw/test_raw.txt --output-dir data
```

## Step 8: Understand The Converted Files

Open the first few text lines:

Windows:

```powershell
Get-Content data\train_text.txt -TotalCount 3
```

Linux:

```bash
head -n 3 data/train_text.txt
```

Open the first few label lines:

Windows:

```powershell
Get-Content data\train_label.txt -TotalCount 6
```

Linux:

```bash
head -n 6 data/train_label.txt
```

For each one line in `train_text.txt`, there must be two lines in `train_label.txt`.

Example:

```text
train_text.txt:
hello how are you

train_label.txt:
2 0 0 0
1 0 0 3
```

The first label line is casing:

| ID | Meaning |
| --- | --- |
| `0` | lowercase |
| `1` | uppercase |
| `2` | capitalize |
| `3` | mixed case |

The second label line is punctuation:

| ID | Meaning |
| --- | --- |
| `0` | no punctuation |
| `1` | comma |
| `2` | period |
| `3` | question mark |

The number of words and the number of labels must match.

If the text line has 4 words, the casing label line must have 4 numbers and the punctuation label line must have 4 numbers.

## Step 9: Train The SentencePiece BPE Tokenizer

The model does not work with normal words directly. It first splits words into smaller pieces called SentencePiece tokens.

Create a tokenizer training file:

Windows:

```powershell
Get-Content data\train_text.txt,data\valid_text.txt | Set-Content bpe_model\spm_corpus.txt
```

Linux:

```bash
cat data/train_text.txt data/valid_text.txt > bpe_model/spm_corpus.txt
```

Train the BPE model:

Windows or Linux:

```shell
python -c "import sentencepiece as spm; spm.SentencePieceTrainer.train(input='bpe_model/spm_corpus.txt', model_prefix='bpe_model/bpe', vocab_size=9000, model_type='bpe', character_coverage=1.0, pad_id=0, unk_id=1, bos_id=2, eos_id=3, hard_vocab_limit=False)"
```

This creates:

```text
bpe_model/bpe.model
bpe_model/bpe.vocab
```

Important settings:

- `--pad_id=0` is important because the repo pads token IDs with `0`.
- `--vocab_size=5000` matches the default training parameters, but the code will read the actual vocabulary size from `bpe.model`.
- `--hard_vocab_limit=false` helps small datasets by allowing SentencePiece to create fewer than 5000 pieces if needed.

Check the tokenizer:

Windows or Linux:

```shell
python -c "import sentencepiece as spm; sp=spm.SentencePieceProcessor(model_file='bpe_model/bpe.model'); print('pad', sp.piece_to_id('<pad>'), 'unk', sp.piece_to_id('<unk>'), 'bos', sp.piece_to_id('<s>'), 'eos', sp.piece_to_id('</s>'), 'size', sp.get_piece_size())"
```

You want `pad 0`.

## Step 10: Start A Small Smoke Training Run

Before running a long training job, run a small job to check that the data, labels, tokenizer, and imports work.

Windows or Linux:

```shell
python train.py --world-size 1 --data_dir data --exp_dir exp_smoke --bpe_model bpe_model/bpe.model --epochs 0 --batch_size 2 --tensorboard False
```

What should happen:

- The script loads `bpe_model/bpe.model`.
- It extracts train and validation features.
- It creates cached feature files in `data/`.
- It starts training.
- It writes at least one checkpoint into `exp_smoke/`.

After this run, you should see:

```text
data/train_features.txt
data/valid_features.txt
exp_smoke/epoch-0.pt
```

If this step fails, fix it before starting a long training run.

## Step 11: Delete Smoke Outputs If Needed

If the smoke run was only a test, you can leave it alone. It uses `exp_smoke/`, not your real `exp/` folder.

If you changed the source data, labels, tokenizer, or `max_seq_length`, delete the cached feature files before training again:

Windows:

```powershell
Remove-Item data\train_features.txt,data\valid_features.txt,data\test_features.txt -ErrorAction SilentlyContinue
```

Linux:

```bash
rm -f data/train_features.txt data/valid_features.txt data/test_features.txt
```

The next training or evaluation command will rebuild them.

## Step 12: Train The Model

For a first real run on CPU or one GPU:

Windows or Linux:

```shell
python train.py --world-size 1 --data_dir data --exp_dir exp --bpe_model bpe_model/bpe.model --epochs 10 --batch_size 64
```

If your computer runs out of memory, lower the batch size:

Windows or Linux:

```shell
python train.py --world-size 1 --data_dir data --exp_dir exp --bpe_model bpe_model/bpe.model --epochs 10 --batch_size 16
```

If you have a strong NVIDIA GPU, you can try a larger batch:

Windows or Linux:

```shell
python train.py --world-size 1 --data_dir data --exp_dir exp --bpe_model bpe_model/bpe.model --epochs 30 --batch_size 256
```

Useful training options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--world-size` | `1` | Number of GPU processes. Use `1` for CPU or one GPU. |
| `--data_dir` | required | Folder containing converted data files. |
| `--exp_dir` | required | Folder where checkpoints and logs are written. |
| `--bpe_model` | required | Path to `bpe.model`. |
| `--max_seq_length` | `200` | Maximum number of SentencePiece tokens per sample. |
| `--batch_size` | `256` | Number of samples per training batch. Lower it if memory is low. |
| `--base-lr` | `0.002` | Initial learning rate. |
| `--epochs` | `10` | Training epoch setting. The current loop also creates `epoch-0.pt`. |
| `--weight_decay` | `2.5e-5` | Weight decay for Adam. |
| `--tensorboard` | `True` | Write TensorBoard logs. |

The script saves checkpoint files like:

```text
exp/epoch-0.pt
exp/epoch-1.pt
exp/checkpoint-200.pt
exp/checkpoint-400.pt
exp/best-valid-loss.pt
```

For beginners, use a `checkpoint-N.pt` file for evaluation and inference because the command line maps directly to `--batch N`.

## Step 13: Watch Training Progress

Training logs are printed in the terminal and also written to `exp/log-train-*`.

You should see lines similar to:

```text
Epoch 0, batch 0, lr: [2.00e-03], loss[...], case_loss[...], punct_loss[...]
start validation
Epoch 0, validation loss[...]
Saving checkpoint to exp/checkpoint-200.pt
```

To open TensorBoard:

Windows:

```powershell
tensorboard --logdir exp\tensorboard
```

Linux:

```bash
tensorboard --logdir exp/tensorboard
```

Then open the URL that TensorBoard prints, usually:

```text
http://localhost:6006
```

## Step 14: Choose A Checkpoint

Look inside `exp/`:

Windows:

```powershell
Get-ChildItem exp
```

Linux:

```bash
ls exp
```

If you see:

```text
checkpoint-200.pt
```

then the batch number is:

```text
200
```

Use that number with `--batch 200`.

If you want to use an epoch checkpoint, remember the current decode scripts use an offset:

| File you want | Decode argument |
| --- | --- |
| `epoch-0.pt` | `--epoch 1` |
| `epoch-1.pt` | `--epoch 2` |
| `epoch-10.pt` | `--epoch 11` |

The easiest path is to use `--batch`.

## Step 15: Evaluate On The Test Set With PyTorch

The test files must exist:

```text
data/0_IWSLT2011_asr_test_text.txt
data/0_IWSLT2011_asr_test_label.txt
```

Run evaluation:

Windows or Linux:

```shell
python decode.py --data_dir data --exp_dir exp --bpe_model bpe_model/bpe.model --batch 200 --batch_size 256
```

Replace `200` with the batch number from your real checkpoint.

The script logs metrics to:

```text
exp/log-decode-...
```

Metrics include:

- Precision
- Recall
- F1
- Overall F1 for non-zero labels

For punctuation, non-zero labels are comma, period, and question mark.

## Step 16: Use The PyTorch Model On New Text

Create a plain text file:

Windows:

```powershell
Set-Content examples\input.txt "hello how are you"
Add-Content examples\input.txt "this is a test"
```

Linux:

```bash
printf "hello how are you\nthis is a test\n" > examples/input.txt
```

Run sentence decoding:

Windows:

```powershell
python decode_sentence.py --text_file examples\input.txt --exp_dir exp --bpe_model bpe_model/bpe.model --batch 200
```

Linux:

```bash
python decode_sentence.py --text_file examples/input.txt --exp_dir exp --bpe_model bpe_model/bpe.model --batch 200
```

Replace `200` with your checkpoint batch number.

The script prints restored text in the terminal.

Input:

```text
hello how are you
this is a test
```

Possible output:

```text
Hello, how are you?
This is a test.
```

The exact result depends on your training data and checkpoint quality.

## Step 17: Export To ONNX

ONNX is useful when you want CPU inference without loading the full PyTorch training stack.

Export:

Windows or Linux:

```shell
python export-onnx.py --exp_dir exp --batch 200
```

Replace `200` with your checkpoint batch number.

This creates:

```text
exp/model.onnx
exp/model_sim.onnx
exp/model.int8.onnx
```

Use `model.int8.onnx` for a small CPU model.

## Step 18: Evaluate The ONNX Model

Run:

Windows:

```powershell
python onnx_decode.py --model_filename exp\model.int8.onnx --data_dir data --bpe_model bpe_model\bpe.model --batch_size 256
```

Linux:

```bash
python onnx_decode.py --model_filename exp/model.int8.onnx --data_dir data --bpe_model bpe_model/bpe.model --batch_size 256
```

The script logs ONNX evaluation results to:

```text
exp/log-onnx-decode-...
```

## Step 19: Use The ONNX Model On New Text

Run:

Windows:

```powershell
python onnx_decode_sentence.py --text_file examples\input.txt --model_filename exp\model.int8.onnx --bpe_model bpe_model\bpe.model
```

Linux:

```bash
python onnx_decode_sentence.py --text_file examples/input.txt --model_filename exp/model.int8.onnx --bpe_model bpe_model/bpe.model
```

This prints restored text using ONNX Runtime.

## Step 20: Fine-Tune From An Existing Checkpoint

Fine-tuning means starting from an existing trained checkpoint and continuing training on new data.

Use the same `bpe.model` unless you intentionally changed the model vocabulary. If you train a new tokenizer, the checkpoint embedding size may not match.

Example:

Windows:

```powershell
python train.py --world-size 1 --do_finetune True --finetune_ckpt exp\checkpoint-200.pt --data_dir data_new --exp_dir exp_finetune --bpe_model bpe_model\bpe.model --base-lr 0.0001 --epochs 5 --batch_size 64
```

Linux:

```bash
python train.py --world-size 1 --do_finetune True --finetune_ckpt exp/checkpoint-200.pt --data_dir data_new --exp_dir exp_finetune --bpe_model bpe_model/bpe.model --base-lr 0.0001 --epochs 5 --batch_size 64
```

Recommended fine-tuning settings:

- Use a smaller learning rate, for example `--base-lr 0.0001`.
- Use a separate output folder, for example `exp_finetune`.
- Keep the same tokenizer.
- Check the validation loss often.

## Common Problems And Fixes

### `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'sentencepiece'
```

Fix:

Windows or Linux:

```shell
pip install sentencepiece
```

Use the package name from the error message.

### `FileNotFoundError`

Example:

```text
No such file or directory: 'data/train_text.txt'
```

Fix:

- Check that the data folder exists.
- Check that the file names are exact.
- Run the data preparation step again.

Required names:

```text
train_text.txt
train_label.txt
valid_text.txt
valid_label.txt
0_IWSLT2011_asr_test_text.txt
0_IWSLT2011_asr_test_label.txt
```

### `AssertionError` While Reading Data

This usually means the text and labels do not match.

Check:

- Every text line has two label lines.
- The number of words equals the number of casing labels.
- The number of words equals the number of punctuation labels.
- Label values are only `0`, `1`, `2`, or `3`.

### Training Uses Old Data After You Changed Files

The repo caches tokenized features.

Delete the feature cache:

Windows:

```powershell
Remove-Item data\train_features.txt,data\valid_features.txt,data\test_features.txt -ErrorAction SilentlyContinue
```

Linux:

```bash
rm -f data/train_features.txt data/valid_features.txt data/test_features.txt
```

Then run training or evaluation again.

### CUDA Out Of Memory

Fix:

- Lower `--batch_size`.
- Close other GPU programs.
- Use `--world-size 1`.
- Use CPU for testing.

Example smaller batch:

```shell
python train.py --world-size 1 --data_dir data --exp_dir exp --bpe_model bpe_model/bpe.model --epochs 10 --batch_size 16
```

### Multi-GPU Error On Windows

The distributed setup uses the `nccl` backend, which is intended for NVIDIA GPU training environments.

If multi-GPU fails, use:

```shell
python train.py --world-size 1 --data_dir data --exp_dir exp --bpe_model bpe_model/bpe.model --epochs 10 --batch_size 64
```

### Poor Punctuation Quality

Possible reasons:

- Too little training data.
- Training data does not match your real ASR text.
- Labels are wrong.
- The checkpoint is too early in training.
- Batch size or learning rate is not suitable.

Improvements:

- Add more clean training text.
- Use text from the same domain as your ASR output.
- Train for more epochs.
- Check validation loss and test F1.

### Mixed-Case Words Are Not Restored Correctly

The current decoder handles labels like this:

- `LOWER`: keep the input word as written.
- `UPPER`: convert the word to uppercase.
- `CAP`: title-case the word.
- `MIX_CASE`: keep the input word as written.

That means a word like `iphone` will not automatically become `iPhone`.

For product names, acronyms, and special names, use one of these approaches:

- Keep known mixed-case words in the input text.
- Add a post-processing dictionary after decoding.
- Extend `decode_sentence.py` to map known words to their preferred spelling.

## Beginner Checklist

Use this checklist when training from scratch:

```text
[ ] I am inside the ProjectII folder.
[ ] Python works: python --version on Windows, or python3 --version on Linux
[ ] The virtual environment is active.
[ ] Packages are installed.
[ ] raw/all.txt exists and has punctuated, cased text.
[ ] data/train_text.txt exists.
[ ] data/train_label.txt exists.
[ ] data/valid_text.txt exists.
[ ] data/valid_label.txt exists.
[ ] data/0_IWSLT2011_asr_test_text.txt exists.
[ ] data/0_IWSLT2011_asr_test_label.txt exists.
[ ] bpe_model/bpe.model exists.
[ ] Smoke training created exp_smoke/epoch-0.pt.
[ ] Real training created checkpoint files in exp/.
[ ] decode.py ran on the test set.
[ ] decode_sentence.py restored text from examples/input.txt.
[ ] export-onnx.py created exp/model.int8.onnx.
[ ] onnx_decode_sentence.py ran successfully.
```

## Minimal End-To-End Command List

This is the short version after you understand the steps above:

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch numpy sentencepiece tqdm tensorboard onnx onnxruntime onnxsim onnxconverter-common
New-Item -ItemType Directory -Force raw,data,bpe_model,exp,examples
python tools\prepare_punct_case_data.py --input raw\all.txt --output-dir data --valid-ratio 0.1 --test-ratio 0.1 --seed 42
Get-Content data\train_text.txt,data\valid_text.txt | Set-Content bpe_model\spm_corpus.txt
python -m sentencepiece.spm_train --input=bpe_model/spm_corpus.txt --model_prefix=bpe_model/bpe --vocab_size=5000 --model_type=bpe --character_coverage=1.0 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --hard_vocab_limit=false
python train.py --world-size 1 --data_dir data --exp_dir exp --bpe_model bpe_model/bpe.model --epochs 10 --batch_size 64
python decode.py --data_dir data --exp_dir exp --bpe_model bpe_model/bpe.model --batch 200 --batch_size 256
python decode_sentence.py --text_file examples\input.txt --exp_dir exp --bpe_model bpe_model/bpe.model --batch 200
python export-onnx.py --exp_dir exp --batch 200
python onnx_decode_sentence.py --text_file examples\input.txt --model_filename exp\model.int8.onnx --bpe_model bpe_model\bpe.model
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch numpy sentencepiece tqdm tensorboard onnx onnxruntime onnxsim onnxconverter-common
mkdir -p raw data bpe_model exp examples
python tools/prepare_punct_case_data.py --input raw/all.txt --output-dir data --valid-ratio 0.1 --test-ratio 0.1 --seed 42
cat data/train_text.txt data/valid_text.txt > bpe_model/spm_corpus.txt
python -m sentencepiece.spm_train --input=bpe_model/spm_corpus.txt --model_prefix=bpe_model/bpe --vocab_size=5000 --model_type=bpe --character_coverage=1.0 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --hard_vocab_limit=false
python train.py --world-size 1 --data_dir data --exp_dir exp --bpe_model bpe_model/bpe.model --epochs 10 --batch_size 64
python decode.py --data_dir data --exp_dir exp --bpe_model bpe_model/bpe.model --batch 200 --batch_size 256
python decode_sentence.py --text_file examples/input.txt --exp_dir exp --bpe_model bpe_model/bpe.model --batch 200
python export-onnx.py --exp_dir exp --batch 200
python onnx_decode_sentence.py --text_file examples/input.txt --model_filename exp/model.int8.onnx --bpe_model bpe_model/bpe.model
```

Replace `200` with the checkpoint number that actually exists in your `exp/` folder.
