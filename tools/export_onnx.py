"""Export ViACaPu checkpoint sang ONNX.

Model.forward() nhận một dict batch; ONNX cần input là tensor rời rạc, nên ta bọc
model trong một wrapper nhận đúng các tensor cần thiết theo thứ tự cố định.

Cách dùng:
    python3 -m tools.export_onnx --ckpt exp_seed/best_ac_s43_aux5.pt --out ac_aux.onnx
    python3 -m tools.export_onnx --ckpt exp_seed/best_text_s42.pt --out text.onnx --no-acoustic
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import ViACaPu


class ONNXWrapper(nn.Module):
    """Bọc ViACaPu để nhận/trả tensor rời rạc thay vì dict — yêu cầu của ONNX export."""

    def __init__(self, model: ViACaPu, use_acoustic: bool):
        super().__init__()
        self.model = model
        self.use_acoustic = use_acoustic

    def forward(self, tokens, tok_mask, word_pos, word_mask, mel=None, mel_len=None):
        batch = dict(tokens=tokens, tok_mask=tok_mask, word_pos=word_pos, word_mask=word_mask)
        if self.use_acoustic:
            batch["mel"] = mel
            batch["mel_len"] = mel_len
        case_logits, punct_logits, _ = self.model(batch)   # bỏ aux_energy: chỉ dùng lúc train
        return case_logits, punct_logits


def build_dummy_inputs(use_acoustic: bool, vocab_size: int, B=1, L=12, W=8, T=200):
    tokens = torch.randint(4, vocab_size, (B, L), dtype=torch.long)
    tok_mask = torch.ones(B, L, dtype=torch.bool)
    word_pos = torch.arange(W, dtype=torch.long).unsqueeze(0).expand(B, W).clone()
    word_mask = torch.ones(B, W, dtype=torch.bool)
    args = [tokens, tok_mask, word_pos, word_mask]
    names = ["tokens", "tok_mask", "word_pos", "word_mask"]
    dynamic_axes = {
        "tokens": {0: "batch", 1: "seq_len"},
        "tok_mask": {0: "batch", 1: "seq_len"},
        "word_pos": {0: "batch", 1: "n_words"},
        "word_mask": {0: "batch", 1: "n_words"},
    }
    if use_acoustic:
        mel = torch.randn(B, T, 80, dtype=torch.float32)
        mel_len = torch.full((B,), T, dtype=torch.long)
        args += [mel, mel_len]
        names += ["mel", "mel_len"]
        dynamic_axes["mel"] = {0: "batch", 1: "n_frames"}
        dynamic_axes["mel_len"] = {0: "batch"}
    return tuple(args), names, dynamic_axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt (vd exp_seed/best_ac_s43_aux5.pt)")
    ap.add_argument("--out", default=None, help="đường dẫn .onnx đầu ra (mặc định: cùng tên ckpt)")
    ap.add_argument("--bpe_model", default=str(ROOT / "bpe_model/bpe.model"))
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--no-acoustic", action="store_true", help="export model text-only")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(); sp.load(args.bpe_model)
    vocab_size = sp.get_piece_size()

    use_acoustic = not args.no_acoustic
    model = ViACaPu(vocab_size, d=args.d, use_acoustic=use_acoustic)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[cảnh báo] missing={missing} unexpected={unexpected}")
    model.eval()

    wrapper = ONNXWrapper(model, use_acoustic)
    dummy, input_names, dynamic_axes = build_dummy_inputs(use_acoustic, vocab_size)
    out_names = ["case_logits", "punct_logits"]
    dynamic_axes["case_logits"] = {0: "batch", 1: "n_words"}
    dynamic_axes["punct_logits"] = {0: "batch", 1: "n_words"}

    out_path = Path(args.out) if args.out else Path(args.ckpt).with_suffix(".onnx")

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(out_path),
            input_names=input_names, output_names=out_names,
            dynamic_axes=dynamic_axes, opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,  # exporter dựa trên TorchScript: ổn định hơn với nn.GRU/LSTM
        )

    npar = sum(p.numel() for p in model.parameters())
    size_mb = out_path.stat().st_size / 1e6
    print(f"Đã export: {out_path}  ({size_mb:.1f} MB, model {npar/1e6:.2f}M tham số, "
          f"{'acoustic+text' if use_acoustic else 'text-only'})")


if __name__ == "__main__":
    main()
