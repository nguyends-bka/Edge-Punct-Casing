"""Export ViACaPu checkpoint sang ONNX.

Model.forward() nhận một dict batch; ONNX cần input là tensor rời rạc, nên ta bọc
model trong một wrapper nhận đúng các tensor cần thiết theo thứ tự cố định.

Cách dùng:
    python3 -m tools.export_onnx --ckpt exp_seed/best_ac_s43_aux5.pt --out ac_aux.onnx
    python3 -m tools.export_onnx --ckpt exp_seed/best_text_s42.pt --out text.onnx --no-acoustic
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        m = self.model
        text_w = m._text_words(tokens, tok_mask, word_pos, word_mask)   # [B,W,d]

        if self.use_acoustic:
            af, af_len = m.acoustic(mel, mel_len)                       # [B,Ta',d]
            key_pad = torch.arange(af.shape[1], device=af.device)[None, :] >= af_len[:, None]
            fused = text_w
            for attn, gate in zip([m.cross, *m.cross_extra], [m.gate, *m.gate_extra]):
                ctx = _mha(attn, fused, af, key_pad)
                z = torch.sigmoid(gate(torch.cat([fused, ctx], dim=-1)))
                fused = z * fused + (1 - z) * ctx
        else:
            fused = text_w

        h, _ = m.wlstm(fused)
        h = m.drop(h)
        if m.adjacent_heads:
            pad_l = F.pad(h, (0, 0, 1, 0), mode="replicate")
            h_case = torch.cat([pad_l[:, :-1], h], dim=-1)
            pad_r = F.pad(h, (0, 0, 0, 1), mode="replicate")
            h_punct = torch.cat([h, pad_r[:, 1:]], dim=-1)
        else:
            h_case = h_punct = h
        return m.head_case(h_case), m.head_punct(h_punct)


def _mha(attn: nn.MultiheadAttention, q, kv, key_pad):
    """Multi-head attention viết tay, toán học giống hệt nn.MultiheadAttention
    (batch_first=True, dropout tắt ở eval), nhưng KHÔNG cứng hoá độ dài khoá khi
    export ONNX.

    `nn.MultiheadAttention` kèm `key_padding_mask` bị TorchScript cố định số khung
    âm thanh theo giá trị dummy lúc export, khiến mô hình chỉ chạy đúng với đoạn
    dài bằng đúng dummy. Bản viết tay này giữ trục thời gian động.
    """
    B, Wq, d = q.shape
    H = attn.num_heads
    dh = d // H
    wq, wk, wv = attn.in_proj_weight.chunk(3, dim=0)
    bq, bk, bv = attn.in_proj_bias.chunk(3, dim=0)

    Q = (q @ wq.T + bq).view(B, Wq, H, dh).transpose(1, 2)      # [B,H,Wq,dh]
    K = (kv @ wk.T + bk).view(B, -1, H, dh).transpose(1, 2)     # [B,H,Tk,dh]
    V = (kv @ wv.T + bv).view(B, -1, H, dh).transpose(1, 2)

    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dh)          # [B,H,Wq,Tk]
    scores = scores.masked_fill(key_pad[:, None, None, :], float("-inf"))
    ctx = (scores.softmax(-1) @ V)                              # [B,H,Wq,dh]
    ctx = ctx.transpose(1, 2).reshape(B, Wq, d)
    return attn.out_proj(ctx)


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
