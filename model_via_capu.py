"""ViA-CaPu: lightweight acoustic-aware punctuation + casing.

From-scratch, alignment-free. A tiny audio encoder (log-mel -> conv downsample ->
BiGRU) produces frame features; word-level text hidden states cross-attend over
those frames (soft alignment, no timestamps needed); a gated fusion mixes text
and acoustic context per word; two heads predict punctuation and casing.

`use_acoustic=False` disables the whole acoustic path -> pure text-only baseline
with the identical text branch & heads, for a clean ablation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
class AcousticEncoder(nn.Module):
    """log-mel [B,Ta,80] -> frame features [B,Ta//4,d]. ~0.6M params."""

    def __init__(self, n_mels=80, d=256):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, 128, 5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(128, d, 5, stride=2, padding=2)
        self.ln1, self.ln2 = nn.LayerNorm(128), nn.LayerNorm(d)
        self.gru = nn.GRU(d, d // 2, 1, bidirectional=True, batch_first=True)
        # aux head: predict per-frame energy (encourages encoding pauses/dynamics)
        self.energy_head = nn.Linear(d, 1)

    def forward(self, mel, mel_len):
        x = mel.transpose(1, 2)                     # [B,80,Ta]
        x = F.relu(self.ln1(self.conv1(x).transpose(1, 2)))   # [B,Ta/2,128]
        x = F.relu(self.ln2(self.conv2(x.transpose(1, 2)).transpose(1, 2)))  # [B,Ta/4,d]
        x, _ = self.gru(x)
        out_len = torch.div(mel_len - 1, 4, rounding_mode="floor") + 1
        return x, out_len


class ViACaPu(nn.Module):
    def __init__(self, vocab_size, d=256, n_case=4, n_punct=4,
                 use_acoustic=True, dropout=0.3):
        super().__init__()
        self.use_acoustic = use_acoustic
        self.d = d

        # --- text branch (from scratch) ---
        self.emb = nn.Embedding(vocab_size, d)
        self.tconv = nn.ModuleList([nn.Conv1d(d, d, 3, padding=1) for _ in range(3)])
        self.tnorm = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.tlstm = nn.LSTM(d, d // 2, 2, bidirectional=True, batch_first=True)

        # --- acoustic branch + fusion ---
        if use_acoustic:
            self.acoustic = AcousticEncoder(80, d)
            self.cross = nn.MultiheadAttention(d, 4, dropout=dropout, batch_first=True)
            self.gate = nn.Linear(2 * d, d)

        # --- word-level sequential + heads ---
        self.wlstm = nn.LSTM(d, d // 2, 1, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head_case = nn.Linear(d, n_case)
        self.head_punct = nn.Linear(d, n_punct)

    def _text_words(self, tokens, tok_mask, word_pos, word_mask):
        x = self.emb(tokens)                                   # [B,L,d]
        h = x.transpose(1, 2)
        for c, n in zip(self.tconv, self.tnorm):
            h = n((h + F.relu(c(h))).transpose(1, 2)).transpose(1, 2)  # residual conv
        h = h.transpose(1, 2)                                  # [B,L,d]
        h, _ = self.tlstm(h)                                   # [B,L,d]
        # gather word-start positions -> [B,W,d]
        B, W = word_pos.shape
        idx = word_pos.unsqueeze(-1).expand(-1, -1, self.d)
        return h.gather(1, idx)                                # [B,W,d]

    def forward(self, batch):
        tokens = batch["tokens"]; tok_mask = batch["tok_mask"]
        word_pos = batch["word_pos"]; word_mask = batch["word_mask"]
        text_w = self._text_words(tokens, tok_mask, word_pos, word_mask)  # [B,W,d]

        aux_energy = None
        if self.use_acoustic:
            mel = batch["mel"]; mel_len = batch["mel_len"]
            af, af_len = self.acoustic(mel, mel_len)           # [B,Ta',d]
            Ta = af.shape[1]
            key_pad = torch.arange(Ta, device=af.device)[None, :] >= af_len[:, None]
            ctx, _ = self.cross(text_w, af, af, key_padding_mask=key_pad,
                                need_weights=False)            # [B,W,d]
            z = torch.sigmoid(self.gate(torch.cat([text_w, ctx], dim=-1)))
            fused = z * text_w + (1 - z) * ctx
            aux_energy = self.acoustic.energy_head(af).squeeze(-1)  # [B,Ta']
        else:
            fused = text_w

        h, _ = self.wlstm(fused)
        h = self.drop(h)
        case_logits = self.head_case(h)                        # [B,W,n_case]
        punct_logits = self.head_punct(h)
        return case_logits, punct_logits, aux_energy
