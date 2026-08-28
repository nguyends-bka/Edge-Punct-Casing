"""v4 "breakthrough" architecture for punctuation + casing.

Adds, on top of the v2w recipe (embedding 256, packed BiLSTM, adjacent-token
concat heads):

  1. Conformer-lite encoder blocks (macaron FFN + LOCAL windowed multi-head
     self-attention + depthwise conv module) -> mid-range clause context,
     bounded window keeps it fast & streaming-friendly.
  2. Linear-chain CRF on the punctuation head -> structured decoding so
     comma/period are placed consistently (targets precision / "đặt chuẩn").
  3. Focal loss on the casing head -> gentler handling of class imbalance.

Toggles via params: use_conformer, use_crf, use_focal, conformer_layers,
attn_window. Keeps the exact subword->word gathering of Model_new.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# --------------------------------------------------------------------------- #
# Conformer-lite pieces
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, d_model, mult=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LocalMHSA(nn.Module):
    """Multi-head self-attention restricted to a +/- window band."""

    def __init__(self, d_model, n_head, window, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.h = n_head
        self.dk = d_model // n_head
        self.window = window
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_pad):  # key_pad: [B, L] True=valid
        B, L, D = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2)  # [B,h,L,dk]
        k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)  # [B,h,L,L]

        idx = torch.arange(L, device=x.device)
        band = (idx[None, :] - idx[:, None]).abs() <= self.window  # [L,L]
        allow = band[None, None] & key_pad[:, None, None, :]  # [B,1,1,L] broadcast over queries
        scores = scores.masked_fill(~allow, float("-inf"))
        attn = scores.softmax(dim=-1)
        # rows that are fully -inf (padded queries) -> softmax gives nan; zero them
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.drop(attn)
        out = attn @ v  # [B,h,L,dk]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


class ConvModule(nn.Module):
    """Conformer depthwise-separable conv module."""

    def __init__(self, d_model, kernel=7, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, d_model * 2, 1)  # -> GLU
        self.dw = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, groups=d_model)
        self.post_norm = nn.LayerNorm(d_model)  # per-position, mask-safe (no batch stats)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_pad):
        h = self.norm(x) * key_pad.unsqueeze(-1)  # zero padded positions
        h = h.transpose(1, 2)  # [B,D,L]
        h = F.glu(self.pw1(h), dim=1)
        h = self.dw(h)
        h = h.transpose(1, 2)  # [B,L,D]
        h = F.silu(self.post_norm(h))
        h = self.pw2(h.transpose(1, 2)).transpose(1, 2)
        return self.drop(h)


class ConformerLite(nn.Module):
    def __init__(self, d_model, n_head=4, window=8, ff_mult=2, kernel=7, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, ff_mult, dropout)
        self.attn = LocalMHSA(d_model, n_head, window, dropout)
        self.conv = ConvModule(d_model, kernel, dropout)
        self.ff2 = FeedForward(d_model, ff_mult, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_pad):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, key_pad)
        x = x + self.conv(x, key_pad)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


# --------------------------------------------------------------------------- #
# Linear-chain CRF
# --------------------------------------------------------------------------- #
class CRF(nn.Module):
    """Standard linear-chain CRF. Masks must be left-aligned (mask[:,0] True)."""

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        self.trans = nn.Parameter(torch.zeros(num_tags, num_tags))  # trans[i,j]: i->j
        nn.init.uniform_(self.start, -0.1, 0.1)
        nn.init.uniform_(self.end, -0.1, 0.1)
        nn.init.uniform_(self.trans, -0.1, 0.1)

    def _numerator(self, emis, tags, mask):
        B, T, K = emis.shape
        score = self.start[tags[:, 0]] + emis[torch.arange(B), 0, tags[:, 0]]
        for t in range(1, T):
            m = mask[:, t]
            trans_t = self.trans[tags[:, t - 1], tags[:, t]]
            emit_t = emis[torch.arange(B), t, tags[:, t]]
            score = score + (trans_t + emit_t) * m
        last = mask.sum(1).long() - 1  # index of last valid step
        last_tags = tags[torch.arange(B), last]
        score = score + self.end[last_tags]
        return score

    def _denominator(self, emis, mask):
        B, T, K = emis.shape
        alpha = self.start.unsqueeze(0) + emis[:, 0]  # [B,K]
        for t in range(1, T):
            broadcast = alpha.unsqueeze(2) + self.trans.unsqueeze(0) + emis[:, t].unsqueeze(1)
            new_alpha = torch.logsumexp(broadcast, dim=1)  # [B,K]
            m = mask[:, t].unsqueeze(1)
            alpha = torch.where(m.bool(), new_alpha, alpha)
        alpha = alpha + self.end.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)  # [B]

    def neg_log_likelihood(self, emis, tags, mask):
        """Returns mean NLL per token (comparable scale to per-token CE)."""
        num = self._numerator(emis, tags, mask)
        den = self._denominator(emis, mask)
        nll = (den - num).sum()
        return nll / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def viterbi_decode(self, emis, mask):
        B, T, K = emis.shape
        score = self.start.unsqueeze(0) + emis[:, 0]  # [B,K]
        history = []
        for t in range(1, T):
            broadcast = score.unsqueeze(2) + self.trans.unsqueeze(0)  # [B,K,K]
            best, idx = broadcast.max(dim=1)  # over previous tag
            best = best + emis[:, t]
            m = mask[:, t].unsqueeze(1).bool()
            score = torch.where(m, best, score)
            history.append(idx)  # [B,K]
        score = score + self.end.unsqueeze(0)
        best_last = score.argmax(dim=1)  # [B]
        lengths = mask.sum(1).long()
        paths = torch.zeros(B, T, dtype=torch.long, device=emis.device)
        for b in range(B):
            L = int(lengths[b].item())
            tag = int(best_last[b].item())
            paths[b, L - 1] = tag
            for t in range(L - 2, -1, -1):
                tag = int(history[t][b, tag].item())
                paths[b, t] = tag
        return paths


# --------------------------------------------------------------------------- #
# Focal loss
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        # NB: not named `weight` so nn.Module.apply(init) helpers that probe
        # `m.weight` don't trip over a None attribute.
        self.cls_weight = weight

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        ce = F.nll_loss(logp, target, weight=self.cls_weight, reduction="none")
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ModelV4(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.config = params
        D = params.embedding_dim
        self.embedding = nn.Embedding(params.vocab_size, D)

        self.use_conformer = bool(getattr(params, "use_conformer", True))
        self.use_crf = bool(getattr(params, "use_crf", True))
        self.use_focal = bool(getattr(params, "use_focal", True))
        n_layers = int(getattr(params, "conformer_layers", 2))
        window = int(getattr(params, "attn_window", 8))
        drop = params.dropout

        if self.use_conformer:
            self.posenc = PositionalEncoding(D)
            self.blocks = nn.ModuleList(
                [ConformerLite(D, n_head=4, window=window, ff_mult=2, kernel=7, dropout=drop)
                 for _ in range(n_layers)]
            )

        self.biGRU = nn.LSTM(D, params.hidden_size1, 2, bidirectional=True, batch_first=True)
        self.GRU = nn.LSTM(params.hidden_size1 * 2, params.hidden_size2, 1, batch_first=True)

        self.decoder_case = nn.Linear(params.hidden_size2 * 2, params.out_size_case)
        self.decoder_punct = nn.Linear(params.hidden_size2 * 2, params.out_size_punct)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)

        if self.use_crf:
            self.crf = CRF(params.out_size_punct)

        cw = getattr(params, "case_weights", None)
        if cw:
            self.register_buffer("case_w", torch.tensor(cw, dtype=torch.float32), persistent=False)
        else:
            self.case_w = None
        self.focal = FocalLoss(gamma=2.0, weight=None) if self.use_focal else None

    def _encode(self, input_token_ids, valid_ids, label_lens):
        pad = input_token_ids != 0  # [B,L] True=valid token
        x = self.embedding(input_token_ids)
        if self.use_conformer:
            x = self.posenc(x)
            for blk in self.blocks:
                x = blk(x, pad)

        B, L, D = x.shape
        valid_output = torch.zeros_like(x)
        valid_mask = valid_ids.to(torch.bool)
        flat_valid_mask = valid_mask.view(-1)
        flat_x = x.view(-1, D)
        valid_embeddings = flat_x[flat_valid_mask]
        cum = valid_ids.cumsum(dim=1) - 1
        flat_cum = cum.view(-1)
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, L).reshape(-1)
        batch_indices = batch_indices[flat_valid_mask]
        valid_output[batch_indices, flat_cum[flat_valid_mask]] = valid_embeddings

        label_lens, indx = label_lens.sort(dim=0, descending=True)
        valid_output = valid_output[indx]

        packed = pack_padded_sequence(valid_output, lengths=label_lens.cpu(), batch_first=True)
        biGRU_out, _ = self.biGRU(packed)
        biGRU_out, label_lens = pad_packed_sequence(biGRU_out, batch_first=True)
        biGRU_out = pack_padded_sequence(biGRU_out, lengths=label_lens.cpu(), batch_first=True)
        GRU_out, _ = self.GRU(biGRU_out)
        GRU_out, _ = pad_packed_sequence(GRU_out, batch_first=True)  # [B,T,H]

        padded_case = F.pad(GRU_out, (0, 0, 1, 0), mode="replicate")
        concat_case = torch.cat((padded_case[:, :-1, :], GRU_out), dim=-1)
        padded_punct = F.pad(GRU_out, (0, 0, 0, 1), mode="replicate")
        concat_punct = torch.cat((GRU_out, padded_punct[:, 1:, :]), dim=-1)

        case_logits = self.decoder_case(self.dropout1(concat_case))   # [B,T,4]
        punct_logits = self.decoder_punct(self.dropout2(concat_punct))  # [B,T,4]
        return case_logits, punct_logits, indx

    def forward(self, input_token_ids, valid_ids=None, label_lens=None, label_masks=None, labels=None):
        case_logits, punct_logits, indx = self._encode(input_token_ids, valid_ids, label_lens)
        T = case_logits.shape[1]

        if labels is not None:
            labels = labels[indx]
            label_masks = label_masks[indx][:, :T]
            lm = label_masks.bool()
            labels = labels[:, :, :T]
            case_labels = labels[:, 0, :]
            punct_labels = labels[:, 1, :]

            active = lm.reshape(-1)
            case_logits_flat = case_logits.reshape(-1, self.config.out_size_case)[active]
            case_labels_flat = case_labels.reshape(-1)[active]
            if self.use_focal:
                case_loss = self.focal(case_logits_flat, case_labels_flat)
            else:
                case_loss = F.cross_entropy(case_logits_flat, case_labels_flat, weight=self.case_w)

            if self.use_crf:
                punct_loss = self.crf.neg_log_likelihood(punct_logits, punct_labels, lm.float())
            else:
                punct_logits_flat = punct_logits.reshape(-1, self.config.out_size_punct)[active]
                punct_labels_flat = punct_labels.reshape(-1)[active]
                punct_loss = F.cross_entropy(punct_logits_flat, punct_labels_flat)
            return case_loss, punct_loss

        # ---- inference ----
        valid_ids = valid_ids[indx]
        valid_ids_sorted, _ = valid_ids.sort(dim=1, descending=True)
        vs = valid_ids_sorted[:, :T]
        non_zero = vs != 0
        cum = vs.cumsum(dim=1)
        exclude_first = cum != 1
        cumf = vs.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        exclude_last = cumf != 1
        final_mask = non_zero & exclude_first & exclude_last

        if self.use_crf:
            punct_path = self.crf.viterbi_decode(punct_logits, non_zero.float())  # [B,T] indices
            return case_logits, punct_path, final_mask, True
        return case_logits, punct_logits, final_mask, False
