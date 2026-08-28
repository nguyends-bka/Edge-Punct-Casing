"""ViACaPu architecture diagram for the paper (Figure 1)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures" / "fig0_architecture.png"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
C_TEXT = "#2a78d6"    # text branch
C_AC = "#008300"      # acoustic branch
C_FUSE = "#eda100"    # fusion
C_HEAD = "#4a3aa7"    # heads
GREY = "#e1e0d9"

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(11, 8))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
fig.patch.set_facecolor(SURFACE)


def box(x, y, w, h, text, color, tcolor="#ffffff", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=color, ec="none", zorder=3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=tcolor, fontsize=fs, fontweight="bold", zorder=4)


def arrow(x1, y1, x2, y2, color=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, color=color, lw=1.6, zorder=2))


# --- TEXT branch (left column) ---
ax.text(20, 97, "NHÁNH TEXT", ha="center", fontsize=11, fontweight="bold", color=C_TEXT)
box(5, 88, 30, 6, "BPE token ids (vocab 3500)", "#898781", fs=9)
box(5, 78, 30, 6, "Embedding (d=256)", C_TEXT)
box(5, 68, 30, 6, "3 × Conv1d + LN (residual)", C_TEXT)
box(5, 58, 30, 6, "BiLSTM × 2", C_TEXT)
box(5, 48, 30, 6, "gather đầu-từ → [B,W,d]", "#898781", fs=8.5)
for y1, y2 in [(88, 84), (78, 74), (68, 64), (58, 54)]:
    arrow(20, y1, 20, y2)

# --- ACOUSTIC branch (right column) ---
ax.text(78, 97, "NHÁNH ACOUSTIC", ha="center", fontsize=11, fontweight="bold", color=C_AC)
box(63, 88, 32, 6, "log-mel 80 (1 frame/10ms)", "#898781", fs=9)
box(63, 78, 32, 6, "2 × Conv1d stride-2 (↓4×)", C_AC)
box(63, 68, 32, 6, "BiGRU → frame feats [B,Ta/4,d]", C_AC, fs=8.5)
for y1, y2 in [(88, 84), (78, 74)]:
    arrow(79, y1, 79, y2)

# --- FUSION (center) ---
box(30, 36, 40, 7, "Cross-Attention (query=từ, key/value=frame)\n+ Gated fusion", C_FUSE, tcolor=INK, fs=9)
arrow(20, 48, 35, 43)          # text -> fusion
arrow(79, 68, 65, 43)          # acoustic -> fusion

# --- word LSTM + heads ---
box(35, 26, 30, 6, "BiLSTM (word-level)", C_TEXT)
arrow(50, 36, 50, 32)
box(18, 14, 28, 7, "ghép từ TRƯỚC → case\n(LOWER/UPPER/CAP/MIX)", C_HEAD, fs=8.5)
box(54, 14, 28, 7, "ghép từ SAU → punct\n(NONE/COMMA/PERIOD/?)", C_HEAD, fs=8.5)
arrow(45, 26, 34, 21)
arrow(55, 26, 66, 21)

ax.text(50, 6, "ViACaPu — 3.58M tham số · alignment-free · gắn sau bất kỳ ASR nào",
        ha="center", fontsize=10, style="italic", color=INK2)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print("saved", OUT)
