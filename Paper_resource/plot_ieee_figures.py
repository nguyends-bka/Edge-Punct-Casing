"""IEEE-style figures for the ViACaPu paper (Dolly, Vietnamese) — color version.

Refined palette (color but restrained & print-safe), serif font, single-column
width ~3.4in, consistent spacing/alignment. Vector PDF + high-DPI PNG.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

SCRATCH = "/tmp/claude-1000/-home-ai-nguyends-Edge-Punct-Casing/9f39f5e8-7573-4f9e-9665-bc36d864d7e1/scratchpad"
OUT = Path("/home/ai/nguyends/Edge-Punct-Casing/Paper_resource/figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "axes.axisbelow": True,
})

# --- refined color palette (restrained, print-safe, colour-blind aware) ---
C_BASE  = "#7f8fa6"   # muted slate  (baseline)
C_TEXT  = "#4c8bf5"   # clear blue   (text-only)
C_AC    = "#e8743b"   # warm orange  (acoustic = the win)
INK = "#1a1a1a"; GRID = "#dddddd"; MUTED = "#666666"
COL, DBL = 3.4, 7.0
# sequential blue for heatmap (light -> deep, clean)
BLUES = LinearSegmentedColormap.from_list(
    "seq", ["#f7fbff", "#c9e0f5", "#89bced", "#4c8bf5", "#2a5fb0", "#16336b"])

CONF = json.load(open(f"{SCRATCH}/confusion.json"))
PUNCT_LB = ["NONE", "COMMA", "PERIOD", "QUES"]
CASE_LB = ["LOWER", "UPPER", "CAP", "MIX"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved", name)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=3)


# ---------------------------------------------------------------- fig1: compare
def fig_compare():
    models = [("Baseline [1]", 7.26, 0.936, 0.569, C_BASE),
              ("ViACaPu\n(text-only)", 2.68, 0.938, 0.820, C_TEXT),
              ("ViACaPu\n(acoustic+aux)", 3.58, 0.944, 0.927, C_AC)]
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.7))
    x = np.arange(len(models))
    for ax, idx, ylab, lo in [(axes[0], 3, "Punctuation F1", 0.60),
                              (axes[1], 2, "Casing F1", 0.80)]:
        vals = [m[idx] for m in models]
        cols = [m[4] for m in models]
        bars = ax.bar(x, vals, width=0.60, color=cols, edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.006, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x, [m[0] for m in models])
        ax.set_ylim(lo, 1.0); ax.set_ylabel(ylab, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        style(ax)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig1_compare_all_models")


# ---------------------------------------------------------------- fig2: per-class
def f1pc(cm):
    cm = np.array(cm, float); out = []
    for k in range(4):
        tp = cm[k, k]; fp = cm[:, k].sum()-tp; fn = cm[k, :].sum()-tp
        p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
        out.append(2*p*r/(p+r) if p+r else 0)
    return out


def fig_perclass():
    classes = ["COMMA", "PERIOD", "QUESTION"]
    text = f1pc(CONF["base_text"]["punct"])[1:]
    ac = f1pc(CONF["base_ac"]["punct"])[1:]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    x = np.arange(3); w = 0.38
    b1 = ax.bar(x-w/2, text, w, color=C_TEXT, edgecolor="white", linewidth=0.7, label="Text-only")
    b2 = ax.bar(x+w/2, ac, w, color=C_AC, edgecolor="white", linewidth=0.7, label="Acoustic")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 1.02); ax.set_ylabel("F1", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper right", handlelength=1.3)
    style(ax)
    fig.tight_layout()
    save(fig, "fig2_punct_f1_perclass")


# ---------------------------------------------------------------- fig3/4: confusion
def fig_confusion(task, labels, name):
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 3.1))
    for ax, key, sub in [(axes[0], "base_text", "(a) Text-only"),
                         (axes[1], "base_ac", "(b) Acoustic")]:
        cm = np.array(CONF[key][task], float)
        share = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(share, cmap=BLUES, vmin=0, vmax=1, aspect="equal")
        for i in range(4):
            for j in range(4):
                c = "white" if share[i, j] > 0.5 else INK
                ax.text(j, i, f"{share[i,j]*100:.1f}", ha="center", va="center",
                        color=c, fontsize=7.5, fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(4), labels, fontsize=7)
        ax.set_yticks(range(4), labels, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8, color=INK)
        if key == "base_text":
            ax.set_ylabel("True", fontsize=8, color=INK)
        ax.set_title(sub, fontsize=8.5, pad=5, color=INK)
        ax.tick_params(length=0, colors=MUTED)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(w_pad=1.8)
    save(fig, name)


# ---------------------------------------------------------------- fig6: domain shift
def fig_domain():
    groups = [("Case", 0.584, 0.856), ("Punct", 0.393, 0.703),
              ("COMMA", 0.291, 0.577), ("PERIOD", 0.489, 0.813),
              ("QUES", 0.016, 0.658)]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    x = np.arange(len(groups)); w = 0.38
    old = [g[1] for g in groups]; new = [g[2] for g in groups]
    ax.bar(x-w/2, old, w, color=C_BASE, edgecolor="white", linewidth=0.7, label="Released ckpt")
    ax.bar(x+w/2, new, w, color=C_AC, edgecolor="white", linewidth=0.7, label="Retrained")
    ax.set_xticks(x, [g[0] for g in groups], fontsize=7.5)
    ax.set_ylim(0, 1.0); ax.set_ylabel("F1", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper left", handlelength=1.3)
    style(ax)
    fig.tight_layout()
    save(fig, "fig6_base_domain_shift")


fig_compare()
fig_perclass()
fig_confusion("punct", PUNCT_LB, "fig3_confusion_punct_acoustic")
fig_confusion("case", CASE_LB, "fig4_confusion_case_acoustic")
fig_domain()
print("ALL IEEE COLOR FIGURES DONE")
