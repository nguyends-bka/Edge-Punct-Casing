"""Fig 5: capacity ablation — punctuation F1 vs parameter count.
All configs measured on the same Dolly audio validation split, 12-epoch recipe.
Shows quality does NOT scale with parameters. IEEE style, color, PDF+PNG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/home/ai/nguyends/Edge-Punct-Casing/Paper_resource/figures")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 9, "axes.linewidth": 0.7,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "axes.axisbelow": True,
})
C_TEXT = "#4c8bf5"   # text-only
C_AC   = "#e8743b"   # acoustic (the win)
INK, GRID, MUTED = "#1a1a1a", "#dddddd", "#666666"

# (label, params_M, punctF1, is_acoustic, marker)
pts = [
    ("Text, $d$=256",  2.68, 0.820, False),
    ("Text, $d$=384",  5.34, 0.820, False),
    ("Acoustic+aux, $d$=256", 3.58, 0.927, True),   # star — the proposed model
    ("Acoustic, $d$=384", 7.34, 0.821, True),
    ("Acoustic, $d$=384, 2-layer fusion", 8.90, 0.818, True),
]

fig, ax = plt.subplots(figsize=(6.4, 4.0))

# connect text-only and acoustic families with faint guide lines
tx = [(p[1], p[2]) for p in pts if not p[3]]
ac = [(p[1], p[2]) for p in pts if p[3]]
ax.plot([x for x, _ in sorted(tx)], [y for _, y in sorted(tx)],
        "--", color=C_TEXT, lw=1.0, alpha=0.5, zorder=1)
ax.plot([x for x, _ in sorted(ac)], [y for _, y in sorted(ac)],
        "--", color=C_AC, lw=1.0, alpha=0.5, zorder=1)

for label, pm, f1, is_ac in pts:
    col = C_AC if is_ac else C_TEXT
    if abs(pm - 3.58) < 0.01:          # proposed model: star, larger
        ax.scatter(pm, f1, s=280, marker="*", color=col, edgecolor=INK,
                   linewidth=0.8, zorder=5)
        ax.annotate("ViACaPu (proposed)\n3.58M, 0.927", (pm, f1),
                    textcoords="offset points", xytext=(10, -4), fontsize=8.5,
                    color=INK, fontweight="bold")
    else:
        ax.scatter(pm, f1, s=70, color=col, edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(f"{pm:.2f}M", (pm, f1), textcoords="offset points",
                    xytext=(6, 5), fontsize=7.5, color=MUTED)

ax.set_xlabel("Parameters (millions)")
ax.set_ylabel("Punctuation F1")
ax.set_xlim(2.0, 9.6)
ax.set_ylim(0.79, 0.94)
ax.yaxis.grid(True, color=GRID, lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
ax.tick_params(colors=MUTED, length=3)

# legend
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_TEXT,
              markeredgecolor="white", markersize=8, label="Text-only"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AC,
              markeredgecolor="white", markersize=8, label="Acoustic"),
       Line2D([0], [0], marker="*", color="w", markerfacecolor=C_AC,
              markeredgecolor=INK, markersize=13, label="Proposed (3.58M)")]
ax.legend(handles=leg, frameon=False, loc="lower right", handletextpad=0.4)

fig.tight_layout()
fig.savefig(OUT / "fig5_capacity_ablation.pdf")
fig.savefig(OUT / "fig5_capacity_ablation.png", dpi=300)
print("saved fig5_capacity_ablation")
