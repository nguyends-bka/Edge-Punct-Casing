"""fig7: lưới tách biến aux vs gate-bias — mỗi seed một điểm.

Thông điệp: cấu hình không có aux phân tán rộng theo seed; bật aux thì chụm lại
và nhảy lên. gate_bias một mình nằm đúng mức text-only.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "figures"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.7,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "axes.axisbelow": True,
})
C_TEXT, C_AC = "#4c8bf5", "#e8743b"
INK, GRID, MUTED = "#1a1a1a", "#dddddd", "#666666"

# (nhãn, bias cổng, aux, [F1 theo seed 42/43/44], có aux?)
cfgs = [
    ("Chỉ văn bản",            "---",    "---", [0.820, 0.819, 0.822], False),
    ("Ngữ âm",                 "mặc định", "0",   [0.922, 0.819, 0.820], False),
    ("Ngữ âm\n+ bias cổng",    "$-2.0$", "0",   [0.817, 0.819, 0.818], False),
    ("Ngữ âm\n+ aux",          "mặc định", "0.5", [0.924, 0.929, 0.924], True),
    ("Ngữ âm\n+ bias + aux",   "$-2.0$", "0.5", [0.928, 0.930, 0.923], True),
]

fig, ax = plt.subplots(figsize=(3.4, 2.7))
rng = np.random.RandomState(0)
for i, (lab, gb, aux, vals, has_aux) in enumerate(cfgs):
    col = C_AC if has_aux else C_TEXT
    jit = np.linspace(-0.13, 0.13, len(vals))
    ax.scatter(np.full(len(vals), i) + jit, vals, s=28, color=col,
               edgecolor="white", linewidth=0.7, zorder=4)
    m = float(np.mean(vals))
    ax.plot([i - 0.26, i + 0.26], [m, m], color=col, lw=1.6, zorder=3)

ax.axhline(0.820, color=MUTED, lw=0.7, ls=":", zorder=1)
ax.text(-0.42, 0.8215, "mức chỉ văn bản", fontsize=6.5, color=MUTED,
        ha="left", va="bottom")

ax.set_xticks(range(len(cfgs)))
ax.set_xticklabels([c[0] for c in cfgs], fontsize=7)
ax.set_ylabel("F1 dấu câu", color=INK)
ax.set_ylim(0.802, 0.942)
ax.yaxis.grid(True, color=GRID, lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
ax.tick_params(colors=MUTED, length=3)

h = [plt.Line2D([], [], marker="o", ls="none", color=C_TEXT,
                markeredgecolor="white", markersize=6, label="không có aux"),
     plt.Line2D([], [], marker="o", ls="none", color=C_AC,
                markeredgecolor="white", markersize=6, label="có aux")]
ax.legend(handles=h, frameon=False, loc="upper center", ncol=2,
          bbox_to_anchor=(0.5, -0.22), handlelength=1.1)

fig.tight_layout()
fig.savefig(OUT / "fig7_disentangle.pdf")
fig.savefig(OUT / "fig7_disentangle.png", dpi=300)
print("saved fig7_disentangle")
