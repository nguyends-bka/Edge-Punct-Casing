"""ViACaPu architecture diagram for the paper (Figure 1).

Khớp đúng cấu hình 3.58M dùng trong bài:
  - adjacent_heads=False  -> hai đầu ra cùng nhận h_i
  - có aux energy head trên nhánh ngữ âm (đóng góp phương pháp cốt lõi)
Mỗi khối ghi kèm số tham số thật (đếm từ checkpoint best_ac_s43_aux5.pt) và
shape tensor, để hình tự đứng được mà không cần tra bảng.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "figures"
SURFACE, INK, INK2 = "#ffffff", "#101010", "#5a5a5a"
C_TEXT = "#2a6fc9"    # nhánh văn bản
C_AC = "#1f7a3d"      # nhánh ngữ âm
C_FUSE = "#d99000"    # hợp nhất
C_HEAD = "#4a3aa7"    # đầu ra
C_AUX = "#b3541e"     # mất mát phụ
GREY = "#8a8a8a"

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(12, 8.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
fig.patch.set_facecolor(SURFACE)


def box(x, y, w, h, text, color, tcolor="#ffffff", fs=9.5, params=None, shape=None):
    """Hộp một lớp. params: chuỗi số tham số ghi bên phải. shape: tensor ghi dưới."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
                                fc=color, ec="none", zorder=3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=tcolor, fontsize=fs, fontweight="bold", zorder=4)
    if params:
        ax.text(x + w + 0.8, y + h/2, params, ha="left", va="center",
                fontsize=7.8, color=color, fontweight="bold", zorder=4)
    if shape:
        ax.text(x + w/2, y - 1.5, shape, ha="center", va="top",
                fontsize=7.6, color=INK2, family="monospace", zorder=4)


def arrow(x1, y1, x2, y2, color=INK2, lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=12, color=color, lw=lw, zorder=2))


# ======================= NHÁNH VĂN BẢN (trái) =======================
ax.text(19, 96.5, "NHÁNH VĂN BẢN", ha="center", fontsize=10.5, fontweight="bold", color=C_TEXT)
box(4, 88, 30, 5.2, "văn bản ASR → BPE (vocab 3500)", GREY, fs=8.5, shape="tokens [B, L]")
box(4, 76.5, 30, 5.2, "Embedding (d = 256)", C_TEXT, params="896 K", fs=9)
box(4, 66, 30, 5.2, "3 × Conv1d (k=3) + LN, phần dư", C_TEXT, params="592 K", fs=8.3)
box(4, 55.5, 30, 5.2, "BiLSTM × 2 (128 / hướng)", C_TEXT, params="791 K", fs=9)
box(4, 45, 30, 5.2, "gather sub-word đầu mỗi từ", GREY, fs=8.5)
ax.text(4, 43.4, "$H^t$ [B, W, 256]   1 vector / TỪ", ha="left", va="top",
        fontsize=7.6, color=INK2, family="monospace", zorder=4)
for y in [88, 76.5, 66, 55.5]:
    arrow(19, y, 19, y - 5.3 + 0.1 if False else y - 5.3)

# ======================= NHÁNH NGỮ ÂM (phải) =======================
ax.text(76, 96.5, "NHÁNH NGỮ ÂM", ha="center", fontsize=10.5, fontweight="bold", color=C_AC)
box(61, 88, 30, 5.2, "audio 16 kHz → log-Mel 80", GREY, fs=8.5, shape="mel [B, T, 80]  hop 10 ms")
box(61, 76.5, 30, 5.2, "Conv1d stride 2  (↓2×)", C_AC, params="51 K", fs=9)
box(61, 66, 30, 5.2, "Conv1d stride 2  (↓2× nữa)", C_AC, params="164 K", fs=9)
box(61, 55.5, 30, 5.2, "BiGRU × 1 (128 / hướng)", C_AC, fs=9)
ax.text(61, 53.9, "296 K", ha="left", va="top", fontsize=7.8,
        color=C_AC, fontweight="bold", zorder=4)
for y in [88, 76.5, 66]:
    arrow(76, y, 76, y - 5.3)
ax.text(91, 53.9, "$A$ [B, T/4, 256]  1 vector / KHUNG 40 ms",
        ha="right", va="top", fontsize=7.6, color=INK2, family="monospace")

# --- đầu phụ năng lượng: rẽ xuống dưới bên phải, tách khỏi luồng chính ---
box(66, 45.5, 25, 4.6, "energy_head (d→1)", C_AUX, fs=8.5)
arrow(88, 55.5, 88, 50.1, color=C_AUX, lw=1.3)
ax.text(91, 43.8, "$\\mathcal{L}_{aux}$ : hồi quy năng lượng khung (chỉ khi huấn luyện)",
        ha="right", va="top", fontsize=7.6, color=C_AUX, fontweight="bold")

# ======================= HỢP NHẤT (giữa) =======================
arrow(24, 41.8, 34, 36.5, color=C_TEXT)
ax.text(30.5, 40.2, "Q", fontsize=9.5, color=C_TEXT, fontweight="bold")
arrow(66, 53.5, 56, 36.5, color=C_AC)
ax.text(62.5, 43.5, "K, V", fontsize=9.5, color=C_AC, fontweight="bold")

box(28, 29.5, 34, 6.6,
    "Cross-Attention (4 đầu)\nmỗi TỪ chú ý lên mọi KHUNG", C_FUSE, tcolor=INK, fs=8.5,
    params="263 K")
arrow(45, 29.5, 45, 25.8, color=C_FUSE)
ax.text(46.5, 27.6, "$C$ [B, W, 256]", fontsize=7.6, color=INK2, family="monospace", va="center")

box(28, 20.5, 34, 5.2, "cổng  $Z = \\sigma(W\\,[H^t ; C] + b)$", C_FUSE, tcolor=INK, fs=9,
    params="131 K")
arrow(45, 20.5, 45, 17.2, color=C_FUSE)

ax.add_patch(FancyBboxPatch((25, 12.2), 40, 4.8, boxstyle="round,pad=0.35,rounding_size=1.2",
                            fc="none", ec=C_FUSE, lw=1.8, zorder=3))
ax.text(45, 14.6, "$H^f = Z \\odot H^t + (1-Z) \\odot C$", ha="center", va="center",
        fontsize=10, color=INK, fontweight="bold", zorder=4)

# ======================= ĐẦU RA =======================
arrow(45, 12.2, 45, 9.2)
box(30, 4.0, 30, 5.0, "BiLSTM cấp từ (128 / hướng)", C_TEXT, params="395 K", fs=8.8)

arrow(37, 4.0, 22, 0.2)
arrow(53, 4.0, 68, 0.2)
box(4, -5.4, 30, 5.0, "đầu viết hoa → 4 lớp\nLOWER / UPPER / CAP / MIX", C_HEAD, fs=7.8)
box(61, -5.4, 30, 5.0, "đầu dấu câu → 4 lớp\nNONE / COMMA / PERIOD / ?", C_HEAD, fs=7.8)
ax.text(45, 1.6, "cùng nhận $h_i$\n(2 056 tham số)", ha="center", va="center",
        fontsize=7.4, color=INK2, style="italic")

ax.set_ylim(-8, 100)
ax.text(50, -7.4,
        "ViACaPu — 3 583 369 tham số · nhánh ngữ âm + hợp nhất chỉ chiếm 0.91 M (25 %) "
        "· không cần căn chỉnh thời gian mức từ",
        ha="center", fontsize=8.6, style="italic", color=INK2)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"fig0_architecture.{ext}", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
print("saved fig0_architecture.png + .pdf")
