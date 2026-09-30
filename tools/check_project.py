#!/usr/bin/env python3
"""Kiểm tra project còn thiếu gì — chạy được trên cả Linux lẫn Windows.

Dùng khi vừa clone/copy project sang máy mới và muốn biết:
  - file nào trong git bị thiếu hoặc hỏng (so khớp hash với git)
  - data đã có chưa, có đúng kích thước không
  - môi trường Python đã đủ thư viện chưa
  - checkpoint có nạp được không

Cách dùng:
    python tools/check_project.py
    python tools/check_project.py --data_dir E:\\2026\\data\\data_audio_full
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Kích thước thật của data Dolly (đo trên máy gốc), dùng để phát hiện tải dở.
DATA_FILES = {
    "mel.f16": 56_068_359_360,
    "meta.npz": 135_258_199,
}

OK, WARN, BAD = "  OK  ", " THIẾU", " HỎNG "


def hr(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def section(title):
    print()
    print(title)
    print("-" * 64)


def check_git_files():
    """So khớp mọi file trong git với đĩa, dùng `git status` + `git ls-files`."""
    section("1. FILE TRONG GIT")
    try:
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"],
            capture_output=True, text=True, check=True).stdout.split()
    except Exception as e:
        print(f"[{BAD}] không chạy được git: {e}")
        print("       -> thư mục này có phải bản clone từ GitHub không?")
        return False

    missing = [f for f in tracked if not (ROOT / f).exists()]
    print(f"git theo dõi     : {len(tracked)} file")
    print(f"có trên đĩa      : {len(tracked) - len(missing)} file")

    if missing:
        print(f"[{BAD}] thiếu {len(missing)} file:")
        for f in missing[:15]:
            print(f"         {f}")
        if len(missing) > 15:
            print(f"         ... và {len(missing) - 15} file nữa")
        print("       -> chạy: git checkout .")
        return False

    # Nội dung có khớp không (bắt trường hợp file hỏng / LFS pointer).
    try:
        diff = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        diff = ""
    changed = [l for l in diff.splitlines() if l and not l.startswith("??")]
    if changed:
        print(f"[{WARN}] {len(changed)} file khác bản gốc (có thể bạn đã sửa):")
        for l in changed[:10]:
            print(f"         {l}")
    else:
        print(f"[{OK}] mọi file khớp bản trên GitHub")
    return True


def check_checkpoints():
    section("2. CHECKPOINT")
    groups = {
        "exp_seed": 10, "exp_disentangle": 6,
        "exp_via_384_aux": 1, "exp_via_full": 2,
    }
    total_ok = True
    for d, want in groups.items():
        p = ROOT / d
        got = sorted(p.glob("*.pt")) if p.is_dir() else []
        size = sum(f.stat().st_size for f in got)
        tag = OK if len(got) == want else BAD
        if len(got) != want:
            total_ok = False
        print(f"[{tag}] {d:20s} {len(got)}/{want} ckpt   {hr(size):>10s}")
        # ckpt hỏng thường chỉ còn vài trăm byte (git-lfs pointer)
        tiny = [f.name for f in got if f.stat().st_size < 1_000_000]
        if tiny:
            print(f"         -> file quá nhỏ, nhiều khả năng hỏng: {tiny}")
            total_ok = False
    return total_ok


def check_data(data_dir):
    section("3. DATA")
    p = Path(data_dir) if data_dir else ROOT / "data" / "data_audio_full"
    print(f"đường dẫn: {p}")
    if not p.exists():
        print(f"[{WARN}] chưa có data — KHÔNG huấn luyện lại được")
        print("       (vẫn xem được bài báo và chạy suy luận với checkpoint)")
        print("       -> lấy bằng: rsync -avP --partial \\")
        print("            ai@172.18.24.128:/mnt/hdd_ngocmx/Edge-Punct-Casing/data_audio_full/ \\")
        print("            <đích>/data_audio_full/")
        return False

    ok = True
    for name, want in DATA_FILES.items():
        f = p / name
        if not f.exists():
            print(f"[{BAD}] thiếu {name}")
            ok = False
            continue
        got = f.stat().st_size
        if got == want:
            print(f"[{OK}] {name:10s} {hr(got):>10s}")
        else:
            pct = 100 * got / want
            print(f"[{BAD}] {name:10s} {hr(got):>10s}  (cần {hr(want)}, mới {pct:.1f}%)")
            print("       -> tải chưa xong; chạy lại rsync để nối tiếp")
            ok = False
    return ok


def check_env():
    section("4. MÔI TRƯỜNG PYTHON")
    print(f"Python {sys.version.split()[0]}")
    need = ["torch", "numpy", "sentencepiece"]
    optional = ["torchaudio", "soundfile", "onnx", "matplotlib"]
    ok = True
    for m in need:
        try:
            mod = __import__(m)
            v = getattr(mod, "__version__", "?")
            print(f"[{OK}] {m:16s} {v}")
        except ImportError:
            print(f"[{BAD}] {m:16s} thiếu  -> pip install {m}")
            ok = False
    for m in optional:
        try:
            mod = __import__(m)
            print(f"[{OK}] {m:16s} {getattr(mod, '__version__', '?')}  (tuỳ chọn)")
        except ImportError:
            print(f"[{WARN}] {m:16s} thiếu   (tuỳ chọn)")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[{OK}] CUDA  {torch.cuda.get_device_name(0)}")
        else:
            print(f"[{WARN}] không có CUDA — suy luận chạy CPU được, huấn luyện sẽ rất chậm")
    except Exception:
        pass
    return ok


def check_load():
    section("5. NẠP THỬ CHECKPOINT")
    ck = ROOT / "exp_seed" / "best_ac_s43_aux5.pt"
    if not ck.exists():
        print(f"[{WARN}] không có {ck.name}, bỏ qua")
        return False
    try:
        import torch
    except ImportError:
        print(f"[{WARN}] chưa có torch, bỏ qua")
        return False
    try:
        d = torch.load(ck, map_location="cpu")
        sd = d["model"] if "model" in d else d
        n = sum(v.numel() for v in sd.values())
        print(f"[{OK}] {ck.name}: {len(sd)} tensor, {n:,} tham số")
        if n == 3_583_369:
            print(f"[{OK}] khớp con số 3 583 369 trong bài báo")
            return True
        print(f"[{BAD}] lệch so với 3 583 369 của bài")
        return False
    except Exception as e:
        print(f"[{BAD}] nạp lỗi: {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None,
                    help="đường dẫn data_audio_full (mặc định: data/data_audio_full)")
    args = ap.parse_args()

    print("=" * 64)
    print(f"KIỂM TRA PROJECT — {ROOT}")
    print("=" * 64)

    r = {
        "file git": check_git_files(),
        "checkpoint": check_checkpoints(),
        "data": check_data(args.data_dir),
        "môi trường": check_env(),
        "nạp ckpt": check_load(),
    }

    section("KẾT LUẬN")
    for k, v in r.items():
        print(f"[{OK if v else WARN}] {k}")

    core = r["file git"] and r["checkpoint"]
    print()
    if core and r["data"]:
        print(">> ĐẦY ĐỦ: xem bài, chạy suy luận và huấn luyện lại đều được.")
    elif core:
        print(">> ĐỦ DÙNG: xem bài báo và chạy suy luận với checkpoint sẵn có.")
        print("   Thiếu data nên chưa huấn luyện lại được (xem mục 3).")
    else:
        print(">> THIẾU PHẦN CỐT LÕI — xem mục 1 và 2 ở trên.")
    print()


if __name__ == "__main__":
    main()
