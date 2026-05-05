"""
preprocessing.py — Skin Disease Dataset Splitter + Resizer
============================================================
Reads raw images from:
    dataset/IMG_CLASSES/<ClassName>/...

Creates a clean split:
    dataset/IMG_CLASSES/train/<ClassName>/...       (80%)
    dataset/IMG_CLASSES/validation/<ClassName>/...  (10%)
    dataset/IMG_CLASSES/test/<ClassName>/...        (10%)

• Skips any folder already named train / validation / test (safe to re-run)
• Resizes every image to 256x256 — the loader then random-crops to 224x224
  (keeping 256 gives the augmentation pipeline 16px of wiggle room,
   which is standard practice for training from scratch)
• Converts everything to RGB (handles RGBA PNGs, grayscale JPEGs, etc.)
• Prints a per-class summary table and overall stats when done
"""

import os
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# =============================================================
# CONFIG — adjust BASE_PATH to match your machine
# =============================================================
BASE_PATH   = Path(r"D:\uni\6th sem\ML\dataset\IMG_CLASSES")

SPLITS      = {"train": 0.80, "validation": 0.10, "test": 0.10}
TARGET_SIZE = (256, 256)   # Loader crops to 224 — keeping 256 gives augmentation room
SEED        = 42
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Folders that are already split destinations — never treat as source classes
RESERVED    = {"train", "validation", "test"}

random.seed(SEED)


# =============================================================
# HELPERS
# =============================================================

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def resize_and_save(src: Path, dst: Path):
    """
    Open src, convert to RGB, resize to TARGET_SIZE with high-quality
    Lanczos resampling, save to dst as JPEG quality 95.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst = dst.with_suffix(".jpg")   # normalise to JPEG
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            img.save(dst, "JPEG", quality=95)
    except Exception as e:
        print(f"  WARNING: Skipping {src.name}: {e}")


def collect_images(class_dir: Path):
    """Recursively collect all image files inside a class folder."""
    return [p for p in class_dir.rglob("*") if p.is_file() and is_image(p)]


# =============================================================
# MAIN
# =============================================================

def main():
    print(f"\n{'='*60}")
    print(f"  Skin Disease Dataset Preprocessor")
    print(f"  Base path  : {BASE_PATH}")
    print(f"  Split      : {SPLITS}")
    print(f"  Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} px")
    print(f"{'='*60}\n")

    if not BASE_PATH.exists():
        raise FileNotFoundError(
            f"BASE_PATH does not exist: {BASE_PATH}\n"
            "Please update BASE_PATH at the top of preprocessing.py."
        )

    # Discover source class folders (skip already-split destinations)
    class_dirs = [
        d for d in sorted(BASE_PATH.iterdir())
        if d.is_dir() and d.name.lower() not in RESERVED
    ]

    if not class_dirs:
        print("ERROR: No source class folders found. Check BASE_PATH.")
        return

    print(f"Found {len(class_dirs)} classes:\n")
    for d in class_dirs:
        print(f"  {d.name}")
    print()

    # Pre-create all destination folders
    for split in SPLITS:
        for d in class_dirs:
            (BASE_PATH / split / d.name).mkdir(parents=True, exist_ok=True)

    # Per-class stats for summary table
    stats = defaultdict(lambda: {"total": 0, "train": 0, "validation": 0, "test": 0})

    # ── Process each class ────────────────────────────────────
    for class_dir in class_dirs:
        class_name = class_dir.name
        images     = collect_images(class_dir)

        if not images:
            print(f"  WARNING: {class_name}: no images found, skipping.")
            continue

        random.shuffle(images)
        n       = len(images)
        n_val   = max(1, int(n * SPLITS["validation"]))
        n_test  = max(1, int(n * SPLITS["test"]))
        n_train = n - n_val - n_test

        split_map = (
            [("train",      img) for img in images[:n_train]]
            + [("validation", img) for img in images[n_train : n_train + n_val]]
            + [("test",       img) for img in images[n_train + n_val :]]
        )

        stats[class_name]["total"]      = n
        stats[class_name]["train"]      = n_train
        stats[class_name]["validation"] = n_val
        stats[class_name]["test"]       = n - n_train - n_val

        desc = f"{class_name[:35]:<35}"
        for split, src in tqdm(split_map, desc=desc, unit="img"):
            dst = BASE_PATH / split / class_name / src.name
            resize_and_save(src, dst)

    # ── Summary table ─────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {'Class':<45} {'Total':>6}  {'Train':>6}  {'Val':>5}  {'Test':>5}")
    print(f"  {'-'*45} {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")

    totals = {"total": 0, "train": 0, "validation": 0, "test": 0}
    for cls, s in sorted(stats.items()):
        print(
            f"  {cls:<45} {s['total']:>6}  {s['train']:>6}  "
            f"{s['validation']:>5}  {s['test']:>5}"
        )
        for k in totals:
            totals[k] += s[k]

    print(f"  {'-'*45} {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")
    print(
        f"  {'TOTAL':<45} {totals['total']:>6}  {totals['train']:>6}  "
        f"{totals['validation']:>5}  {totals['test']:>5}"
    )
    print(f"{'='*72}\n")
    print("Preprocessing complete!")
    print(f"\nSplit folders created inside: {BASE_PATH}")
    print(f"  train/      - {totals['train']} images")
    print(f"  validation/ - {totals['validation']} images")
    print(f"  test/       - {totals['test']} images")
    print("\nNext steps:")
    print("  1. Verify the summary above looks balanced per class")
    print("  2. Run data_loader.py to confirm loaders work")
    print("  3. Run model.py to start training\n")


if __name__ == "__main__":
    main()