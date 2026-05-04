import os
import shutil
import random
from pathlib import Path
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path("dataset")          # adjust if your dataset folder is elsewhere
IMG_CLASSES   = BASE_DIR / "IMG_CLASSES"
TRAIN_DIR     = BASE_DIR / "train"
VAL_DIR       = BASE_DIR / "validation"
TEST_DIR      = BASE_DIR / "test"

IMG_SIZE      = (256, 256)
TRAIN_RATIO   = 0.80
VAL_RATIO     = 0.10
# TEST_RATIO  = 0.10  (whatever remains)

SEED          = 42
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
# ─────────────────────────────────────────────────────────────────────────────


def resize_and_save(src: Path, dst: Path):
    """Open an image, resize it to IMG_SIZE, and save to dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")           # ensure 3-channel (handles RGBA/grayscale)
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        img.save(dst)


def split_disease_folder(disease_dir: Path):
    """Split one disease folder across train / val / test."""
    disease_name = disease_dir.name

    # Collect all valid images
    images = sorted([
        f for f in disease_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    ])

    if not images:
        print(f"  [SKIP] No images found in: {disease_dir}")
        return

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)
    # give any remainder to test
    n_test  = n_total - n_train - n_val

    splits = {
        TRAIN_DIR / disease_name: images[:n_train],
        VAL_DIR   / disease_name: images[n_train : n_train + n_val],
        TEST_DIR  / disease_name: images[n_train + n_val :],
    }

    for dest_folder, img_list in splits.items():
        dest_folder.mkdir(parents=True, exist_ok=True)
        for img_path in img_list:
            dest_file = dest_folder / img_path.name
            resize_and_save(img_path, dest_file)

    print(f"  ✓  {disease_name:45s}  total={n_total:4d}  "
          f"train={n_train:4d}  val={n_val:4d}  test={n_test:4d}")


def main():
    random.seed(SEED)

    if not IMG_CLASSES.exists():
        raise FileNotFoundError(f"IMG_CLASSES folder not found: {IMG_CLASSES.resolve()}")

    # Collect disease folders (exclude train/val/test if they somehow live here)
    skip = {"train", "test", "validation"}
    disease_dirs = sorted([
        d for d in IMG_CLASSES.iterdir()
        if d.is_dir() and d.name.lower() not in skip
    ])

    if not disease_dirs:
        raise RuntimeError("No disease sub-folders found inside IMG_CLASSES.")

    print(f"\nFound {len(disease_dirs)} disease class(es) in {IMG_CLASSES}\n")
    print(f"{'Class':<45}  {'Total':>5}  {'Train':>5}  {'Val':>4}  {'Test':>4}")
    print("─" * 70)

    for d in disease_dirs:
        split_disease_folder(d)

    print("\n✅  Done!  Resized images written to:")
    print(f"   train      → {TRAIN_DIR.resolve()}")
    print(f"   validation → {VAL_DIR.resolve()}")
    print(f"   test       → {TEST_DIR.resolve()}")


if __name__ == "__main__":
    main()