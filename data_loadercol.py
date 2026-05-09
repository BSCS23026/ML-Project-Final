# # ============================================================
# # data_loader.py — Skin Diseases Image Dataset (Kaggle)
# # Dataset : https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset
# # Images  : 27,153  |  Classes: 10  |  Format: ImageFolder
# #
# # Auto-detects environment and loads data in the best way:
# #   MODE 1 — Already extracted on disk (fastest)
# #   MODE 2 — Kaggle API download       (Colab, recommended)
# #   MODE 3 — Manual ZIP upload         (Colab, fallback)
# #   MODE 4 — DATASET_ROOT_OVERRIDE     (manual path)
# # ============================================================

# import os
# import sys
# import zipfile
# import subprocess

# import torch
# from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
# from torchvision import datasets, transforms
# import numpy as np


# # ============================================================
# # USER CONFIG — only edit this section if needed
# # ============================================================

# # Kaggle slug (the part after kaggle.com/datasets/)
# KAGGLE_DATASET = "ismailpromus/skin-diseases-image-dataset"

# # Set this to your dataset path if auto-detection fails.
# # Example: "/content/drive/MyDrive/skin-diseases/train"
# # Leave as None to use auto-detection.
# DATASET_ROOT_OVERRIDE = None

# # Split ratios (must sum to 1.0)
# TRAIN_RATIO = 0.75
# VAL_RATIO   = 0.15
# TEST_RATIO  = 0.10

# # DataLoader settings
# BATCH_SIZE   = 32    # overridden by train_colab.py CONFIG at runtime
# NUM_WORKERS  = 2     # 2 is safe for Colab; raise to 4 on local machines
# IMAGE_SIZE   = 224
# PIN_MEMORY   = torch.cuda.is_available()

# # ============================================================


# # ────────────────────────────────────────────────────────────
# # HELPERS
# # ────────────────────────────────────────────────────────────

# def _is_colab() -> bool:
#     """Returns True when running inside Google Colab."""
#     try:
#         import google.colab  # noqa: F401
#         return True
#     except ImportError:
#         return False


# def _find_dataset_root(search_dirs: list) -> str | None:
#     """
#     Walks a list of directories looking for an ImageFolder-style
#     dataset (folders of folders containing image files).
#     Returns the first valid root found, or None.
#     """
#     image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

#     def _looks_like_image_folder(path: str) -> bool:
#         """True if path contains subdirs that hold image files."""
#         try:
#             subdirs = [
#                 d for d in os.listdir(path)
#                 if os.path.isdir(os.path.join(path, d))
#             ]
#             if not subdirs:
#                 return False
#             sample_dir = os.path.join(path, subdirs[0])
#             return any(
#                 os.path.splitext(f)[1].lower() in image_exts
#                 for f in os.listdir(sample_dir)
#             )
#         except PermissionError:
#             return False

#     for base in search_dirs:
#         if not os.path.isdir(base):
#             continue
#         if _looks_like_image_folder(base):
#             return base
#         # Check one level deeper
#         for sub in os.listdir(base):
#             deeper = os.path.join(base, sub)
#             if os.path.isdir(deeper) and _looks_like_image_folder(deeper):
#                 return deeper

#     return None


# def _setup_kaggle_credentials_colab() -> bool:
#     """
#     Guides the user to upload kaggle.json inside Colab and places
#     it at ~/.kaggle/kaggle.json where the Kaggle CLI expects it.
#     Returns True if credentials are ready.
#     """
#     kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
#     if os.path.exists(kaggle_json):
#         return True   # already configured

#     print("\n" + "=" * 60)
#     print("KAGGLE API CREDENTIALS REQUIRED")
#     print("=" * 60)
#     print("Steps:")
#     print("  1. kaggle.com → Your profile → Settings → API")
#     print("  2. Click 'Create New Token' → kaggle.json downloads")
#     print("  3. Upload it in the file picker that appears below\n")

#     try:
#         from google.colab import files
#         print("⬆️  Upload your kaggle.json now:")
#         uploaded = files.upload()
#         if "kaggle.json" in uploaded:
#             os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
#             with open(kaggle_json, "wb") as fh:
#                 fh.write(uploaded["kaggle.json"])
#             os.chmod(kaggle_json, 0o600)
#             print("✅ kaggle.json saved.")
#             return True
#         print("❌ kaggle.json not found in upload.")
#         return False
#     except Exception as exc:
#         print(f"❌ Upload error: {exc}")
#         return False


# def _download_via_kaggle_api(dest_dir: str) -> bool:
#     """Downloads and extracts the dataset via the Kaggle CLI."""
#     print(f"📥 Downloading '{KAGGLE_DATASET}' via Kaggle API → {dest_dir}")
#     try:
#         subprocess.run(
#             ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
#              "--unzip", "-p", dest_dir],
#             check=True
#         )
#         print("✅ Kaggle download complete.")
#         return True
#     except (subprocess.CalledProcessError, FileNotFoundError) as exc:
#         print(f"⚠️  Kaggle API download failed: {exc}")
#         return False


# def _handle_zip_upload_colab() -> str | None:
#     """
#     Fallback for users who manually downloaded the ZIP from Kaggle.
#     Pops up a file picker in Colab and extracts the uploaded ZIP.
#     Returns the extraction directory, or None on failure.
#     """
#     print("\n" + "=" * 60)
#     print("FALLBACK: Upload the dataset ZIP manually")
#     print("=" * 60)
#     print("  1. Download the ZIP from Kaggle to your PC")
#     print("  2. Upload it in the file picker below\n")

#     try:
#         from google.colab import files
#         print("⬆️  Upload your dataset ZIP now:")
#         uploaded = files.upload()
#         for fname, data in uploaded.items():
#             if fname.endswith(".zip"):
#                 zip_path = f"/content/{fname}"
#                 extract_dir = "/content/skin_dataset"
#                 with open(zip_path, "wb") as fh:
#                     fh.write(data)
#                 print(f"✅ ZIP saved: {zip_path}")
#                 print("📦 Extracting — this may take a minute...")
#                 with zipfile.ZipFile(zip_path, "r") as zf:
#                     zf.extractall(extract_dir)
#                 print("✅ Extraction complete.")
#                 return extract_dir
#         print("❌ No .zip file found in upload.")
#         return None
#     except Exception as exc:
#         print(f"❌ ZIP upload failed: {exc}")
#         return None


# def get_dataset_root() -> str:
#     """
#     Master function: finds the dataset root using the best available
#     method and returns its path. Exits with a clear message on failure.
#     """

#     # ── 0. Manual override ───────────────────────────────────
#     if DATASET_ROOT_OVERRIDE and os.path.isdir(DATASET_ROOT_OVERRIDE):
#         print(f"✅ Using override path: {DATASET_ROOT_OVERRIDE}")
#         return DATASET_ROOT_OVERRIDE

#     # ── 1. Search common locations already on disk ───────────
#     candidates = [
#         "/content/skin_dataset",
#         "/content/skin-diseases-image-dataset",
#         "/content/train",
#         "/content/dataset",
#         "/content/drive/MyDrive/DeepSkinCNN/dataset",
#         "./dataset",
#         "./skin_dataset",
#         "./train",
#         "../dataset",
#     ]
#     found = _find_dataset_root(candidates)
#     if found:
#         print(f"✅ Dataset found at: {found}")
#         return found

#     # ── 2. Colab: try Kaggle API ─────────────────────────────
#     if _is_colab():
#         print("Dataset not found locally. Trying Kaggle API...")
#         if _setup_kaggle_credentials_colab():
#             dest = "/content/skin_dataset"
#             os.makedirs(dest, exist_ok=True)
#             if _download_via_kaggle_api(dest):
#                 found = _find_dataset_root([dest])
#                 if found:
#                     return found

#         # ── 3. Colab: manual ZIP upload ──────────────────────
#         print("\nFalling back to manual ZIP upload...")
#         extracted = _handle_zip_upload_colab()
#         if extracted:
#             found = _find_dataset_root([extracted])
#             if found:
#                 return found

#     # ── 4. Give up with a helpful error ──────────────────────
#     print("\n" + "=" * 60)
#     print("❌  DATASET NOT FOUND")
#     print("=" * 60)
#     print("Fix options:")
#     print("  A) Set DATASET_ROOT_OVERRIDE at the top of data_loader.py")
#     print("  B) Place the extracted dataset at ./dataset/")
#     print("  C) Re-run in Colab — the Kaggle / ZIP upload prompt will appear")
#     print("=" * 60)
#     sys.exit(1)


# # ────────────────────────────────────────────────────────────
# # TRANSFORMS
# # ImageNet mean/std centres activations even for scratch training.
# # ────────────────────────────────────────────────────────────

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD  = [0.229, 0.224, 0.225]

# train_transform = transforms.Compose([
#     # Slight oversize then random crop = implicit translation augment
#     transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
#     transforms.RandomCrop(IMAGE_SIZE),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.2),
#     transforms.RandomRotation(degrees=20),
#     transforms.ColorJitter(
#         brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
#     ),
#     transforms.RandomGrayscale(p=0.05),
#     # RandAugment: 2 random ops at magnitude 9 (strong but consistent)
#     transforms.RandAugment(num_ops=2, magnitude=9),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     # Simulates occlusion (hair, ruler marks on dermoscopic images)
#     transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
# ])

# val_test_transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
# ])


# # ────────────────────────────────────────────────────────────
# # LOADER BUILDER
# # ────────────────────────────────────────────────────────────

# def build_loaders(dataset_root: str, batch_size: int = BATCH_SIZE):
#     """
#     Loads the full dataset from `dataset_root` (ImageFolder layout),
#     performs a reproducible 75/15/10 train/val/test split, and returns:
#         train_loader, val_loader, test_loader, class_weights, train_base

#     `train_base` is the full ImageFolder object (has .classes attribute).
#     `class_weights` is a float32 tensor for use with FocalLoss.
#     """

#     print(f"\n📂 Loading dataset from: {dataset_root}")

#     # Load once just to count samples and get targets/classes
#     full_dataset = datasets.ImageFolder(root=dataset_root)
#     total        = len(full_dataset)
#     num_classes  = len(full_dataset.classes)

#     print(f"  Total images : {total:,}")
#     print(f"  Classes ({num_classes}): {full_dataset.classes}")

#     # ── Reproducible split (seed-independent of torch.manual_seed) ──
#     rng     = np.random.RandomState(42)
#     indices = rng.permutation(total)

#     n_train = int(TRAIN_RATIO * total)
#     n_val   = int(VAL_RATIO   * total)

#     train_idx = indices[:n_train]
#     val_idx   = indices[n_train : n_train + n_val]
#     test_idx  = indices[n_train + n_val :]

#     print(f"  Split  → Train: {len(train_idx):,}  "
#           f"Val: {len(val_idx):,}  "
#           f"Test: {len(test_idx):,}")

#     # ── Two base datasets with different transforms ──────────
#     # We must use separate ImageFolder instances because transforms
#     # can't be set per-Subset on a single ImageFolder.
#     train_base    = datasets.ImageFolder(root=dataset_root, transform=train_transform)
#     val_test_base = datasets.ImageFolder(root=dataset_root, transform=val_test_transform)

#     train_dataset = Subset(train_base,    train_idx)
#     val_dataset   = Subset(val_test_base, val_idx)
#     test_dataset  = Subset(val_test_base, test_idx)

#     # ── Class weights (inverse-frequency, normalised to sum=1) ──
#     all_train_labels = [full_dataset.targets[i] for i in train_idx]
#     class_counts     = np.bincount(all_train_labels, minlength=num_classes).astype(float)
#     class_weights    = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32)
#     class_weights   /= class_weights.sum()

#     print("\n  Class distribution (train split):")
#     for i, cls in enumerate(full_dataset.classes):
#         print(f"    [{i:2d}] {cls:<35s} "
#               f"{int(class_counts[i]):>5,} images  "
#               f"weight={class_weights[i]:.4f}")

#     # ── WeightedRandomSampler — balances batches for minority classes ──
#     sample_weights = torch.tensor(
#         [class_weights[full_dataset.targets[i]] for i in train_idx],
#         dtype=torch.float32,
#     )
#     sampler = WeightedRandomSampler(
#         weights     = sample_weights,
#         num_samples = len(train_idx),
#         replacement = True,
#     )

#     # ── DataLoaders ──────────────────────────────────────────
#     _common = dict(
#         num_workers        = NUM_WORKERS,
#         pin_memory         = PIN_MEMORY,
#         prefetch_factor    = 2 if NUM_WORKERS > 0 else None,
#         persistent_workers = NUM_WORKERS > 0,
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size = batch_size,
#         sampler    = sampler,      # replaces shuffle=True
#         **_common,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size = batch_size,
#         shuffle    = False,
#         **_common,
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size = batch_size,
#         shuffle    = False,
#         **_common,
#     )

#     print(f"\n✅ DataLoaders ready  |  Train batches/epoch: {len(train_loader)}")
#     # Return train_base (full ImageFolder, has .classes) not the Subset
#     return train_loader, val_loader, test_loader, class_weights, train_base


# # ────────────────────────────────────────────────────────────
# # EXECUTE ON IMPORT
# # train_colab.py imports this file and uses the exported names:
# #   train_loader, val_loader, test_loader, class_weights, train_dataset
# # ────────────────────────────────────────────────────────────

# _dataset_root = get_dataset_root()

# # Allow train_colab.py to pass its CONFIG batch_size before loaders build
# import inspect as _inspect
# _caller_batch = BATCH_SIZE
# try:
#     _frame = _inspect.currentframe().f_back
#     if _frame and "CONFIG" in _frame.f_globals:
#         _caller_batch = _frame.f_globals["CONFIG"].get("batch_size", BATCH_SIZE)
# except Exception:
#     pass

# train_loader, val_loader, test_loader, class_weights, train_dataset = build_loaders(
#     _dataset_root, batch_size=_caller_batch
# )