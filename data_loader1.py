import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter

# ===== PATH =====
BASE_PATH = r"D:\uni\6th sem\ML\dataset"


# ===== TRANSFORMS =====

# TRAIN — strong augmentation pipeline for mixed dermoscopic + clinical images.
# NOTE: NO Resize here — preprocessing.py already saved images at 256×256.
# RandomResizedCrop(224) crops from 256 → gives 16px of augmentation room.
train_transforms = transforms.Compose([
    # ── Geometric ──────────────────────────────────────────────────────────
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),

    # ── Color ──────────────────────────────────────────────────────────────
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05
    ),

    # ── Texture / Frequency ────────────────────────────────────────────────
    # GaussianBlur: dermoscopic images are often slightly blurred due to gel/lens.
    # Wrapped in RandomApply(p=0.3) — only 30% of images get blurred.
    # Previously always-on, which degraded sharp clinical photos every time
    # and removed texture information the model needs to distinguish disease patterns.
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
        p=0.3
    ),

    # Forces model to learn lesion shape, not just color
    transforms.RandomGrayscale(p=0.05),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    # Random erase — mimics occlusions (hair, ruler marks in dermoscopy).
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])


# VALIDATION + TEST — no augmentation, deterministic
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ===== LOAD DATASETS =====
train_dataset = datasets.ImageFolder(
    root=BASE_PATH + r"\train",
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    root=BASE_PATH + r"\validation",
    transform=val_test_transforms
)

test_dataset = datasets.ImageFolder(
    root=BASE_PATH + r"\test",
    transform=val_test_transforms
)


# ===== CLASS WEIGHTS — for FocalLoss =====
# Formula: total_samples / (num_classes × count_i)
# Higher weight = loss penalises rare-class errors more.
#
# WHY WeightedRandomSampler was removed:
#   Previously: sampler + class_weights in FocalLoss were both active.
#   The sampler rebalances the INPUT distribution (batches see ~equal classes).
#   class_weights in FocalLoss rebalances the LOSS SURFACE (minority errors hit harder).
#   Using both simultaneously double-corrects for imbalance:
#     - Sampler makes batches already balanced.
#     - class_weights then over-penalises minority errors on an already-balanced input.
#     - Net effect: model obsesses over rare classes, under-learns majority ones.
#   Fix: class_weights + FocalLoss only. Handles imbalance at the gradient level
#   (more principled). shuffle=True restored since sampler is gone.
labels_list  = [label for _, label in train_dataset.samples]
class_counts = Counter(labels_list)

total_samples = len(train_dataset)
num_classes   = len(class_counts)

class_weights = torch.tensor(
    [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)],
    dtype=torch.float
)


# ===== DATA LOADERS =====
# num_workers=0  — mandatory on Windows CPU to avoid spawn-based deadlocks.
# pin_memory=False — only useful for CUDA; wastes RAM on CPU-only.
# shuffle=True restored — correct now that sampler is removed.
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# ===== DATASET INFO =====
print("\n=== DATASET INFO ===")
print("Classes      :", train_dataset.classes)
print("Num classes  :", len(train_dataset.classes))
print("Train size   :", len(train_dataset))
print("Val size     :", len(val_dataset))
print("Test size    :", len(test_dataset))

print("\n=== CLASS COUNTS ===")
for i, name in enumerate(train_dataset.classes):
    print(f"  [{i}] {name:<40} count={class_counts[i]:>5}  weight={class_weights[i]:.4f}")

print("\n=== BATCH CHECK ===")
images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("\nALL DATA LOADING DONE SUCCESSFULLY ✅")