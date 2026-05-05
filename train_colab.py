# ============================================================
# DeepSkinCNN v4 — Google Colab Training Script
# ============================================================
# QUICK START (Colab):
#   1. Runtime → Change runtime type → T4 GPU (free) or A100 (Pro+)
#   2. Upload train_colab.py + data_loader.py via the Files panel
#   3. Run:  !python train_colab.py
#   4. Checkpoints auto-save to Google Drive every epoch
#
# RESUME after a disconnect:
#   Set RESUME = True (already the default) and re-run.
# ============================================================

import os
import sys
import json
import time
import random
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# ── Mixed Precision (AMP) ───────────────────────────────────
# PyTorch ≥ 2.0: import from torch.amp (not torch.cuda.amp)
# Falls back to the old location for older installs.
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
except ImportError:                          # PyTorch < 2.0
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    _AMP_DEVICE = None                       # autocast needs no device arg

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.notebook import tqdm              # notebook-friendly progress bars


# ============================================================
# STEP 0 — COLAB ENVIRONMENT SETUP
# ============================================================
def setup_colab():
    """Mount Drive, install missing packages, print GPU info."""

    # ── Google Drive ────────────────────────────────────────
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("✅ Google Drive mounted at /content/drive")
    except ModuleNotFoundError:
        print("⚠️  Not in Colab — Drive mount skipped.")
    except Exception as exc:
        print(f"⚠️  Drive mount issue: {exc}")

    # ── Install missing packages ─────────────────────────────
    required = {
        "torch":   "torch torchvision",
        "sklearn": "scikit-learn",
        "tqdm":    "tqdm",
    }
    for pkg, install_name in required.items():
        try:
            __import__(pkg)
        except ImportError:
            print(f"📦 Installing {install_name} ...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", install_name],
                check=True,
            )

    # ── GPU Diagnostics ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("GPU DIAGNOSTICS")
    print("=" * 55)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU     : {gpu_name}")
        print(f"  VRAM    : {vram_gb:.1f} GB")
        print(f"  CUDA    : {torch.version.cuda}")
        print(f"  cuDNN   : {torch.backends.cudnn.version()}")
        print(f"  PyTorch : {torch.__version__}")

        if vram_gb >= 35:
            rec_batch = 128
        elif vram_gb >= 20:
            rec_batch = 64
        elif vram_gb >= 14:
            rec_batch = 32
        else:
            rec_batch = 16
        print(f"  Recommended batch size for {gpu_name}: {rec_batch}")
    else:
        print("  ⚠️  No GPU — training on CPU will be very slow.")
        print("      Go to: Runtime → Change runtime type → T4 GPU")
    print("=" * 55 + "\n")


setup_colab()


# ============================================================
# STEP 1 — OUTPUT PATHS
# Artifacts saved to Drive when available; falls back to /content.
# ============================================================
if os.path.exists("/content/drive/MyDrive"):
    BASE_DIR = "/content/drive/MyDrive/DeepSkinCNN"
elif os.path.exists("/content"):
    BASE_DIR = "/content/DeepSkinCNN"
else:
    BASE_DIR = "./DeepSkinCNN"

os.makedirs(BASE_DIR, exist_ok=True)
print(f"📁 Artifacts directory: {BASE_DIR}")


# ============================================================
# STEP 2 — REPRODUCIBILITY
# ============================================================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False   # deterministic = no benchmark

set_seed(42)


# ============================================================
# STEP 3 — DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================
# STEP 4 — CONFIG
# Batch size auto-selected from VRAM at runtime.
# ============================================================
def _auto_batch_size() -> int:
    if not torch.cuda.is_available():
        return 16
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram >= 35:  return 128   # A100 40 GB
    if vram >= 20:  return 64    # A100 24 GB / V100
    if vram >= 14:  return 32    # T4 16 GB  ← free Colab default
    return 16

CONFIG = {
    "image_size":      224,
    "num_classes":     10,
    "epochs":          100,
    "patience":        20,          # early-stopping patience
    "lr":              5e-4,
    "weight_decay":    1e-4,
    "label_smoothing": 0.05,
    "mixup_alpha":     0.2,
    "batch_size":      _auto_batch_size(),
    "model_path":      os.path.join(BASE_DIR, "model_v4.pth"),        # latest
    "best_model_path": os.path.join(BASE_DIR, "model_v4_best.pth"),   # best F1
    "labels_path":     os.path.join(BASE_DIR, "labels.json"),
    "config_path":     os.path.join(BASE_DIR, "config.json"),
    "history_path":    os.path.join(BASE_DIR, "training_history.json"),
    "use_amp":         torch.cuda.is_available(),
}

print(f"⚙️  Batch size (auto): {CONFIG['batch_size']}")
print(f"⚙️  Mixed precision  : {CONFIG['use_amp']}\n")


# ============================================================
# DATA LOADERS  (imported from data_loader.py)
# ============================================================
try:
    from data_loader import (
        train_loader, val_loader, test_loader,
        class_weights, train_dataset,
    )
    print("✅ data_loader.py imported successfully\n")
except ImportError as exc:
    print(f"\n❌ Cannot import data_loader.py: {exc}")
    print("   Make sure data_loader.py is in the same directory.")
    print("   In Colab: upload via the Files panel (left sidebar).")
    sys.exit(1)


# ============================================================
# FOCAL LOSS  (class-weighted + label smoothing)
# ============================================================
class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight=None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(
            inputs, targets,
            weight         = self.weight,
            label_smoothing= self.label_smoothing,
            reduction      = "none",
        )
        pt    = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ============================================================
# STOCHASTIC DEPTH (DropPath)
# ============================================================
class StochasticDepth(nn.Module):
    def __init__(self, drop_rate: float = 0.0):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training or self.drop_rate == 0.0:
            return x
        survival = 1.0 - self.drop_rate
        mask     = torch.rand(x.size(0), 1, 1, 1, device=x.device) < survival
        return x * mask.float() / survival


# ============================================================
# CHANNEL ATTENTION
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        bottleneck    = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg + mx).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# ============================================================
# SPATIAL ATTENTION
# ============================================================
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg     = torch.mean(x, dim=1, keepdim=True)
        mx, _   = torch.max(x,  dim=1, keepdim=True)
        scale   = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


# ============================================================
# CBAM  (Channel + Spatial Attention)
# ============================================================
class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ============================================================
# CONV BLOCK  (downsampling + optional residual)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout: float        = 0.25,
        use_residual: bool    = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.use_residual = use_residual

        self.conv_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.cbam = CBAM(out_ch)

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if use_residual and in_ch != out_ch
            else None
        )

        self.drop_path = StochasticDepth(drop_path_rate)
        self.act       = nn.GELU()
        self.pool      = nn.MaxPool2d(2)
        self.dropout   = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.drop_path(self.cbam(self.conv_path(x)))
        if self.use_residual:
            out = out + (self.shortcut(x) if self.shortcut is not None else x)
        return self.dropout(self.pool(self.act(out)))


# ============================================================
# REFINE BLOCK  (identity residual, no downsampling)
# ============================================================
class RefineBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.30, drop_path_rate: float = 0.0):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.cbam      = CBAM(channels)
        self.drop_path = StochasticDepth(drop_path_rate)
        self.act       = nn.GELU()
        self.dropout   = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.drop_path(self.cbam(self.conv_path(x)))
        return self.dropout(self.act(out + x))


# ============================================================
# MODEL — DeepSkinCNN v4
# Input : [B, 3, 224, 224]
# Output: [B, num_classes]
# ============================================================
class DeepSkinCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Encoder (each block halves spatial dims)
        self.block1 = ConvBlock(3,   64,  dropout=0.15, use_residual=False, drop_path_rate=0.00)
        self.block2 = ConvBlock(64,  128, dropout=0.20, use_residual=True,  drop_path_rate=0.05)
        self.block3 = ConvBlock(128, 256, dropout=0.25, use_residual=True,  drop_path_rate=0.10)
        self.block4 = ConvBlock(256, 384, dropout=0.25, use_residual=True,  drop_path_rate=0.15)
        self.block5 = ConvBlock(384, 512, dropout=0.30, use_residual=True,  drop_path_rate=0.20)

        # Block 6 — expanded inline (512 → 768, no pooling)
        self.b6_conv = nn.Sequential(
            nn.Conv2d(512, 768, 3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.GELU(),
            nn.Conv2d(768, 768, 3, padding=1, bias=False),
            nn.BatchNorm2d(768),
        )
        self.b6_cbam      = CBAM(768)
        self.b6_drop_path = StochasticDepth(0.20)
        self.b6_shortcut  = nn.Sequential(
            nn.Conv2d(512, 768, 1, bias=False),
            nn.BatchNorm2d(768),
        )
        self.b6_act     = nn.GELU()
        self.b6_dropout = nn.Dropout2d(0.30)

        # Refinement (no change in spatial dims)
        self.refine1 = RefineBlock(768, dropout=0.30, drop_path_rate=0.20)
        self.refine2 = RefineBlock(768, dropout=0.30, drop_path_rate=0.20)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.50),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # Block 6 with residual
        out = self.b6_drop_path(self.b6_cbam(self.b6_conv(x)))
        x   = self.b6_dropout(self.b6_act(out + self.b6_shortcut(x)))

        x = self.refine1(x)
        x = self.refine2(x)
        x = self.gap(x)
        return self.classifier(x)


# ============================================================
# MIXUP
# ============================================================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam   = float(np.random.beta(alpha, alpha))
    idx   = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam: float):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# AMP HELPER — wraps autocast for both PyTorch ≥ 2.0 and < 2.0
# ============================================================
from contextlib import contextmanager

@contextmanager
def amp_autocast(enabled: bool):
    """
    Context manager that calls autocast with the correct signature
    for both PyTorch ≥ 2.0 (requires device string) and < 2.0.
    """
    if not enabled:
        yield
    elif _AMP_DEVICE is not None:      # torch ≥ 2.0
        with autocast(device_type=_AMP_DEVICE, enabled=True):
            yield
    else:                              # torch < 2.0
        with autocast(enabled=True):   # type: ignore
            yield


# ============================================================
# MODEL + OPTIMISER SETUP
# ============================================================
model = DeepSkinCNN(num_classes=CONFIG["num_classes"]).to(device)

criterion = FocalLoss(
    gamma           = 2.0,
    weight          = class_weights.to(device),
    label_smoothing = CONFIG["label_smoothing"],
)
optimizer = optim.AdamW(
    model.parameters(),
    lr           = CONFIG["lr"],
    weight_decay = CONFIG["weight_decay"],
)
scheduler = OneCycleLR(
    optimizer,
    max_lr           = CONFIG["lr"],
    epochs           = CONFIG["epochs"],
    steps_per_epoch  = len(train_loader),
    pct_start        = 0.2,
    anneal_strategy  = "cos",
    div_factor       = 25.0,
    final_div_factor = 1e4,
)

# GradScaler — prevents float16 underflow during AMP backward pass
scaler = GradScaler(enabled=CONFIG["use_amp"])

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

# ── torch.compile (PyTorch ≥ 2.0) ───────────────────────────
# Fuses ops into optimised CUDA kernels → 10–30 % faster.
_torch_major = int(torch.__version__.split(".")[0])
if _torch_major >= 2 and torch.cuda.is_available():
    try:
        model = torch.compile(model)
        print("✅ torch.compile() enabled — optimised CUDA kernels active")
    except Exception as exc:
        print(f"⚠️  torch.compile() skipped: {exc}")


# ============================================================
# TRAINING HISTORY
# Saved to Drive after every epoch so a disconnect loses nothing.
# ============================================================
history: dict = {
    "train_loss": [],
    "val_loss":   [],
    "val_acc":    [],
    "val_f1":     [],
    "lr":         [],
    "epoch_time": [],
}

def save_history():
    with open(CONFIG["history_path"], "w") as fh:
        json.dump(history, fh, indent=2)


# ============================================================
# EPOCH RUNNERS
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, scaler, desc: str = ""):
    """One training epoch with MixUp + AMP.
    Accuracy is not tracked here because MixUp mixes targets.
    """
    model.train()
    total_loss = 0.0
    loop       = tqdm(loader, desc=desc, leave=False)

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        mixed, y_a, y_b, lam = mixup_data(images, labels, CONFIG["mixup_alpha"])

        optimizer.zero_grad(set_to_none=True)   # faster than .zero_grad()

        with amp_autocast(CONFIG["use_amp"]):
            outputs = model(mixed)
            loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def eval_epoch(model, loader, desc: str = ""):
    """Validation / test epoch — returns (accuracy %, avg_loss, macro_f1)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []
    loop       = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp_autocast(CONFIG["use_amp"]):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / total:.1f}%",
            )

    avg_loss = total_loss / len(loader)
    acc      = 100.0 * correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, avg_loss, macro_f1


# ============================================================
# VRAM MONITOR
# ============================================================
def print_vram():
    if not torch.cuda.is_available():
        return
    used  = torch.cuda.memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    pct   = 100 * used / total
    bar   = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
    print(f"  🖥️  VRAM  [{bar}] {used:.1f}/{total:.1f} GB ({pct:.0f}%)")
    if pct > 85:
        print("  ⚠️  VRAM > 85% — consider reducing CONFIG['batch_size'] if OOM.")


# ============================================================
# TRAINING LOOP
# ============================================================
def train_model(
    model,
    epochs:  int = CONFIG["epochs"],
    patience: int = CONFIG["patience"],
):
    best_val_f1      = 0.0
    patience_counter = 0

    print(f"\n🚀 Starting training")
    print(f"   Epochs     : {epochs}")
    print(f"   Patience   : {patience}")
    print(f"   Batch size : {CONFIG['batch_size']}")
    print(f"   AMP        : {CONFIG['use_amp']}")
    print(f"   Checkpoints: {BASE_DIR}\n")

    for epoch in range(epochs):
        ep    = epoch + 1
        t0    = time.time()

        train_loss = train_epoch(
            model, train_loader,
            optimizer = optimizer,
            scheduler = scheduler,
            scaler    = scaler,
            desc      = f"Epoch [{ep}/{epochs}] Train",
        )
        val_acc, val_loss, val_f1 = eval_epoch(
            model, val_loader,
            desc = f"Epoch [{ep}/{epochs}] Val",
        )

        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss,   6))
        history["val_acc"].append(round(val_acc,    4))
        history["val_f1"].append(round(val_f1,     6))
        history["lr"].append(current_lr)
        history["epoch_time"].append(round(elapsed, 2))
        save_history()

        print(f"\nEpoch {ep}/{epochs}  |  LR: {current_lr:.2e}  |  Time: {elapsed:.1f}s")
        print(f"  Train → Loss: {train_loss:.4f}  (acc not shown — MixUp active)")
        print(f"  Val   → Loss: {val_loss:.4f}  |  Acc: {val_acc:.2f}%  |  F1: {val_f1:.4f}")
        print_vram()

        # Always save latest weights (safe against mid-run disconnects)
        torch.save(model.state_dict(), CONFIG["model_path"])

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["best_model_path"])
            print(f"  ✅ Best model saved  →  Val Macro F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("\n🛑 Early stopping triggered.")
                break

    print(f"\n✅ Training complete.")
    print(f"   Best Val Macro F1 : {best_val_f1:.4f}")
    print(f"   Best model        : {CONFIG['best_model_path']}")
    print(f"   History           : {CONFIG['history_path']}")


# ============================================================
# TEST + FULL METRICS
# ============================================================
def test_model(model):
    print("\n=== LOADING BEST MODEL FOR TEST ===")
    model.load_state_dict(
        torch.load(CONFIG["best_model_path"], map_location=device)
    )

    # train_dataset is the full ImageFolder (returned by data_loader.py)
    # — it has a .classes attribute even though loaders use Subsets.
    class_names = train_dataset.classes

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            with amp_autocast(CONFIG["use_amp"]):
                outputs = model(images.to(device, non_blocking=True))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    ))
    print("=== CONFUSION MATRIX ===")
    print(confusion_matrix(all_labels, all_preds))

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"\n✅ Final Test Macro F1: {macro_f1:.4f}")
    return all_preds, all_labels


# ============================================================
# SINGLE-IMAGE INFERENCE
# ============================================================
def predict_single_image(image_tensor: torch.Tensor) -> dict:
    """
    Args:
        image_tensor: shape [1, 3, 224, 224], ImageNet-normalised.

    Returns:
        Dict mapping class name → probability, sorted highest first.
    """
    class_names = train_dataset.classes
    model.eval()
    with torch.no_grad():
        with amp_autocast(CONFIG["use_amp"]):
            probs = torch.softmax(
                model(image_tensor.to(device)), dim=1
            )[0]

    results = {
        class_names[i]: round(float(probs[i]), 4)
        for i in range(len(class_names))
    }
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


# ============================================================
# SAVE LABELS + CONFIG JSON
# ============================================================
def save_artifacts():
    labels = {str(i): name for i, name in enumerate(train_dataset.classes)}
    with open(CONFIG["labels_path"], "w") as fh:
        json.dump(labels, fh, indent=2)
    with open(CONFIG["config_path"], "w") as fh:
        json.dump(CONFIG, fh, indent=2)
    print(f"✅ Saved labels  → {CONFIG['labels_path']}")
    print(f"✅ Saved config  → {CONFIG['config_path']}")


# ============================================================
# RESUME TRAINING
# Loads the latest checkpoint then calls train_model().
# OneCycleLR restarts from step 0 but learned weights are
# preserved — it converges quickly from where it left off.
# ============================================================
def resume_training(model):
    ckpt = CONFIG["model_path"]
    if not os.path.exists(ckpt):
        print(f"No checkpoint at '{ckpt}' — starting from scratch.")
        train_model(model)
        return
    print(f"🔄 Resuming from checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    train_model(model)


# ============================================================
# PLOT TRAINING CURVES (inline in Colab)
# ============================================================
def plot_history():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    if not os.path.exists(CONFIG["history_path"]):
        print("No history file — run training first.")
        return

    with open(CONFIG["history_path"]) as fh:
        h = json.load(fh)

    epochs = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("DeepSkinCNN v4 — Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, h["train_loss"], label="Train Loss", color="#e74c3c")
    axes[0].plot(epochs, h["val_loss"],   label="Val Loss",   color="#3498db")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, h["val_acc"], color="#2ecc71")
    axes[1].set_title("Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)

    # Macro F1
    best_f1 = max(h["val_f1"])
    axes[2].plot(epochs, h["val_f1"], color="#9b59b6")
    axes[2].axhline(
        y=best_f1, color="#e74c3c", linestyle="--", alpha=0.6,
        label=f"Best: {best_f1:.4f}",
    )
    axes[2].set_title("Val Macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "training_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Plot saved → {out}")


# ============================================================
# MAIN
# Set RESUME = False to train from scratch.
# Set RESUME = True  to continue from the last checkpoint.
# ============================================================
RESUME = True   # ← change this flag

if __name__ == "__main__":
    save_artifacts()

    if RESUME:
        resume_training(model)
    else:
        train_model(model)

    test_model(model)
    plot_history()