import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import numpy as np
import random
import json

from data_loader import train_loader, val_loader, test_loader, class_weights, train_dataset

# =========================
# REPRODUCIBILITY
# Sets all random seeds so results are consistent across runs.
# Without this, weight init, MixUp sampling, and DropPath masks
# all differ between runs — making it impossible to compare experiments.
# =========================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(42)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================
# CONFIG
# =========================
CONFIG = {
    "image_size":      224,
    "num_classes":     10,

    # 100 epochs gives OneCycleLR a full warm-up + decay cycle.
    # patience=20 means we tolerate 20 non-improving epochs before stopping.
    # OneCycleLR has a late-cycle LR decay where val F1 often improves
    # significantly — stopping too early at patience=10 would miss this.
    "epochs":          100,
    "patience":        20,

    # lr=5e-4 (down from your 1e-3):
    # Lower max_lr makes training more stable for a deep 8-block model.
    # OneCycleLR warms up from lr/25 → lr, so effective start LR is ~2e-5.
    # 1e-3 caused loss spikes in deeper CNNs trained from scratch on medical data.
    "lr":              5e-4,

    "weight_decay":    1e-4,
    "label_smoothing": 0.05,  # reduced from 0.1 — see over-regularisation note below

    # MixUp alpha=0.2 (down from your 0.4):
    # Your 10 classes include visually ambiguous pairs (nevi vs melanoma,
    # psoriasis vs eczema). alpha=0.4 creates too many near-50/50 blends
    # that confuse the model on already-hard decision boundaries.
    # alpha=0.2 keeps most lambda values close to 0 or 1 — gentler mixing.
    "mixup_alpha":     0.2,

    "batch_size":      32,
    "model_path":      "model_v4.pth",
    "labels_path":     "labels.json",
    "config_path":     "config.json",
}


# =========================
# FOCAL LOSS + LABEL SMOOTHING
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# =========================
# STOCHASTIC DEPTH (DropPath)
#
# Randomly drops entire residual branches during training.
# Each sample in the batch independently decides whether to drop.
#
# WHY this matters for your model:
#   - 8-block deep from-scratch CNNs overfit more than pretrained models.
#     DropPath is the most effective regulariser for this scenario.
#   - Unlike Dropout (drops neurons), DropPath drops entire computation paths,
#     forcing redundant representations across branches — much stronger signal.
#   - Drop rates increase linearly with depth: shallower blocks learn critical
#     low-level features (edges, colours) — dropping them too often breaks
#     training. Deeper blocks learn high-level patterns — need more regularisation.
# =========================
class StochasticDepth(nn.Module):
    def __init__(self, drop_rate: float = 0.0):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training or self.drop_rate == 0.0:
            return x
        survival = 1.0 - self.drop_rate
        # Per-sample binary mask: shape [B, 1, 1, 1] — entire path dropped per sample
        mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) < survival
        return x * mask.float() / survival   # scale to preserve expected value


# =========================
# CHANNEL ATTENTION (improved SE Block)
#
# Uses BOTH average and max pooling (original SE only used avg).
# Avg pooling: captures overall channel importance (smooth signal).
# Max pooling: captures peak feature activation (salient features).
# Both passed through a shared MLP and summed before sigmoid.
# This is empirically stronger than avg-only SE for medical imaging.
# =========================
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
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale   = self.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# =========================
# SPATIAL ATTENTION
#
# Learns WHERE in the image to focus (which spatial regions matter).
#
# WHY this is critical for your mixed dataset:
#   Dermoscopic images: lesion is centred, background is dark gel.
#   Clinical images:    lesion may be off-centre, surrounded by skin/hair/ruler marks.
#   The model must learn to attend to the lesion region regardless of image type.
#   Spatial attention gives it this ability explicitly.
#
# Implementation:
#   Pool across channels → [B, 2, H, W] (avg + max)
#   Conv7×7 (large receptive field for global spatial context) → sigmoid
#   Multiply with input features → attends to relevant spatial regions
# =========================
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out  = torch.mean(x, dim=1, keepdim=True)    # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)   # [B, 2, H, W]
        scale    = self.sigmoid(self.conv(combined))       # [B, 1, H, W]
        return x * scale


# =========================
# CBAM (Convolutional Block Attention Module)
#
# Channel attention THEN spatial attention — sequential application.
# "What features matter?" → "Where do those features appear?"
#
# For skin disease: channel att selects disease-relevant texture/colour channels,
# spatial att then focuses those channels on the lesion region.
# Sequential (channel first) is empirically better than parallel or spatial-first.
# =========================
class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# =========================
# CONV BLOCK (with CBAM + Residual + Stochastic Depth)
#
# Structure:
#   Conv3×3 → BN → GELU
#   Conv3×3 → BN
#   CBAM (channel + spatial attention)
#   StochasticDepth on the conv branch
#   + residual shortcut (1×1 projection if channels differ)
#   GELU (post-addition activation)
#   MaxPool2d(2)
#   Dropout2d
#
# Changes vs your v3:
#   SE → CBAM                    : adds spatial attention
#   ReLU → GELU                  : smoother gradients, better for deep nets
#   StochasticDepth added         : strongest from-scratch regulariser
#   Post-add activation           : more modern (BN→act after residual add)
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.25,
                 use_residual: bool = False, drop_path_rate: float = 0.0):
        super().__init__()
        self.use_residual = use_residual

        self.conv_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.cbam = CBAM(out_ch)

        if use_residual and in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = None

        self.drop_path = StochasticDepth(drop_path_rate)
        self.act       = nn.GELU()
        self.pool      = nn.MaxPool2d(2)
        self.dropout   = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.conv_path(x)
        out = self.cbam(out)
        out = self.drop_path(out)

        if self.use_residual:
            residual = self.shortcut(x) if self.shortcut else x
            out      = out + residual

        out = self.act(out)
        out = self.pool(out)
        out = self.dropout(out)
        return out


# =========================
# REFINE BLOCK (no spatial downsampling — stays at 7×7)
#
# Extra depth at 7×7 for high-level semantic reasoning.
# Two of these run after all ConvBlocks.
# =========================
class RefineBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.30, drop_path_rate: float = 0.0):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.cbam      = CBAM(channels)
        self.drop_path = StochasticDepth(drop_path_rate)
        self.act       = nn.GELU()
        self.dropout   = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.conv_path(x)
        out = self.cbam(out)
        out = self.drop_path(out)
        out = out + x    # identity residual (same channels, no projection needed)
        out = self.act(out)
        return self.dropout(out)


# =========================
# MODEL — DeepSkinCNN v4 (from scratch)
#
# Input: 3 × 224 × 224
#
# Why wider (up to 768ch) and deeper (6 blocks + 2 refine blocks vs your 5+1):
#   10 classes across two visually distinct domains needs more capacity.
#   512ch (v3) was the bottleneck. 768ch gives 50% more representational
#   capacity at the critical 7×7 feature map where disease patterns live.
#
#   Block 1:  3   →  64  | 224→112 | no residual | drop_path=0.00
#   Block 2:  64  → 128  | 112→ 56 | residual ✓  | drop_path=0.05
#   Block 3: 128  → 256  |  56→ 28 | residual ✓  | drop_path=0.10
#   Block 4: 256  → 384  |  28→ 14 | residual ✓  | drop_path=0.15
#   Block 5: 384  → 512  |  14→  7 | residual ✓  | drop_path=0.20
#   Block 6: 512  → 768  |   7→  7 | residual ✓  | drop_path=0.20  ← no pool
#   Refine1: 768  → 768  |   7→  7 | residual ✓  | drop_path=0.20
#   Refine2: 768  → 768  |   7→  7 | residual ✓  | drop_path=0.20
#
#   GAP:  768 × 7 × 7 → 768
#   Head: 768 → 512 → 256 → 10
#
# Drop_path rates increase linearly 0.0 → 0.2 with depth.
# =========================
class DeepSkinCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Blocks 1–5: halve spatial dims each time, 224 → 7
        self.block1 = ConvBlock(3,   64,  dropout=0.15, use_residual=False, drop_path_rate=0.00)
        self.block2 = ConvBlock(64,  128, dropout=0.20, use_residual=True,  drop_path_rate=0.05)
        self.block3 = ConvBlock(128, 256, dropout=0.25, use_residual=True,  drop_path_rate=0.10)
        self.block4 = ConvBlock(256, 384, dropout=0.25, use_residual=True,  drop_path_rate=0.15)
        self.block5 = ConvBlock(384, 512, dropout=0.30, use_residual=True,  drop_path_rate=0.20)

        # Block 6: expand channels 512→768 WITHOUT spatial downsampling.
        # We're already at 7×7 — another MaxPool would give 3×3, losing too
        # much spatial structure for lesion shape/border analysis.
        self.block6_conv = nn.Sequential(
            nn.Conv2d(512, 768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.GELU(),
            nn.Conv2d(768, 768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768),
        )
        self.block6_cbam      = CBAM(768)
        self.block6_drop_path = StochasticDepth(0.20)
        self.block6_shortcut  = nn.Sequential(
            nn.Conv2d(512, 768, kernel_size=1, bias=False),
            nn.BatchNorm2d(768),
        )
        self.block6_act     = nn.GELU()
        self.block6_dropout = nn.Dropout2d(0.30)

        # Two refine blocks at 7×7
        self.refine1 = RefineBlock(768, dropout=0.30, drop_path_rate=0.20)
        self.refine2 = RefineBlock(768, dropout=0.30, drop_path_rate=0.20)

        self.gap = nn.AdaptiveAvgPool2d(1)   # 768×7×7 → 768×1×1

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

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
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

        # Block 6 with manual residual (no MaxPool)
        out      = self.block6_conv(x)
        out      = self.block6_cbam(out)
        out      = self.block6_drop_path(out)
        residual = self.block6_shortcut(x)
        out      = self.block6_act(out + residual)
        x        = self.block6_dropout(out)

        x = self.refine1(x)
        x = self.refine2(x)

        x = self.gap(x)
        x = self.classifier(x)
        return x


# =========================
# MIXUP
# =========================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam     = float(np.random.beta(alpha, alpha))
    index   = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam: float):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =========================
# SETUP
# =========================
model     = DeepSkinCNN(num_classes=CONFIG["num_classes"]).to(device)
criterion = FocalLoss(
    gamma=2.0,
    weight=class_weights.to(device),
    label_smoothing=CONFIG["label_smoothing"]
)

# AdamW instead of Adam:
# AdamW decouples weight decay from adaptive gradient updates — mathematically
# correct L2 regularisation. Adam's weight_decay is actually broken (it
# interacts with the adaptive LR and doesn't regularise correctly). AdamW fixes this.
optimizer = optim.AdamW(
    model.parameters(),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"]
)

scheduler = OneCycleLR(
    optimizer,
    max_lr=CONFIG["lr"],
    epochs=CONFIG["epochs"],
    steps_per_epoch=len(train_loader),
    pct_start=0.2,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=1e4,
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")


# =========================
# EPOCH RUNNERS
#
# Two separate functions — one for train, one for eval — because MixUp
# makes training accuracy/F1 meaningless to report:
#
#   During MixUp, the model sees mixed_images = λ·xᵢ + (1-λ)·xⱼ.
#   The "true" label is a soft blend of y_a and y_b, not a single hard class.
#   Computing (predicted == labels) against the original hard labels therefore
#   gives a number that is neither the true train accuracy nor a valid proxy.
#   It looks plausible but is misleading — especially for monitoring overfitting.
#
#   Solution: train_epoch reports ONLY loss (always valid).
#             eval_epoch reports loss + accuracy + macro F1 (no MixUp → valid).
#
# Val/test metrics are completely unaffected — they never use MixUp.
# =========================

def train_epoch(model, loader, optimizer, scheduler, desc=""):
    """
    Training pass with MixUp.
    Returns: avg_loss only — acc/F1 are not reported (MixUp makes them invalid).
    """
    model.train()
    total_loss = 0
    loop       = tqdm(loader, desc=desc, leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        mixed_images, y_a, y_b, lam = mixup_data(images, labels, CONFIG["mixup_alpha"])

        optimizer.zero_grad()
        outputs = model(mixed_images)
        loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()   # per-batch — required for OneCycleLR

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / len(loader)


def eval_epoch(model, loader, desc=""):
    """
    Eval pass (val or test). No MixUp → acc and F1 are valid and reported.
    Returns: (accuracy %, avg_loss, macro_f1)
    """
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []
    loop       = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loop.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")

    avg_loss = total_loss / len(loader)
    acc      = 100 * correct / total
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, avg_loss, macro_f1


# =========================
# TRAINING
# =========================
def train_model(model, epochs=CONFIG["epochs"], patience=CONFIG["patience"]):
    best_val_f1      = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        ep = epoch + 1

        # Train: MixUp active → only loss is meaningful
        train_loss = train_epoch(
            model, train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            desc=f"Epoch [{ep}/{epochs}] Train"
        )
        # Val: no MixUp → acc + F1 are valid
        val_acc, val_loss, val_f1 = eval_epoch(
            model, val_loader,
            desc=f"Epoch [{ep}/{epochs}] Val"
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {ep}/{epochs}  |  LR: {current_lr:.2e}")
        print(f"  Train → Loss: {train_loss:.4f}  (acc/F1 not shown — MixUp active)")
        print(f"  Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["model_path"])
            print(f"  ✅ Best model saved (Val Macro F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("\n🛑 Early stopping triggered.")
            break

    print(f"\nTraining complete. Best Val Macro F1: {best_val_f1:.4f}")


# =========================
# VALIDATION
# =========================
def validate_model(model):
    return eval_epoch(model, val_loader, desc="Validating")


# =========================
# TEST + METRICS
# =========================
def test_model(model):
    print("\n=== LOADING BEST MODEL FOR TESTING ===")
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))

    with open(CONFIG["labels_path"]) as f:
        labels_map = json.load(f)
    class_names = [labels_map[str(i)] for i in range(len(labels_map))]

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(all_labels, all_preds))

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"\n✅ Final Test Macro F1: {macro_f1:.4f}")
    return all_preds, all_labels


# =========================
# INFERENCE
# =========================
def predict_single_image(image_tensor: torch.Tensor) -> dict:
    """
    Input : image_tensor — shape [1, 3, 224, 224], ImageNet-normalised
    Output: dict mapping disease name → probability, sorted by confidence
    """
    with open(CONFIG["labels_path"]) as f:
        labels_map = json.load(f)
    class_names = [labels_map[str(i)] for i in range(len(labels_map))]

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(image_tensor.to(device)), dim=1)[0]

    results = {class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))}
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


# =========================
# SAVE ARTIFACTS
# =========================
def save_artifacts():
    labels = {str(i): name for i, name in enumerate(train_dataset.classes)}
    with open(CONFIG["labels_path"], "w") as f:
        json.dump(labels, f, indent=2)
    with open(CONFIG["config_path"], "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"✅ Saved: {CONFIG['labels_path']}, {CONFIG['config_path']}")


# =========================
# RESUME TRAINING
#
# Loads saved weights into the model so training continues from where
# it stopped rather than starting from random init again.
#
# NOTE: OneCycleLR cannot be resumed mid-cycle because it is stateful —
# it expects a fixed total number of steps. We restart the scheduler
# from step 0 but keep the learned weights, which is the standard
# approach for interrupted CPU training. The LR will warm up briefly
# again but the weights are already good so convergence is fast.
# =========================
def resume_training(model, resume_path: str = CONFIG["model_path"]):
    import os
    if not os.path.exists(resume_path):
        print(f"No saved model found at '{resume_path}'. Starting from scratch.")
        train_model(model)
        return

    print(f"Loading weights from '{resume_path}' and resuming training...")
    model.load_state_dict(torch.load(resume_path, map_location=device))
    train_model(model)


# =========================
# RUN
# — Set RESUME = True  to continue from model_v4.pth
# — Set RESUME = False to train from scratch
# =========================
RESUME = True   # <- change this flag

if __name__ == "__main__":
    save_artifacts()
    if RESUME:
        resume_training(model)
    else:
        train_model(model)
    test_model(model)