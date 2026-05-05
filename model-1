import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import numpy as np
import json
import os

from data_loader import train_loader, val_loader, test_loader, class_weights, train_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# CONFIG
# =========================
CONFIG = {
    "image_size": 128,
    "num_classes": 10,
    "epochs": 30,
    "patience": 6,
    "lr": 0.0005,
    "batch_size": 32,
    "model_path": "model.pth",
    "labels_path": "labels.json",
    "config_path": "config.json",
}

# =========================
# MODEL
# =========================
class FinalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FinalCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 4 — added for better feature depth
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =========================
# SETUP
# =========================
model = FinalCNN().to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)

# Reduce LR when val F1 plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# =========================
# TRAINING
# =========================
def train_model(model, epochs=CONFIG["epochs"], patience=CONFIG["patience"]):
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")

        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)

        val_acc, val_loss, val_f1 = validate_model(model)

        # Step scheduler based on val F1
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train  → Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val    → Loss: {val_loss:.4f}  | Acc: {val_acc:.2f}% | Macro F1: {val_f1:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Early stopping on macro F1 (better for imbalanced classes)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["model_path"])
            print(f"  ✅ Best model saved (F1: {best_val_f1:.4f})")
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
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return acc, avg_loss, macro_f1


# =========================
# TEST + METRICS
# =========================
def test_model(model):
    print("\n=== LOADING BEST MODEL FOR TESTING ===")
    model.load_state_dict(torch.load(CONFIG["model_path"]))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(all_labels, all_preds,
                                target_names=train_dataset.classes,
                                digits=4))

    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"\n✅ Final Test Macro F1: {macro_f1:.4f}")

    return all_preds, all_labels


# =========================
# SAVE ARTIFACTS
# =========================
def save_artifacts():
    # labels.json
    labels = {str(i): name for i, name in enumerate(train_dataset.classes)}
    with open(CONFIG["labels_path"], "w") as f:
        json.dump(labels, f, indent=2)

    # config.json
    with open(CONFIG["config_path"], "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"✅ Saved: {CONFIG['labels_path']}, {CONFIG['config_path']}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    save_artifacts()
    train_model(model)
    test_model(model)