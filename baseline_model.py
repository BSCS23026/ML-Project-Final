import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import train_loader, val_loader, class_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== SIMPLE BASELINE CNN =====
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# ===== INIT MODEL =====
model = BaselineCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ===== TRAINING LOOP =====
def train_model(model, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # ✅ tqdm wraps the loader — shows batch progress live
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # ✅ Update tqdm bar with live loss & acc
            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"\n✅ Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        validate_model(model, epoch, epochs)


# ===== VALIDATION =====
def validate_model(model, epoch, epochs):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"📊 Epoch {epoch+1}/{epochs} | Val Acc: {acc:.2f}%\n")


# ===== RUN =====
if __name__ == "__main__":
    train_model(model, epochs=5)