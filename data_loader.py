import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter

# ===== PATH =====
BASE_PATH = r"D:\uni\6th sem\ML\dataset\IMG_CLASSES"

# ===== TRANSFORMS =====

# TRAIN (with augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(128, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# VALIDATION + TEST (NO augmentation)
val_test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
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

# ===== DATA LOADERS =====
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ===== BASIC INFO =====
print("\n=== DATASET INFO ===")
print("Classes:", train_dataset.classes)
print("Number of classes:", len(train_dataset.classes))
print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))

# ===== CHECK ONE BATCH (THIS IS WHERE YOUR FIRST SNIPPET GOES) =====
print("\n=== CHECKING BATCH ===")
images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

# ===== CLASS WEIGHTS (THIS IS WHERE YOUR SECOND SNIPPET GOES) =====
print("\n=== CALCULATING CLASS WEIGHTS ===")

labels_list = [label for _, label in train_dataset.samples]
class_counts = Counter(labels_list)

total_samples = len(train_dataset)
num_classes = len(class_counts)

class_weights = []

for i in range(num_classes):
    count = class_counts[i]
    weight = total_samples / (num_classes * count)
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float)

print("Class weights:", class_weights)

print("\nALL DATA LOADING DONE SUCCESSFULLY ✅")