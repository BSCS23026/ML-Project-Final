import os
import random
from PIL import Image

# ===== YOUR DATASET PATH =====
DATASET_PATH = r"D:\uni\6th sem\ML\dataset\IMG_CLASSES"

# ===== SETTINGS =====
IMG_SIZE = (128, 128)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

random.seed(42)


def process_class(class_path):
    print(f"\nProcessing class: {class_path}")

    # get all JPG images ONLY from main folder
    all_files = []

    for f in os.listdir(class_path):
        full_path = os.path.join(class_path, f)

        # skip folders (train, test, validation)
        if os.path.isdir(full_path):
            continue

        if f.lower().endswith(".jpg"):
            all_files.append(f)

    total = len(all_files)

    if total == 0:
        print("No images found, skipping...")
        return

    print(f"Total images found: {total}")

    random.shuffle(all_files)

    # split indices
    train_end = int(total * TRAIN_SPLIT)
    val_end = int(total * (TRAIN_SPLIT + VAL_SPLIT))

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    splits = {
        "train": train_files,
        "validation": val_files,
        "test": test_files
    }

    for split_name, files in splits.items():
        split_folder = os.path.join(class_path, split_name)

        os.makedirs(split_folder, exist_ok=True)

        print(f"{split_name}: {len(files)} images")

        for file in files:
            src_path = os.path.join(class_path, file)
            dst_path = os.path.join(split_folder, file)

            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize(IMG_SIZE)

                img.save(dst_path)

            except Exception as e:
                print(f"Error with {file}: {e}")

    print("Done with class.")


# ===== MAIN =====
def main():
    print("Starting preprocessing and splitting...\n")

    for class_name in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, class_name)

        if os.path.isdir(class_path):
            process_class(class_path)

    print("\nALL DONE SUCCESSFULLY ✅")


if __name__ == "__main__":
    main()