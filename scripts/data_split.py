import os
import shutil
import random

# Define paths and parameters
BASE_DIR = os.path.abspath("datasets")
CANCER_TYPES = ["breast_cancer", "lung_cancer", "skin_cancer"]
SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}

def count_images(directory):
    """Returns a dictionary with the count of images in each category."""
    counts = {"train": {"benign": 0, "malignant": 0},
              "val": {"benign": 0, "malignant": 0},
              "test": {"benign": 0, "malignant": 0}}
    
    for split in counts.keys():
        for category in counts[split].keys():
            path = os.path.join(directory, split, category)
            if os.path.exists(path):
                counts[split][category] = len(os.listdir(path))
    
    return counts

def split_data_balanced(cancer_type):
    raw_data_path = os.path.join(BASE_DIR, cancer_type, "raw_data")
    split_data_path = os.path.join(BASE_DIR, cancer_type, "split_data")

    if not os.path.exists(raw_data_path):
        print(f"❌ Error: Raw data directory not found: {raw_data_path}")
        return  

    # Create directories for train, val, test splits
    for split in ["train", "val", "test"]:
        for category in ["benign", "malignant"]:
            os.makedirs(os.path.join(split_data_path, split, category), exist_ok=True)

    print(f"📂 Balancing and Splitting Data for: {cancer_type}")

    category_images = {}

    # Collect images for both categories
    for category in ["benign", "malignant"]:
        category_path = os.path.join(raw_data_path, category)
        
        if not os.path.exists(category_path):
            print(f"⚠️ Skipping {category} - folder not found")
            continue

        images = os.listdir(category_path)
        random.shuffle(images)
        category_images[category] = images

    # Ensure equal number of benign and malignant images
    min_samples = min(len(category_images.get("benign", [])), len(category_images.get("malignant", [])))

    if min_samples == 0:
        print(f"❌ Error: Not enough data to balance {cancer_type}. Skipping...")
        return

    # Define exact numbers for strict balancing
    num_train = int(min_samples * SPLIT_RATIO["train"])
    num_val = int(min_samples * SPLIT_RATIO["val"])
    num_test = min_samples - (num_train + num_val)  # Ensure the sum is correct

    for category in ["benign", "malignant"]:
        images = category_images[category][:min_samples]  # Limit to min_samples for perfect balance
        random.shuffle(images)

        for idx, img_name in enumerate(images):
            img_path = os.path.join(raw_data_path, category, img_name)

            if idx < num_train:
                dest = os.path.join(split_data_path, "train", category, img_name)
            elif idx < num_train + num_val:
                dest = os.path.join(split_data_path, "val", category, img_name)
            else:
                dest = os.path.join(split_data_path, "test", category, img_name)

            shutil.copy(img_path, dest)

    print(f"✅ {cancer_type} Data Balanced and Split Completed!")

    # Count images in split_data and verify balance
    split_counts = count_images(split_data_path)

    print("\n📊 Data Distribution After Splitting:")
    for split, counts in split_counts.items():
        print(f"   🔹 {split.upper()}: Benign = {counts['benign']}, Malignant = {counts['malignant']}")

    if (split_counts["train"]["benign"] == split_counts["train"]["malignant"] and
        split_counts["val"]["benign"] == split_counts["val"]["malignant"] and
        split_counts["test"]["benign"] == split_counts["test"]["malignant"]):
        print("✅ Dataset is **perfectly balanced** across all splits! 🎯")
    else:
        print("⚠️ Warning: Data is still slightly imbalanced.")

# Run the splitting process
for cancer in CANCER_TYPES:
    split_data_balanced(cancer)

print("\n🎯 ✅ Data Splitting Completed Successfully!")
print("🔹 Balanced images are split into 'train', 'val', and 'test' in 'split_data' folder.")
