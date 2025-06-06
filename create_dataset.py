import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# Paths
source_root = Path("/workspace/data/ASV_data/RD/val/")
output_images = Path("/workspace/data/ASV_diffusion_data/dataset/train/images")
output_masks = Path("/workspace/data/ASV_diffusion_data/dataset/train/masks")

# training_folders = ['10AM_test_03_front_01', '10AM_test_03_rear_02', '10AM_test_03_rear_03', '10AM_test_04_front_00', '10AM_test_04_front_01', 
#  '10AM_test_04_front_03', '11AM_test_05_front_00', '11AM_test_05_front_01', '11AM_test_05_front_02', '11AM_test_05_front_04', 
#  '11AM_test_05_front_05', '11AM_test_05_front_06', '11AM_test_05_front_07', '11AM_test_05_front_09', '11AM_test_05_rear_01', 
#  '11AM_test_05_rear_02', '11AM_test_05_rear_03', '11AM_test_05_rear_04', '11AM_test_05_rear_05', '11AM_test_05_rear_06', 
#  '2PM_test_01_front_03', '2PM_test_01_front_05', '2PM_test_02_front_05', '4PM_test_06_front_02', '9AM_test_02_front_01', 
#  '9AM_test_02_front_04', 'Gopal_autoannotate', 'aaron_001_train', 'atharva_001_train', 'gopal_0001_manual_train', 
#  'kawabuchi_001_manual_train', 'kim_clean_train', 'bhuvan_001_clean_train', 'takayama_001_clean_train', 'takayama_002_clean_train', 
#  'watanabe_001_clean_train', 'watanabe_002_clean_train', '10AM_test_03_front_02', '10AM_test_03_rear_01', '2PM_test_02_rear_04', 
#  '3PM_test_03_rear_04', '9AM_test_02_rear_04', 'ichigo_front_center', 'ichigo_rear_center']
# Configuration
training_folders = [
    # "2PM_test_01_front_03",
    "Gopal_autoannotate",
    # "folder_name_2",
    # Add more
]
selected_classes = ["curb", "curb_side_walk", "lane", "side_walk", "road", "Uturn_area_lane", "Uturn_area"]
# Each group is a list of class folders to be merged into one mask channel
selected_class_groups = [
    ["curb", "lane", "Uturn_area_lane", "curb_side_walk"],             # merged as one
    ["side_walk", "road", "Uturn_area"],    # merged as one
    # ["tree", "bush"]                  # merged as one
]

# Create output directories
output_images.mkdir(parents=True, exist_ok=True)
output_masks.mkdir(parents=True, exist_ok=True)

image_count = 0
mask_count = 0

# Count total files for progress bar
total_images = sum(1 for f in training_folders for _ in (source_root / f / "frames").glob("*.png"))

# Process with tqdm progress bar
with tqdm(total=total_images, desc="Processing images") as pbar:
    for folder_name in training_folders:
        frame_dir = source_root / folder_name / "frames"
        mask_dir = source_root / folder_name / "masks"

        if not frame_dir.exists():
            print(f"❌ Warning: {frame_dir} missing.")
            continue

        for img_file in frame_dir.glob("*.png"):
            # Copy image
            new_img_name = f"{folder_name}_{img_file.name}"
            target_img_path = output_images / new_img_name
            shutil.copy(img_file, target_img_path)
            image_count += 1

            # Prepare multi-channel mask from class groups
            sample_mask = None
            mask_stack = []

            for group in selected_class_groups:
                group_mask = None
                for class_name in group:
                    class_mask_path = mask_dir / class_name / img_file.name
                    if class_mask_path.exists():
                        mask = Image.open(class_mask_path).convert("L")
                        mask_array = np.array(mask, dtype=np.uint8)

                        if sample_mask is None:
                            sample_mask = mask_array
                        else:
                            assert mask_array.shape == sample_mask.shape, f"Shape mismatch: {class_mask_path}"

                        binary = (mask_array > 0).astype(np.uint8)
                        if group_mask is None:
                            group_mask = binary
                        else:
                            group_mask = np.logical_or(group_mask, binary).astype(np.uint8)

                if group_mask is None:
                    # No class in group found for this image
                    if sample_mask is not None:
                        group_mask = np.zeros_like(sample_mask, dtype=np.uint8)
                    else:
                        group_mask = np.zeros_like(Image.open(img_file).convert("L"), dtype=np.uint8)

                mask_stack.append(group_mask)

            if not mask_stack:
                print(f"⚠️  No masks found for {img_file.name}")
                pbar.update(1)
                continue

            # Stack masks and add background
            mask_array = np.stack(mask_stack, axis=0)  # shape: (C, H, W)
            background = (np.sum(mask_array, axis=0) == 0).astype(np.uint8)  # (H, W)
            full_mask = np.concatenate([mask_array, background[None, ...]], axis=0)  # shape: (C+1, H, W)

            # Save
            new_mask_name = new_img_name.replace(".png", ".npy")
            np.save(output_masks / new_mask_name, full_mask)
            mask_count += 1

            pbar.update(1)

print(f"✅ Copied {image_count} images to {output_images}")
print(f"✅ Saved {mask_count} masks with {len(selected_class_groups)} classes + background")
