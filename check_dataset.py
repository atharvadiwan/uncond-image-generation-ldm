from datasets import Dataset
from PIL import Image
import numpy as np
import glob

# Build data list
# image_paths = sorted(glob.glob("/workspace/data/ASV_diffusion_data/dataset/train/images/*.png"))
# mask_paths = [p.replace("images", "masks").replace(".png", ".npy") for p in image_paths]

# dataset = Dataset.from_dict({
#     "image": image_paths,
#     "mask": mask_paths
# })

# # Load example
# example = dataset[0]
# img = Image.open(example["image"])
# mask = np.load(example["mask"])  # shape: (C, H, W)
# print(f"Image shape: {img.size}, Mask shape: {mask.shape}")
import os

dataset_root = "/workspace/data/ASV_data/RD/val/"
all_folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]

print(all_folders)
exit(0)

import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# === Inputs ===
# image_path = Path("/workspace/data/ASV_diffusion_data/dataset/train/images/2PM_test_01_front_03_0003297.png")
image_path = Path("/workspace/data/ASV_diffusion_data/dataset/train/images/10AM_test_03_front_01_0002525.png")
mask_path = image_path.with_name(image_path.name.replace(".png", ".npy")).as_posix().replace("images", "masks")

# === Load image and mask ===
image = np.array(Image.open(image_path).convert("RGB"))
mask = np.load(mask_path)  # shape: (C, H, W)

# === Generate color mask ===
colors = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (255, 128, 0),   # Orange
    (128, 0, 255),   # Purple
    (128, 128, 128), # Gray
]
num_classes = mask.shape[0] - 1
colors = colors[:num_classes]

# Create RGB mask visualization
H, W = mask.shape[1:]
mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)

for i in range(num_classes):
    mask_i = mask[i]
    for c in range(3):
        mask_rgb[..., c] += (mask_i * colors[i][c]).astype(np.uint8)

# === Blend and save ===
alpha = 0.5
blended = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
output_path = "./temp.png"

# Create output dir if needed
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

# Save using OpenCV (BGR)
cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

print(f"✅ Saved blended visualization to {output_path}")
