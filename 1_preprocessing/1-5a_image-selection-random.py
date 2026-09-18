import os
import random
import shutil
from pathlib import Path

# --- SETTINGS ---
source_folder = Path(r"path-to-source-folder")
destination_folder = Path(r"path-to-output-directory")
num_images_to_select = 2500

# Allowed image extensions
image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Create output folder if it doesn't exist
destination_folder.mkdir(parents=True, exist_ok=True)

# Recursively collect all image files from subfolders
all_images = [f for f in source_folder.rglob("*") if f.suffix.lower() in image_exts]

print(f"Found {len(all_images)} images in subfolders.")

# Safety check
if len(all_images) == 0:
    raise ValueError("❌ No images found in the source folder or its subfolders.")

num_to_copy = min(num_images_to_select, len(all_images))
selected_images = random.sample(all_images, num_to_copy)

# Copy selected images
for img in selected_images:
    new_name = f"{img.name}"  # Optional: prefix with subfolder name
    shutil.copy(img, destination_folder / new_name)
    print(f"Copied: {img} → {new_name}")

print(f"✅ Done! Copied {num_images_to_select} random images to: {destination_folder}")
