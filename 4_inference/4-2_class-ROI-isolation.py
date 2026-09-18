from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

# ======================
# Paths
# ======================
images_dir = Path(r"path-to-image-directory")
masks_dir = Path(r"path-to-inference-masks")
output_dir = Path(r"path-to-output-directory")

output_images_dir = output_dir / "images"
output_images_dir.mkdir(parents=True, exist_ok=True)

csv_output = output_dir / "class_cutout_summary.csv"

# ======================
# Settings
# ======================
image_extensions = [".jpg", ".JPG", ".jpeg", ".png", ".tif", ".tiff"]

# Set this to True if you want to skip background class
skip_background = True
background_class_id = 0

# Your class names
class_names = {

    0: "background",  # 0
    1: "sky",  # 1
    2: "snow_ice",  # 2
    3: "bare",  # 3
    4: "ericaceuous_shrub",  # 4
    5: "briar_seedling",  # 5
    6: "graminoid",  # 6
    7: "forb",  # 7
    8: "fern",  # 8
    9: "cryptogam",  # 9
    10: "vine",  # 10
    11: "broadleaf_evergreen",  # 11
    12: "conifer",  # 12
    13: "broadleaf_deciduous",  # 13

}


# ======================
# Helper function
# ======================
def find_image_for_mask(mask_path, images_dir):
    """
    Finds an image with the same stem as the mask.
    Example:
    mask:  image001.png
    image: image001.jpg / image001.png / etc.
    """
    for ext in image_extensions:
        candidate = images_dir / f"{mask_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ======================
# Main processing
# ======================
records = []

mask_files = sorted(masks_dir.glob("*.png"))

print(f"Found {len(mask_files)} mask files.")

for mask_path in tqdm(mask_files, desc="Processing masks"):
    image_path = find_image_for_mask(mask_path, images_dir)

    if image_path is None:
        print(f"Skipping: no matching image found for {mask_path.name}")
        continue

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)

    image_np = np.array(image)

    # Convert mask to class ID array
    # This assumes your mask is single-channel where pixel values are class IDs.
    mask_np = np.array(mask)

    if mask_np.ndim == 3:
        # If mask is RGB, this will not work correctly unless converted first.
        raise ValueError(
            f"{mask_path.name} appears to be RGB. "
            "This script expects masks where pixel values are class IDs."
        )

    if image_np.shape[:2] != mask_np.shape[:2]:
        raise ValueError(
            f"Size mismatch for {image_path.name}: "
            f"image shape {image_np.shape[:2]}, mask shape {mask_np.shape[:2]}"
        )

    total_pixels = mask_np.size
    present_class_ids = sorted(np.unique(mask_np))

    for class_id in present_class_ids:
        class_id = int(class_id)

        if skip_background and class_id == background_class_id:
            continue

        class_mask = mask_np == class_id
        class_pixel_count = int(class_mask.sum())

        if class_pixel_count == 0:
            continue

        class_percentage = (class_pixel_count / total_pixels) * 100

        cutout_np = np.zeros_like(image_np)
        cutout_np[class_mask] = image_np[class_mask]

        class_name = class_names.get(class_id, f"class_{class_id}")

        output_filename = f"{image_path.stem}_class_{class_id}.png"
        output_path = output_images_dir / output_filename

        Image.fromarray(cutout_np).save(output_path)

        records.append({
            "image_filename": image_path.name,
            "mask_filename": mask_path.name,
            "output_filename": output_filename,
            "class_id": class_id,
            "class_name": class_name,
            "class_pixel_count": class_pixel_count,
            "class_pixel_percentage": class_percentage,
            "total_pixels": total_pixels,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "output_path": str(output_path),
        })

# ======================
# Save CSV
# ======================
df = pd.DataFrame(records)
df = df.sort_values(["image_filename", "class_id"])

df.to_csv(csv_output, index=False)

print("Done.")
print(f"Saved cut-out images to: {output_images_dir}")
print(f"Saved CSV to: {csv_output}")
