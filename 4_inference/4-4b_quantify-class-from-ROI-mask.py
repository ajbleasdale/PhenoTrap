mport os
import re
import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm


# ======================
# Paths
# ======================
roi_root = (
    r"path-to-ROI-directory"
)

output_csv = (
    r"path-to-output-csv.csv"
)


# ======================
# Class settings
# ======================
class_names = {
    0: "background",
    1: "sky",
    2: "snow_ice",
    3: "bare",
    4: "ericaceous_shrub",
    5: "briar_seedling",
    6: "graminoid",
    7: "forb",
    8: "fern",
    9: "cryptogam",
    10: "vine",
    11: "broadleaf_evergreen",
    12: "conifer",
    13: "broadleaf_deciduous",
}

valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# ======================
# Find class folders
# ======================
class_folders = {}

for folder_name in os.listdir(roi_root):
    folder_path = os.path.join(roi_root, folder_name)

    if not os.path.isdir(folder_path):
        continue

    match = re.match(r"class[_-]?(\d+)", folder_name, flags=re.IGNORECASE)

    if match:
        class_id = int(match.group(1))
        class_folders[class_id] = folder_path


print("Class folders found:")

for class_id, folder_path in sorted(class_folders.items()):
    print(f"Class {class_id}: {folder_path}")


# ======================
# Collect masks by filename
# ======================
image_records = {}

for class_id, folder_path in sorted(class_folders.items()):

    files = [
        filename
        for filename in os.listdir(folder_path)
        if filename.lower().endswith(valid_exts)
    ]

    for filename in files:
        image_records.setdefault(filename, {})
        image_records[filename][class_id] = os.path.join(
            folder_path,
            filename
        )


print(f"\nFound {len(image_records)} unique filenames.")


# ======================
# Calculate pixel counts
# ======================
rows = []

for filename, class_masks in tqdm(
    image_records.items(),
    desc="Reading ROI masks"
):

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    date = date_match.group(1) if date_match else None

    row = {
        "filename": filename,
        "date": date,
    }

    total_class_pixels = 0

    for class_id, class_name in class_names.items():

        mask_path = class_masks.get(class_id)

        if mask_path is None:
            pixel_count = 0
        else:
            try:
                mask = np.array(
                    Image.open(mask_path).convert("L")
                )

                # Count all non-zero pixels as belonging to this class
                pixel_count = int(np.count_nonzero(mask))

            except Exception as e:
                print(f"Could not read {mask_path}: {e}")
                pixel_count = 0

        row[f"{class_name}_pixels"] = pixel_count
        total_class_pixels += pixel_count

    row["total_class_pixels"] = total_class_pixels

    for class_id, class_name in class_names.items():
        pixel_count = row[f"{class_name}_pixels"]

        percentage = (
            pixel_count / total_class_pixels * 100
            if total_class_pixels > 0
            else 0
        )

        row[f"{class_name}_percent"] = percentage

    rows.append(row)


# ======================
# Save CSV
# ======================
results_df = pd.DataFrame(rows)

results_df.to_csv(output_csv, index=False)

print(f"\nSaved results to:\n{output_csv}")
