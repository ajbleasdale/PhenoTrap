# Find which subfolder each image from a CSV is located in, then write that folder
# into a new column in the SAME CSV file.
# -----------------------------------------------------------------------------
# - Reads image names from a specified CSV column
# - Recursively searches through a root image directory
# - Matches by filename only
# - Adds/updates a new column with the folder name
# - Leaves blank if file is not found
# - Saves changes back to the original CSV
#
# Required:
# pip install pandas

import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
CSV_PATH = r"path-to-results.csv"     # Path to your CSV
IMAGE_ROOT_FOLDER = r"path-to-image-directory"  # Root folder containing all image folders
IMAGE_NAME_COLUMN = "image_name"                # Column containing image filenames
OUTPUT_COLUMN = "folder_name"                   # New column to store located folder
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)

if IMAGE_NAME_COLUMN not in df.columns:
    raise ValueError(f"Column '{IMAGE_NAME_COLUMN}' not found in CSV.")

# =========================
# INDEX ALL IMAGES IN ROOT
# =========================
print("Scanning image folders...")

image_index = {}

for file_path in Path(IMAGE_ROOT_FOLDER).rglob("*"):
    if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
        image_index[file_path.name] = file_path.parent.name  # Store folder name only

print(f"Indexed {len(image_index)} image files.")

# =========================
# MATCH CSV ENTRIES
# =========================
folder_results = []

for image_name in df[IMAGE_NAME_COLUMN]:
    if pd.isna(image_name):
        folder_results.append("")
        continue

    image_name = str(image_name).strip()
    folder_name = image_index.get(image_name, "")
    folder_results.append(folder_name)

# =========================
# WRITE RESULTS
# =========================
df[OUTPUT_COLUMN] = folder_results

# Save back to same CSV
df.to_csv(CSV_PATH, index=False)

# =========================
# SUMMARY
# =========================
found_count = sum(1 for x in folder_results if x)
missing_count = len(folder_results) - found_count

print("Done.")
print(f"Found: {found_count}")
print(f"Missing: {missing_count}")
print(f"Updated CSV saved to: {CSV_PATH}")
