import os
import re
import json
import csv
from urllib.parse import urlparse

# === Paths ===
json_path = r"path-to-JSON-file.json"
mask_folder = r"path-to-mask-directory"
output_file = r"path-to-output-csv.csv"


# ------------------------------------------------------------
# LOAD JSON (supports both list and {"tasks": [...]})
# ------------------------------------------------------------
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, list):
    tasks = data
elif isinstance(data, dict) and "tasks" in data:
    tasks = data["tasks"]
else:
    raise ValueError("JSON structure unsupported: expected list or dict with 'tasks'.")

print(f"[DEBUG] Loaded {len(tasks)} tasks")


# ------------------------------------------------------------
# BUILD task_id → clean original image name
# ------------------------------------------------------------
id_to_image = {}

for task in tasks:
    task_id = task.get("id")
    if task_id is None:
        continue

    # Prefer the image reference inside task["data"]
    image_ref = task.get("data", {}).get("image") or task.get("file_upload") or ""
    if not image_ref:
        continue

    # Extract the filename from URL or /data/upload/... path
    path_part = urlparse(image_ref).path if "://" in image_ref else image_ref
    base = os.path.basename(path_part)  # e.g. ab474137-Pass_r1-n_39_1_2017-02-18.JPG

    stem, ext = os.path.splitext(base)

    # Only strip prefix if it looks like Label Studio's random/hash prefix (e.g. ab474137-...)
    m = re.match(r"^[0-9a-fA-F]{6,}-", stem)
    if m:
        clean_stem = stem.split("-", 1)[1]
    else:
        clean_stem = stem

    clean_name = clean_stem + ext

    id_to_image[str(task_id)] = clean_name

print(f"[DEBUG] Built mapping for {len(id_to_image)} tasks with images")


# ------------------------------------------------------------
# COLLECT MASK FILES
# (Your masks are named: mask_551.png etc.)
# ------------------------------------------------------------
if not os.path.isdir(mask_folder):
    raise FileNotFoundError(f"Mask folder not found: {mask_folder}")

mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(".png")]
mask_files.sort()

print(f"[DEBUG] Found {len(mask_files)} mask PNGs")
print("[DEBUG] Example masks:", mask_files[:5])


# ------------------------------------------------------------
# WRITE CSV
# mask_551.png → task_id 551 → real image name
# ------------------------------------------------------------
rows_written = 0

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_name", "mask_name", "class_name"])

    for mask_file in mask_files:
        # Match mask_<taskID>.png
        match = re.match(r"mask_(\d+)\.png$", mask_file, flags=re.IGNORECASE)
        if not match:
            continue

        task_id = match.group(1)
        image_name = id_to_image.get(task_id, "Unknown")

        # Assign a simple class label since these are full masks
        class_name = "mask"

        writer.writerow([image_name, mask_file, class_name])
        rows_written += 1

print(f"[DEBUG] Wrote {rows_written} rows to {output_file}")
if rows_written == 0:
    print("[DEBUG] WARNING: No rows written. Likely no mask files matched the pattern 'mask_<id>.png'.")
