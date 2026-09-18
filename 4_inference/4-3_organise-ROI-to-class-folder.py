rom pathlib import Path
import shutil
import re
from tqdm import tqdm

# ======================
# Paths
# ======================
cutouts_dir = Path(
    r"path-to-ROI-directory")

organised_dir = cutouts_dir.parent / "organised_by_class"
organised_dir.mkdir(parents=True, exist_ok=True)

# ======================
# Class names
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


# ======================
# Settings
# ======================
image_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# ======================
# Check input folder
# ======================
print(f"Looking in: {cutouts_dir}")
print(f"Folder exists: {cutouts_dir.exists()}")

if not cutouts_dir.exists():
    raise FileNotFoundError(f"Folder does not exist: {cutouts_dir}")

# ======================
# Find cut-out files
# ======================
cutout_files = [
    p for p in cutouts_dir.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]

print(f"Found {len(cutout_files):,} cut-out images.")

if len(cutout_files) == 0:
    print("No images found. Check that cutouts_dir points to the correct folder.")
    raise SystemExit

# ======================
# Move files into class folders
# ======================
moved_count = 0
skipped_count = 0

for file_path in tqdm(cutout_files, desc="Moving files by class"):

    stem = file_path.stem

    match = re.search(r"_class_(\d+)$", stem)

    if match is None:
        print(f"Skipping, no class ID found: {file_path.name}")
        skipped_count += 1
        continue

    class_id = int(match.group(1))
    class_name = class_names.get(class_id, f"class_{class_id}")

    class_folder = organised_dir / f"class_{class_id}_{class_name}"
    class_folder.mkdir(parents=True, exist_ok=True)

    destination = class_folder / file_path.name

    if destination.exists():
        print(f"Skipping, already exists: {destination.name}")
        skipped_count += 1
        continue

    shutil.move(str(file_path), str(destination))
    moved_count += 1

# ======================
# Final summary
# ======================
print("\nDone.")
print(f"Moved files: {moved_count:,}")
print(f"Skipped files: {skipped_count:,}")
print(f"Organised files saved to: {organised_dir}")
