#for use if specific imagery is to be excluded

import os
import random
import shutil
from pathlib import Path

# ================== CONFIGURATION ==================

# Path to the folder containing the images to sample from
SOURCE_DIR = Path(r"path-to-source-directory")

# Path to the folder where the random selection will be copied
DEST_DIR = Path(r"path-to-destination-directory")

# Path to the folder whose images you want to exclude (by filename)
EXCLUSION_DIR = Path(r"path-to-directory-containing-exclusion-imagery")

# Number of images to copy
NUM_IMAGES_TO_COPY = 8000

# Set to True to skip images that already exist (same filename) in EXCLUSION_DIR
SKIP_IMAGES_IN_EXCLUSION_FOLDER = True

# Image extensions to consider
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}

# ===================================================


def collect_images(folder: Path):
    """Return a list of image Paths in the given folder (non-recursive)."""
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():
    if not SOURCE_DIR.is_dir():
        raise NotADirectoryError(f"SOURCE_DIR does not exist: {SOURCE_DIR}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # All candidate images from source
    source_images = collect_images(SOURCE_DIR)

    # If switch is on, build a set of filenames in exclusion folder
    if SKIP_IMAGES_IN_EXCLUSION_FOLDER:
        if not EXCLUSION_DIR.is_dir():
            print(f"Warning: EXCLUSION_DIR does not exist: {EXCLUSION_DIR}")
            excluded_filenames = set()
        else:
            excluded_filenames = {p.name for p in collect_images(EXCLUSION_DIR)}
        # Filter out images whose filenames are already in the exclusion folder
        source_images = [p for p in source_images if p.name not in excluded_filenames]

    if not source_images:
        print("No images available to sample after applying filters.")
        return

    # Decide how many to sample
    k = min(NUM_IMAGES_TO_COPY, len(source_images))

    # Randomly pick k images
    selected = random.sample(source_images, k=k)

    # Copy them to DEST_DIR
    copied_count = 0
    for src_path in selected:
        dest_path = DEST_DIR / src_path.name

        # Optional: skip if already exists in DEST_DIR to avoid overwriting
        if dest_path.exists():
            print(f"Skipping (already in destination): {dest_path.name}")
            continue

        shutil.copy2(src_path, dest_path)
        copied_count += 1

    print(f"Requested: {NUM_IMAGES_TO_COPY}")
    print(f"Available to sample from: {len(source_images)}")
    print(f"Copied: {copied_count} images to {DEST_DIR}")


if __name__ == "__main__":
    main()
