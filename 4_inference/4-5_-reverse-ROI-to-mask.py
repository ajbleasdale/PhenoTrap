# For use if Single-channel masks are deleted or corrupted

from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import re
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

# Folder created by your organisation script:
# organised_by_class/
#     class_1_sky/
#     class_2_snow_ice/
#     ...
organised_dir = Path(
    r"path-to/organised_by_class"
)

# Reconstructed class-ID masks will be saved here
output_dir = organised_dir.parent / "reconstructed_masks"
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS NAMES
# ============================================================

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


# ============================================================
# SETTINGS
# ============================================================

image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Safety setting:
# If True, the script will NOT overwrite an existing mask.
skip_existing = True


# ============================================================
# CHECK INPUT
# ============================================================

print(f"Looking in: {organised_dir}")
print(f"Folder exists: {organised_dir.exists()}")

if not organised_dir.exists():
    raise FileNotFoundError(
        f"Organised ROI folder does not exist:\n{organised_dir}"
    )


# ============================================================
# FIND ALL CLASS CUT-OUTS
# ============================================================

# Structure:
#
# image_groups["original_image_name"] =
#       [(class_id, cutout_path), ...]
#
image_groups = defaultdict(list)

class_folders = [
    p for p in organised_dir.iterdir()
    if p.is_dir()
]

print(f"Found {len(class_folders)} class folders.")

for class_folder in class_folders:

    # Example:
    # class_6_graminoid
    folder_match = re.match(r"class_(\d+)_", class_folder.name)

    if folder_match is None:
        print(f"Skipping unrecognised folder: {class_folder.name}")
        continue

    class_id = int(folder_match.group(1))

    for file_path in class_folder.iterdir():

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in image_extensions:
            continue

        # ----------------------------------------------------
        # Extract original image name
        #
        # Example:
        # ABC123_class_6.png
        #
        # becomes:
        # ABC123
        # ----------------------------------------------------

        match = re.match(
            r"(.+)_class_(\d+)$",
            file_path.stem
        )

        if match is None:
            print(f"Could not interpret filename: {file_path.name}")
            continue

        original_stem = match.group(1)
        filename_class_id = int(match.group(2))

        # Safety check
        if filename_class_id != class_id:
            print(
                f"WARNING: class mismatch for {file_path.name}: "
                f"folder says {class_id}, "
                f"filename says {filename_class_id}"
            )

        image_groups[original_stem].append(
            (filename_class_id, file_path)
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print(f"Found {len(image_groups):,} unique original images.")

if len(image_groups) == 0:
    raise RuntimeError(
        "No cut-out images could be grouped. "
        "Check the directory and filename structure."
    )


# ============================================================
# RECONSTRUCT MASKS
# ============================================================

created = 0
skipped = 0
warnings = 0
corrupted_files = []

for original_stem, cutouts in tqdm(
    image_groups.items(),
    desc="Reconstructing masks"
):

    output_path = output_dir / f"{original_stem}.png"

    # --------------------------------------------------------
    # Skip masks already successfully reconstructed
    # --------------------------------------------------------
    if skip_existing and output_path.exists():
        skipped += 1
        continue

    # --------------------------------------------------------
    # Find the first readable cut-out so we can determine
    # the image dimensions
    # --------------------------------------------------------
    first_np = None

    for first_class_id, first_path in cutouts:
        try:
            with Image.open(first_path) as img:
                first_np = np.array(img.convert("RGB"))
            break

        except Exception as e:
            print(f"\nCORRUPTED/UNREADABLE - skipping: {first_path}")
            print(f"Reason: {e}")

            corrupted_files.append({
                "image": original_stem,
                "class_id": first_class_id,
                "file": str(first_path),
                "error": str(e)
            })

    # If NONE of the cut-outs for this image can be read,
    # skip the entire image
    if first_np is None:
        print(
            f"\nWARNING: No readable cut-outs for {original_stem}. "
            f"Skipping entire mask."
        )
        continue

    height, width = first_np.shape[:2]

    # --------------------------------------------------------
    # Create empty reconstructed mask
    # --------------------------------------------------------
    reconstructed_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    assigned_pixels = np.zeros(
        (height, width),
        dtype=bool
    )

    # --------------------------------------------------------
    # Add each class
    # --------------------------------------------------------
    for class_id, cutout_path in cutouts:

        # ----------------------------------------------------
        # Try opening the cut-out.
        # If corrupted, record it and continue.
        # ----------------------------------------------------
        try:
            with Image.open(cutout_path) as img:
                cutout_np = np.array(img.convert("RGB"))

        except Exception as e:

            print(
                f"\nCORRUPTED/UNREADABLE - skipping: "
                f"{cutout_path}"
            )
            print(f"Reason: {e}")

            corrupted_files.append({
                "image": original_stem,
                "class_id": class_id,
                "file": str(cutout_path),
                "error": str(e)
            })

            # IMPORTANT:
            # Continue to the next class instead of
            # terminating the entire script.
            continue

        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------
        if cutout_np.shape[:2] != (height, width):

            print(
                f"\nSIZE MISMATCH - skipping: {cutout_path.name}"
            )
            print(
                f"Expected: {(height, width)}, "
                f"found: {cutout_np.shape[:2]}"
            )

            corrupted_files.append({
                "image": original_stem,
                "class_id": class_id,
                "file": str(cutout_path),
                "error": (
                    f"Size mismatch: expected {(height, width)}, "
                    f"found {cutout_np.shape[:2]}"
                )
            })

            continue

        # ----------------------------------------------------
        # Recover class pixels
        # ----------------------------------------------------
        class_pixels = np.any(
            cutout_np != 0,
            axis=2
        )

        # ----------------------------------------------------
        # Check for overlaps
        # ----------------------------------------------------
        overlap = assigned_pixels & class_pixels

        if np.any(overlap):

            overlap_count = int(overlap.sum())

            print(
                f"\nWARNING: {original_stem} has "
                f"{overlap_count:,} overlapping pixels "
                f"for class {class_id}."
            )

            warnings += 1

        # ----------------------------------------------------
        # Assign class
        # ----------------------------------------------------
        reconstructed_mask[class_pixels] = class_id
        assigned_pixels[class_pixels] = True

    # --------------------------------------------------------
    # SAVE RECONSTRUCTED MASK
    # --------------------------------------------------------
    Image.fromarray(reconstructed_mask).save(output_path)

    created += 1


# ============================================================
# SAVE LIST OF CORRUPTED FILES
# ============================================================

if corrupted_files:

    corrupted_df = pd.DataFrame(corrupted_files)

    corrupted_csv = output_dir / "corrupted_cutouts.csv"

    corrupted_df.to_csv(
        corrupted_csv,
        index=False
    )

    print()
    print(f"Corrupted/unreadable files: {len(corrupted_files):,}")
    print(f"Corrupted file list saved to:")
    print(corrupted_csv)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("RECONSTRUCTION COMPLETE")
print("========================================")

print(f"Original images identified: {len(image_groups):,}")
print(f"New masks created:          {created:,}")
print(f"Existing masks skipped:     {skipped:,}")
print(f"Corrupted files skipped:    {len(corrupted_files):,}")
print(f"Overlap warnings:           {warnings:,}")

print()
print("Reconstructed masks saved to:")
print(output_dir)
