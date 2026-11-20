from pathlib import Path
import shutil
import csv

from PIL import Image


# ============================================================
# USER SETTINGS
# ============================================================

# Folder containing *_overlay.png files
OVERLAYS_DIR = Path(
    r"path-to-overlay-folder"
)

# Folder containing the original masks, e.g.
# SE-NM_1_2017-03-26_120000.png
MASKS_SOURCE_DIR = Path(
    r"path-to-mask-directory"
)

# Folder containing the original images, e.g.
# SE-NM_1_2017-03-26_120000.jpg
IMAGES_SOURCE_DIR = Path(
    r"path-to-image-directory"
)


# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

OUTPUT_IMAGES_DIR = Path(
    r"path-to-image-output-folder"
)

OUTPUT_MASKS_DIR = Path(
    r"path-to-mask-output-folder"
)

OUTPUT_CSV = Path(
    r"path-to-output-csv.csv"
)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MASKS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_image(folder, stem):
    """
    Find an image regardless of whether its extension is
    .jpg, .JPG, .jpeg, or .JPEG.
    """
    extensions = [".jpg", ".JPG", ".jpeg", ".JPEG"]

    for ext in extensions:
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path

    return None


def find_mask(folder, stem):
    """
    Find corresponding PNG mask.
    """
    extensions = [".png", ".PNG"]

    for ext in extensions:
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path

    return None


def get_subfolder(stem):
    """
    Extract deployment/subfolder from filename.

    Example:
        SE-NM_1_2017-03-26_120000
        -> SE-NM-1

    Adjust this function if your deployment naming convention differs.
    """

    parts = stem.split("_")

    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"

    return ""


# ============================================================
# FIND OVERLAYS - PREVENT DUPLICATES
# ============================================================

overlays = sorted(
    [
        p for p in OVERLAYS_DIR.iterdir()
        if p.is_file()
        and p.stem.lower().endswith("_overlay")
        and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ],
    key=lambda p: p.name.lower()
)

# Remove duplicate file paths
overlays = list(dict.fromkeys(overlays))

print(f"Found {len(overlays):,} overlays.")

# ============================================================
# PROCESS
# ============================================================

csv_rows = []

copied = 0
missing_image = 0
missing_mask = 0
failed_mask = 0

# Track processed overlays to prevent duplicates
processed_stems = set()

for i, overlay_path in enumerate(overlays, start=1):

    # --------------------------------------------------------
    # Remove "_overlay" suffix
    # --------------------------------------------------------

    overlay_stem = overlay_path.stem

    if not overlay_stem.endswith("_overlay"):
        continue

    base_stem = overlay_stem[:-len("_overlay")]

    # Skip duplicate overlays belonging to the same image
    if base_stem.lower() in processed_stems:
        print(f"[DUPLICATE SKIPPED] {base_stem}")
        continue

    processed_stems.add(base_stem.lower())

    # Example:
    #
    # overlay:
    # SE-NM_1_2017-03-26_120000_overlay.png
    #
    # base_stem:
    # SE-NM_1_2017-03-26_120000


    # --------------------------------------------------------
    # Find corresponding image and mask
    # --------------------------------------------------------

    image_path = find_image(IMAGES_SOURCE_DIR, base_stem)
    mask_path = find_mask(MASKS_SOURCE_DIR, base_stem)


    if image_path is None:
        print(f"[MISSING IMAGE] {base_stem}")
        missing_image += 1
        continue

    if mask_path is None:
        print(f"[MISSING MASK]  {base_stem}")
        missing_mask += 1
        continue


    # --------------------------------------------------------
    # Destination names
    # --------------------------------------------------------

    # Preserve actual image extension
    output_image_name = image_path.name

    # Masks always saved as PNG
    output_mask_name = f"{base_stem}.png"

    output_image_path = OUTPUT_IMAGES_DIR / output_image_name
    output_mask_path = OUTPUT_MASKS_DIR / output_mask_name


    # --------------------------------------------------------
    # Copy original image
    # --------------------------------------------------------

    shutil.copy2(
        image_path,
        output_image_path
    )


    # --------------------------------------------------------
    # Convert mask to single channel
    # --------------------------------------------------------

    try:

        with Image.open(mask_path) as mask:

            # If already single-channel, preserve values.
            #
            # "L" = 8-bit grayscale
            # "P" = palette image
            # "I" = integer image

            if mask.mode == "P":
                # Palette masks can contain class IDs directly.
                # Convert pixel indices to an array-preserving
                # single-channel representation.
                mask = mask.convert("L")

            elif mask.mode not in ("L", "I", "I;16"):
                mask = mask.convert("L")

            mask.save(
                output_mask_path,
                format="PNG"
            )

    except Exception as e:

        print(f"[MASK ERROR] {mask_path}")
        print(f"             {e}")

        # Remove image if its mask failed, so output stays paired
        if output_image_path.exists():
            output_image_path.unlink()

        failed_mask += 1
        continue


    # --------------------------------------------------------
    # Determine subfolder/deployment ID
    # --------------------------------------------------------

    subfolder = get_subfolder(base_stem)


    # --------------------------------------------------------
    # Add CSV row
    # --------------------------------------------------------

    csv_rows.append({
        "image_name": output_image_name,
        "mask_name": output_mask_name,
        "subfolder": subfolder,
    })

    copied += 1

    if i % 100 == 0:
        print(
            f"Processed {i:,}/{len(overlays):,} "
            f"| copied: {copied:,}"
        )


# ============================================================
# WRITE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_name",
            "mask_name",
            "subfolder"
        ]
    )

    writer.writeheader()
    writer.writerows(csv_rows)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINISHED")
print("=" * 60)

print(f"Overlays found:        {len(overlays):,}")
print(f"Pairs copied:          {copied:,}")
print(f"Missing images:        {missing_image:,}")
print(f"Missing masks:         {missing_mask:,}")
print(f"Mask conversion errors:{failed_mask:,}")

print()
print(f"Images: {OUTPUT_IMAGES_DIR}")
print(f"Masks:  {OUTPUT_MASKS_DIR}")
print(f"CSV:    {OUTPUT_CSV}")
