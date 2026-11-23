
import shutil
from pathlib import Path

# ============================================================
# USER SETTINGS
# ============================================================

# Folder containing your approximately 2,000 original images
IMAGE_FOLDER = Path(
    r"path-to-image-folder"
)

# Main Label Studio folder containing all HITL subfolders
LABELSTUDIO_FOLDER = Path(
    r"path-to-large-folder"
)

# Subfolder where missing images should be copied
HITL_FOLDER = LABELSTUDIO_FOLDER / "HITL-1"

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff",
    ".bmp"
}

# True = preview only, no copying
# False = copy missing images
DRY_RUN = False


# ============================================================
# VALIDATE FOLDERS
# ============================================================

if not IMAGE_FOLDER.is_dir():
    raise FileNotFoundError(
        f"Image folder not found: {IMAGE_FOLDER}"
    )

if not LABELSTUDIO_FOLDER.is_dir():
    raise FileNotFoundError(
        f"Label Studio folder not found: {LABELSTUDIO_FOLDER}"
    )


# ============================================================
# FIND ORIGINAL IMAGES
# ============================================================

original_images = sorted(
    file
    for file in IMAGE_FOLDER.iterdir()
    if file.is_file()
    and file.suffix.lower() in IMAGE_EXTENSIONS
)

print(f"Original images found: {len(original_images)}")


# ============================================================
# FIND ALL IMAGES IN LABEL STUDIO
# INCLUDING ALL SUBFOLDERS
# ============================================================

labelstudio_images = [
    file
    for file in LABELSTUDIO_FOLDER.rglob("*")
    if file.is_file()
    and file.suffix.lower() in IMAGE_EXTENSIONS
]

# Compare using filenames (case insensitive)
existing_names = {
    file.name.lower()
    for file in labelstudio_images
}

print(
    f"Images found across Label Studio: "
    f"{len(labelstudio_images)}"
)


# ============================================================
# IDENTIFY MISSING IMAGES
# ============================================================

missing_images = [
    file
    for file in original_images
    if file.name.lower() not in existing_names
]

print(f"Missing images: {len(missing_images)}")


# ============================================================
# COPY MISSING IMAGES INTO HITL-1
# ============================================================

copied = 0

if not DRY_RUN:
    HITL_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

for image in missing_images:

    destination = HITL_FOLDER / image.name

    if DRY_RUN:
        print(f"[WOULD COPY] {image.name}")
        continue

    if destination.exists():
        print(f"[SKIPPED] {image.name}")
        continue

    shutil.copy2(
        image,
        destination
    )

    copied += 1

    print(f"[COPIED] {image.name}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("HITL IMAGE RECOVERY SUMMARY")
print("=" * 55)

print(f"Original images: {len(original_images)}")
print(f"Label Studio images: {len(labelstudio_images)}")
print(f"Missing images identified: {len(missing_images)}")

if DRY_RUN:
    print(f"Images to copy: {len(missing_images)}")
    print("DRY RUN: No files were copied.")
else:
    print(f"Images copied into HITL-1: {copied}")

print("=" * 55)
