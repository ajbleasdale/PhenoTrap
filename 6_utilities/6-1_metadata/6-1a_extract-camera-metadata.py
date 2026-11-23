
from PIL import Image, ExifTags
import os
import re
import pandas as pd
from collections import defaultdict


# =========================
# SETTINGS
# =========================
input_folder = r"path-to-camera-trap-dataset"
output_xlsx = r"path-to-metadata.xlsx"
recursive = True
valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# =========================
# EXIF TAG SETUP
# =========================
EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
make_tag = EXIF_TAGS.get("Make")
model_tag = EXIF_TAGS.get("Model")

# =========================
# CLEAN TEXT FOR EXCEL
# =========================
def clean_excel_string(value):
    if value is None:
        return "Unknown"

    value = str(value)

    # Remove illegal Excel/XML control characters
    value = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", value)

    # Collapse repeated whitespace and strip ends
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else "Unknown"

# =========================
# GET IMAGE FILES
# =========================
def get_image_files(folder, recursive=True):
    image_files = []

    if recursive:
        for root, _, files in os.walk(folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if os.path.isfile(path) and os.path.splitext(file)[1].lower() in valid_extensions:
                image_files.append(path)

    return image_files

# =========================
# EXTRACT METADATA
# =========================
def extract_metadata(image_path):
    make = "Unknown"
    model = "Unknown"
    width = None
    height = None

    try:
        with Image.open(image_path) as img:
            width, height = img.size

            try:
                exif = img.getexif()
            except Exception:
                exif = None

            if exif:
                if make_tag in exif and exif[make_tag]:
                    make = clean_excel_string(exif[make_tag])

                if model_tag in exif and exif[model_tag]:
                    model = clean_excel_string(exif[model_tag])

    except Exception as e:
        print(f"Skipping {image_path} | Error: {e}")

    return make, model, width, height

# =========================
# MAIN
# =========================
image_files = get_image_files(input_folder, recursive)
print(f"Found {len(image_files)} images")

summary = defaultdict(int)

for i, image_path in enumerate(image_files, 1):
    make, model, width, height = extract_metadata(image_path)

    if width is None or height is None:
        continue

    key = (make, model, width, height)
    summary[key] += 1

    if i % 100 == 0 or i == len(image_files):
        print(f"Processed {i}/{len(image_files)}")

# =========================
# SAVE TO EXCEL
# =========================
rows = []
for (make, model, width, height), count in summary.items():
    rows.append({
        "camera_make": clean_excel_string(make),
        "camera_model": clean_excel_string(model),
        "width_px": width,
        "height_px": height,
        "image_count": count,
    })

df = pd.DataFrame(rows)

if not df.empty:
    df = df.sort_values(
        by=["camera_make", "camera_model", "width_px", "height_px"],
        ascending=True
    ).reset_index(drop=True)

df.to_excel(output_xlsx, index=False)

print(f"Done. Saved to:\n{output_xlsx}")
