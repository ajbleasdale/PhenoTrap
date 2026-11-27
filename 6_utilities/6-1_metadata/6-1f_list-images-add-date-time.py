
# Script to create a CSV listing image filenames and their corresponding acquisition date and time.
# Dates are extracted from EXIF metadata when available, otherwise the file modified time is used.
#
# Note: Dates are formatted as "%Y&%m&%d" to prevent Excel from automatically
# converting values to date format. After constructing filenames, you can replace
# "&" with "-" if needed.

import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from PIL import Image, ExifTags

# ==========================
# SETTINGS
# ==========================
IMAGE_FOLDER = r"path-to-image-directory" # folder containing the images
OUT_CSV = r"path-to-output-csv.csv"   # output CSV file
FILENAME_COLUMN = "Filename"
DATE_COLUMN = "date"
TIME_COLUMN = "time"
PREFER_EXIF = True  # True: use EXIF capture date if available

# ==========================
# EXIF SETUP
# ==========================
EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
TAG_DATETIME_ORIGINAL = EXIF_TAGS.get("DateTimeOriginal")
TAG_DATETIME = EXIF_TAGS.get("DateTime")

def get_capture_datetime(img_path: Path):
    """
    Returns (date, time) as:
    - date: YYYY&MM&DD
    - time: HHMMSS
    Uses EXIF when available, otherwise file modified time.
    """
    # 1) EXIF capture datetime (preferred)
    if PREFER_EXIF:
        try:
            with Image.open(img_path) as im:
                exif = im.getexif()
                if exif:
                    dt_str = None
                    if TAG_DATETIME_ORIGINAL and TAG_DATETIME_ORIGINAL in exif:
                        dt_str = exif.get(TAG_DATETIME_ORIGINAL)
                    elif TAG_DATETIME and TAG_DATETIME in exif:
                        dt_str = exif.get(TAG_DATETIME)

                    if isinstance(dt_str, str) and dt_str.strip():
                        dt = datetime.strptime(dt_str.strip(), "%Y:%m:%d %H:%M:%S")
                        return dt.strftime("%Y&%m&%d"), dt.strftime("%H%M%S")
        except Exception:
            pass  # fall back

    # 2) Filesystem modified time (fallback)
    try:
        ts = os.path.getmtime(img_path)
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y&%m&%d"), dt.strftime("%H%M%S")
    except Exception:
        return None, None

# ==========================
# MAIN
# ==========================
img_dir = Path(IMAGE_FOLDER)

if not img_dir.exists():
    raise FileNotFoundError(f"Image folder not found: {img_dir}")

valid_ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
image_files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext])

rows = []
for img_path in image_files:
    date_val, time_val = get_capture_datetime(img_path)

    rows.append({
        FILENAME_COLUMN: img_path.name,
        DATE_COLUMN: date_val,
        TIME_COLUMN: time_val
    })

df = pd.DataFrame(rows)

# Save as CSV
out_path = Path(OUT_CSV)
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Saved CSV: {out_path}")
print(f"Files processed: {len(df)}")
