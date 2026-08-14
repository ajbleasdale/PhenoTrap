# For use on imagery captured between 11:30-12:30 if timelapse not set at noon

from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import shutil
from datetime import datetime, time

# ===== USER SETTINGS =====
SOURCE_DIR = r"root-to-image-directory"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
# ==========================

START_TIME = time(11, 30, 0)
END_TIME = time(12, 30, 0)


def get_exif_datetime(image_path: Path):
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if not exif:
                return None

            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTimeOriginal":
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def ensure_unique(dest: Path) -> Path:
    if not dest.exists():
        return dest

    counter = 1
    while True:
        candidate = dest.with_name(f"{dest.stem}_{counter}{dest.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def main():
    source_path = Path(SOURCE_DIR)
    output_path = source_path.parent / f"{source_path.name}_noon"
    output_path.mkdir(parents=True, exist_ok=True)

    total = 0
    copied = 0

    for file in source_path.rglob("*"):
        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total += 1

        dt = get_exif_datetime(file)
        if dt is None:
            dt = datetime.fromtimestamp(file.stat().st_mtime)

        if START_TIME <= dt.time() <= END_TIME:
            destination = ensure_unique(output_path / file.name)
            shutil.copy2(file, destination)
            copied += 1

        if total % 5000 == 0:
            print(f"Checked: {total} | Copied: {copied}")

    print(f"Finished. Total checked: {total} | Copied: {copied}")


if __name__ == "__main__":
    main()
