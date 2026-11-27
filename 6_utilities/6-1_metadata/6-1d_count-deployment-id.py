
import os
import csv
from collections import Counter
from pathlib import Path
import re

# ==========================
# CONFIG
# ==========================
INPUT_DIR = r"path-to-image-directory" # folder containing images (can include subfolders if RECURSIVE=True)
OUTPUT_CSV = r"path-to-output-csv"

RECURSIVE = False  # set True to include images in subfolders

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# How to derive "deployment_id":
# This assumes filenames like: DEPLOYMENTID_YYYY-MM-DD_HHMMSS.JPG
# Example: NM_148_2017-07-15_120000.JPG  -> deployment_id = NM_148
# If your pattern differs, adjust `deployment_id_from_stem()`.


DATE_REGEX = re.compile(r"\d{4}-\d{2}-\d{2}")


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTENSIONS


def deployment_id_from_stem(stem: str) -> str:
    """
    Returns filename stem with trailing _YYYY-MM-DD_HHMMSS removed.
    Examples:
      SE-NM_148_2017-07-15_120000 -> SE-NM_148
      SE-NM_149-2017-07-16-120000 -> SE-NM_149
    """
    m = DATE_REGEX.search(stem)
    if not m:
        return stem
    return stem[:m.start()].rstrip("_-")


def iter_images(root: Path, recursive: bool):
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and is_image_file(p):
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and is_image_file(p):
                yield p


def main():
    root = Path(INPUT_DIR)
    if not root.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {root}")

    counts = Counter()

    for img_path in iter_images(root, RECURSIVE):
        deployment_id = deployment_id_from_stem(img_path.stem)
        counts[deployment_id] += 1

    out_path = Path(OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["deployment_id", "file_count"])
        for deployment_id, n in sorted(counts.items()):
            w.writerow([deployment_id, n])

    print(f"Wrote {len(counts)} deployment IDs to: {out_path}")


if __name__ == "__main__":
    main()
