
import json
import shutil
from pathlib import Path
import string

# ========= CONFIGURE =========
JSON_PATH = Path(r"path-to-input-json.json")
IMAGES_ROOT = Path(r"path-to-image-directory")   # searched recursively

OUT_PRESENT = Path(r"path-to-output-directory-present")
OUT_ABSENT = Path(r"path-to-output-directory-missin")
# =============================

def collect_json_strings(data) -> list[str]:
    """
    Collect strings from Label Studio export where filenames live.

    We will later just check:  image_stem in any(json_string)
    which is equivalent to manual Ctrl+F.
    """
    collected = []

    # top-level list of tasks
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # in case it is wrapped
        if isinstance(data.get("tasks"), list):
            items = data["tasks"]
        else:
            items = [data]
    else:
        raise ValueError(f"Unexpected JSON top-level type: {type(data)}")

    for item in items:
        if not isinstance(item, dict):
            continue

        # file_upload if present
        fu = item.get("file_upload")
        if isinstance(fu, str):
            collected.append(fu.lower())

        # data.image if present
        img = item.get("data", {}).get("image")
        if isinstance(img, str):
            collected.append(img.lower())

    return collected


def main():
    if not JSON_PATH.is_file():
        raise FileNotFoundError(JSON_PATH)

    OUT_PRESENT.mkdir(parents=True, exist_ok=True)
    OUT_ABSENT.mkdir(parents=True, exist_ok=True)

    # Load JSON
    with JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    json_strings = collect_json_strings(data)
    print(f"Collected {len(json_strings)} JSON strings to search in.")

    # Quick sanity: show a few
    for s in json_strings[:5]:
        print("JSON sample:", s)

    copied_present = 0
    copied_absent = 0

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for img_path in IMAGES_ROOT.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in image_exts:
            continue

        # Basename without extension, lowercased
        stem = img_path.stem.lower()

        # Does this stem appear in any of the JSON strings?
        in_json = any(stem in s for s in json_strings)

        if in_json:
            dest = OUT_PRESENT / img_path.name
            copied_present += 1
        else:
            dest = OUT_ABSENT / img_path.name
            copied_absent += 1

        shutil.copy2(img_path, dest)

    print(f"Copied {copied_present} images found in JSON → {OUT_PRESENT}")
    print(f"Copied {copied_absent} images NOT found in JSON → {OUT_ABSENT}")


if __name__ == "__main__":
    main()
