
import os
import json
import csv
import re

# ==========================
# CONFIGURATION (EDIT THIS)
# ==========================

CSV_PATH = r"path-to-metadata-csv.csv"          # columns: old_name,new_name,new_image_id
IMAGES_DIR = r"path-to-image-directory"

JSON_IN_PATH = r"path-to-input-JSON.JSON"
JSON_OUT_PATH = r"path-to-output-JSON.JSON"

DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})")

# Column names in your CSV:
CSV_OLD_COL = "file_name"   # old name or old stem; adjust if your CSV uses a different column
CSV_NEW_COL = "new_name"    # new filename to write into JSON

DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})")

# ==========================
# HELPERS
# ==========================
def split_upload_and_filename(s):
    if not isinstance(s, str):
        return None, None
    base = s.split("/")[-1].split("\\")[-1]
    if "-" not in base:
        return None, base
    upload_id, filename = base.split("-", 1)
    return upload_id + "-", filename

def replace_basename(original, new_base):
    if not isinstance(original, str):
        return original
    i = max(original.rfind("/"), original.rfind("\\"))
    return new_base if i == -1 else original[: i + 1] + new_base

def key_from_old_filename(fn):
    # keep full filename (no path), case-insensitive match later
    fn = fn.split("/")[-1].split("\\")[-1].strip()
    return fn

def normalize_csv_key(s):
    return key_from_old_filename(s).lower()


# ==========================
# LOAD CSV
# ==========================
def load_mapping(csv_path):
    """
    mapping[old_key_lower] = new_name
    old_key_lower is normalized (date/ext stripped if present).
    """
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # basic column check
        if CSV_OLD_COL not in reader.fieldnames or CSV_NEW_COL not in reader.fieldnames:
            raise ValueError(
                f"CSV is missing required columns. Found: {reader.fieldnames}. "
                f"Need: {CSV_OLD_COL!r} and {CSV_NEW_COL!r}"
            )

        for row in reader:
            old_raw = (row.get(CSV_OLD_COL) or "").strip()
            new_name = (row.get(CSV_NEW_COL) or "").strip()
            if not old_raw or not new_name:
                continue

            k = normalize_csv_key(old_raw)
            if k:
                mapping[k] = new_name

    return mapping

# ==========================
# UPDATE TASK
# ==========================
def update_task(task, mapping):
    old_filename = None
    upload_prefix = None

    data = task.get("data", {})
    if isinstance(data, dict) and isinstance(data.get("image"), str):
        upload_prefix, old_filename = split_upload_and_filename(data["image"])

    if old_filename is None and isinstance(task.get("file_upload"), str):
        upload_prefix, old_filename = split_upload_and_filename(task["file_upload"])

    if old_filename is None:
        return None, False

    key = key_from_old_filename(old_filename).lower()

    if key not in mapping:
        return None, False

    new_name = mapping[key]

    # update data.image
    if isinstance(data, dict) and isinstance(data.get("image"), str):
        upfx, _ = split_upload_and_filename(data["image"])
        upfx = upfx or upload_prefix or ""
        data["image"] = replace_basename(data["image"], f"{upfx}{new_name}")

    # update file_upload if present
    if isinstance(task.get("file_upload"), str):
        upfx, _ = split_upload_and_filename(task["file_upload"])
        upfx = upfx or upload_prefix or ""
        task["file_upload"] = f"{upfx}{new_name}"

    return task, True

# ==========================
# RUN
# ==========================
def main():
    mapping = load_mapping(CSV_PATH)
    print(f"[INFO] Loaded {len(mapping)} CSV keys (normalized).")
    print("[DIAG] First 10 normalized CSV keys:")
    for k in list(mapping.keys())[:10]:
        print("  ", k)

    with open(JSON_IN_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # diagnostics: show first 10 extracted JSON keys
    json_keys = []
    for t in tasks[:10]:
        old_filename = None
        data = t.get("data", {})
        if isinstance(data, dict) and isinstance(data.get("image"), str):
            _, old_filename = split_upload_and_filename(data["image"])
        if old_filename is None and isinstance(t.get("file_upload"), str):
            _, old_filename = split_upload_and_filename(t["file_upload"])
        if old_filename:
            json_keys.append(key_from_old_filename(old_filename).lower())

    print("[DIAG] First 10 extracted JSON keys:")
    for k in json_keys:
        print("  ", k)

    kept = []
    dropped = 0
    dropped_examples = []

    for t in tasks:
        updated, keep = update_task(t, mapping)
        if keep:
            kept.append(updated)
        else:
            dropped += 1
            if len(dropped_examples) < 20:
                # capture a filename example to debug mismatches
                fn = None
                d = t.get("data", {})
                if isinstance(d, dict) and isinstance(d.get("image"), str):
                    _, fn = split_upload_and_filename(d["image"])
                if fn is None and isinstance(t.get("file_upload"), str):
                    _, fn = split_upload_and_filename(t["file_upload"])
                if fn:
                    dropped_examples.append(fn)

    print(f"[INFO] Kept {len(kept)} tasks")
    print(f"[INFO] Dropped {dropped} tasks not present in CSV")
    if len(kept) == 0 and dropped_examples:
        print("[DIAG] Example JSON filenames that did NOT match CSV (up to 20):")
        for x in dropped_examples:
            print("  ", x)

    with open(JSON_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote filtered JSON: {JSON_OUT_PATH}")

if __name__ == "__main__":
    main()
