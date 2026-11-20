import os
import json
import cv2
import pandas as pd
import numpy as np

# ----------------------------
# CONFIG (EDIT THESE)
# ----------------------------
CSV_PATH   = r"path-to-index.csv"
IMAGE_COL  = "image_name"          # column with image filename (e.g. SE-NM_1_2017-03-26_120000.JPG)
MASK_COL   = "mask_name"             # set to a column name if mask filename differs; else None => same as image
SUBFOLDER_COL = "subfolder"   # <-- column in CSV that contains e.g. SE-NM-1

IMAGES_DIR = r"path-to-image-directory"
MASKS_DIR  = r"path-to-mask-directory"
IMAGE_BASE_URL = "http://127.0.0.1:8001" # run python -m http.server inside IMAGES_DIR

# These MUST match your labeling config tag names
FROM_NAME = "label"   # <PolygonLabels name="label" ...>
TO_NAME   = "image"   # <Image name="image" ...>

# The polygon label name to apply (must match one label in your PolygonLabels list)
# Classes + colors


class_map = {
    "background": 0,
    "sky": 1,
    "snow_ice": 2,
    "bare": 3,


    "ericaceous_shrub": 4,
    "briar_seedling": 5,
    "graminoid": 6,
    "forb": 7,
    "fern": 8,

    "cryptogam": 9,

    "vine": 10,
    "broadleaf_evergreen": 11,



    "conifer": 12,
    "broadleaf_deciduous": 13,

    "ignore": 255,
}

# ---- include only non-background, non-ignore labels ----
PREDICT_CLASSES = [k for k, v in class_map.items() if v not in (0, 255)]

# Polygon extraction tuning
MIN_AREA_PX     = 50
APPROX_EPS_FRAC = 0.0005

# ---- NEW: overlap + boundary + splitting controls ----
NO_OVERLAP = True          # ensures no pixel overlaps between classes (priority = PREDICT_CLASSES order)
ERODE_PX   = 1            # 0=off, try 1 (shrinks regions to reduce boundary overlaps)
SPLIT_OPEN_PX = 2          # 0=off, try 1 or 2 to split big blobs into more components (opening)

OUT_JSON = r"path-to-output-JSON.JSON"  # Change to output location


# ----------------------------
# Helpers
# ----------------------------
def file_exists(p: str) -> bool:
    try:
        return os.path.isfile(p)
    except OSError:
        return False


def make_image_url(subfolder: str, fname: str) -> str:
    return f"{IMAGE_BASE_URL}/{subfolder}/{fname}"


def find_pred_mask_path(subfolder: str, img_filename: str, mask_name: str | None):
    """
    If your masks are stored flat in MASKS_DIR (recommended), this will find them there.
    If your masks are stored in per-subfolder directories, it will try that too.
    """
    base, _ = os.path.splitext(img_filename)

    candidates = []
    if mask_name:
        candidates.append(mask_name)
    candidates.append(f"{base}.png")
    candidates.append(f"{img_filename}.png")

    # Try flat
    for cand in candidates:
        p = os.path.join(MASKS_DIR, cand)
        if file_exists(p):
            return p

    # Try per-subfolder
    for cand in candidates:
        p = os.path.join(MASKS_DIR, subfolder, cand)
        if file_exists(p):
            return p

    raise FileNotFoundError(f"No predicted mask found for {subfolder}/{img_filename} (tried: {candidates})")


def _apply_erode_and_split(bw_255: np.ndarray) -> np.ndarray:
    """
    bw_255 is uint8 {0,255}
    """
    out = bw_255

    # Split large blobs / remove thin bridges before contouring
    if SPLIT_OPEN_PX > 0:
        k = 2 * SPLIT_OPEN_PX + 1
        kernel = np.ones((k, k), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)

    # Shrink boundaries to reduce tiny overlaps / boundary artifacts
    if ERODE_PX > 0:
        k = 2 * ERODE_PX + 1
        kernel = np.ones((k, k), np.uint8)
        out = cv2.erode(out, kernel, iterations=1)

    return out


def mask_to_polygons_by_class(mask_path: str):
    """
    Returns: dict[label_name] = list of (pts_pct, w, h)
    Ensures:
      - background/ignore excluded (via PREDICT_CLASSES)
      - optional NO_OVERLAP pixel claiming
      - optional SPLIT_OPEN_PX and ERODE_PX morphology
    """
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")

    h, w = m.shape[:2]
    out = {k: [] for k in PREDICT_CLASSES}

    occupied = np.zeros((h, w), dtype=bool) if NO_OVERLAP else None

    for label_name in PREDICT_CLASSES:
        class_id = class_map[label_name]

        if NO_OVERLAP:
            bw_bool = (m == class_id) & (~occupied)
            occupied |= bw_bool
        else:
            bw_bool = (m == class_id)

        bw = bw_bool.astype(np.uint8) * 255
        bw = _apply_erode_and_split(bw)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_PX:
                continue

            peri = cv2.arcLength(cnt, True)
            eps = max(1.0, APPROX_EPS_FRAC * peri)
            approx = cv2.approxPolyDP(cnt, eps, True)

            pts = approx.reshape(-1, 2).astype(np.float32)
            if pts.shape[0] < 3:
                continue

            pts_pct = []
            for x, y in pts:
                x_pct = float(np.clip((x / w) * 100.0, 0.0, 100.0))
                y_pct = float(np.clip((y / h) * 100.0, 0.0, 100.0))
                pts_pct.append([x_pct, y_pct])

            out[label_name].append((pts_pct, w, h))

    return out


# ----------------------------
# Build tasks
# ----------------------------
df = pd.read_csv(CSV_PATH)

tasks = []
missing_images = []
missing_pred_masks = []
made = 0

for _, row in df.iterrows():
    img_name = str(row[IMAGE_COL]).strip()
    subfolder = str(row[SUBFOLDER_COL]).strip()
    mask_name = None if MASK_COL is None else str(row[MASK_COL]).strip()

    # image exists check (kept flexible)
    img_path = os.path.join(IMAGES_DIR, subfolder, img_name) if os.path.isdir(os.path.join(IMAGES_DIR, subfolder)) else os.path.join(IMAGES_DIR, img_name)
    if not file_exists(img_path):
        missing_images.append(f"{subfolder}/{img_name}")
        continue

    try:
        pred_mask_path = find_pred_mask_path(subfolder, img_name, mask_name)
    except FileNotFoundError:
        missing_pred_masks.append(f"{subfolder}/{img_name}")
        continue

    polys_by_class = mask_to_polygons_by_class(pred_mask_path)

    result = []
    for label_name, polygons in polys_by_class.items():
        for pts_pct, w, h in polygons:
            result.append({
                "from_name": FROM_NAME,
                "to_name": TO_NAME,
                "type": "polygonlabels",
                "value": {
                    "points": pts_pct,
                    "closed": True,
                    "polygonlabels": [label_name],
                },
                "original_width": w,
                "original_height": h,
                "image_rotation": 0
            })

    task = {
        "data": {
            "image": make_image_url(subfolder, img_name)
        },
        "predictions": [{
            "model_version": "model_pred_mask2poly",
            "score": 1.0,
            "result": result
        }]
    }

    tasks.append(task)
    made += 1

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False)

print(f"Wrote: {OUT_JSON}")
print(f"Tasks created: {made}")
print(f"Missing images: {len(missing_images)}")
print(f"Missing predicted masks: {len(missing_pred_masks)}")
if missing_images[:10]:
    print("Example missing images:", missing_images[:10])
if missing_pred_masks[:10]:
    print("Example missing predicted masks:", missing_pred_masks[:10])
