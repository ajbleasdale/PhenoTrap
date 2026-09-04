import os
import json
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict

# === Configure Paths ===
json_path = r"path-to-JSON-file.JSON"
output_folder = r"path-to-output-directory"
os.makedirs(output_folder, exist_ok=True)


class_map = {
    "background": 0,
    "sky": 1,
    "snow_ice": 2,
    "bare": 3,

    "understory": 4,
    "canopy": 5,

    "ignore": 255,

}

RAW_IGNORE_VALUE = 255
num_classes = 6
ignore_index = RAW_IGNORE_VALUE


def json_to_multiclass(json_path, output_folder, class_map):

    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    pixel_counts = defaultdict(int)
    polygon_counts = defaultdict(int)

    for task in tasks:
        task_id = task.get("id", "unknown")
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        results = annotations[0].get("result", [])
        if not results:
            continue

        width = results[0].get("original_width")
        height = results[0].get("original_height")
        if not width or not height:
            continue

        combined = np.zeros((height, width), dtype=np.uint8)

        for result in results:
            value = result.get("value", {})
            if "points" not in value:
                continue

            points = value["points"]
            polygon = [(x / 100 * width, y / 100 * height) for x, y in points]

            labels = value.get("polygonlabels", [])
            label = labels[0] if labels else "unlabeled"
            label_clean = "".join(c if c.isalnum() else "_" for c in label).lower()

            if label_clean not in class_map:
                print(f"Skipped label: {label_clean}")
                continue

            class_val = class_map[label_clean]

            polygon_counts[label_clean] += 1

            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(polygon, outline=255, fill=255)

            mask_np = np.array(mask) > 128
            combined[mask_np] = class_val

        out_path = os.path.join(output_folder, f"mask_{task_id}.png")
        Image.fromarray(combined).save(out_path, format="PNG", compress_level=0)
        print(f"Saved multi-class mask: {out_path}")

        # accumulate pixel counts
        vals, counts = np.unique(combined, return_counts=True)
        for v, c in zip(vals, counts):
            pixel_counts[v] += int(c)

    print("\n===== Polygon Counts =====")
    for k in sorted(polygon_counts):
        print(f"{k}: {polygon_counts[k]}")

    print("\n===== Pixel Counts =====")
    id_to_class = {v: k for k, v in class_map.items()}
    for cid in sorted(pixel_counts):
        name = id_to_class.get(cid, "unknown")
        print(f"{cid:3d} ({name}): {pixel_counts[cid]}")

    print("\n✅ Streamlined conversion complete.")


json_to_multiclass(json_path, output_folder, class_map)
