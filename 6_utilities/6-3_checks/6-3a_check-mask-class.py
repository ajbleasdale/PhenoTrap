import os
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
from PIL import Image


def _load_mask_as_array(mask_path: str, mode: str = "auto") -> np.ndarray:
    """
    Load a mask image into a 2D integer array.

    mode:
      - "auto": if image has 3/4 channels, treat it as RGB and encode colors to a single int
      - "grayscale": force to L and return 2D values (0..255 or higher if 16-bit source)
      - "rgb": force to RGB and encode colors to a single int
    """
    img = Image.open(mask_path)

    if mode == "grayscale":
        arr = np.array(img.convert("L"))
        return arr

    if mode == "rgb":
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        # Encode RGB -> int: R<<16 | G<<8 | B
        return (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2].astype(np.uint32)

    # auto
    if img.mode in ("RGB", "RGBA", "P"):
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        return (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2].astype(np.uint32)

    # Anything else -> treat as grayscale-like (L, I;16, etc.)
    arr = np.array(img)
    if arr.ndim == 3:
        # Fallback if some weird format yields 3D
        arr = arr[..., 0]
    return arr


def list_classes_in_masks_to_csv(
    mask_folder: str,
    output_csv: str,
    class_map: Optional[Dict[int, str]] = None,
    mask_exts: Tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg", ".jpeg"),
    mode: str = "auto",
    include_background: bool = True,
    background_values: Tuple[int, ...] = (0,),
) -> None:
    """
    Scan all masks in a folder, find which classes are present per mask, and write a CSV where
    each class name is a column (1 = present, 0 = absent).

    class_map:
      - If provided: maps pixel values (grayscale ids OR encoded RGB ints) -> class name
      - If not provided: columns will be "class_<value>" for each observed unique value

    mode:
      - "auto" (recommended), "grayscale", or "rgb" (see _load_mask_as_array)

    include_background/background_values:
      - If include_background=False, those values are ignored when detecting classes.
    """
    mask_folder = str(Path(mask_folder))
    output_csv = str(Path(output_csv))

    # Collect mask files
    mask_paths: List[str] = []
    for root, _, files in os.walk(mask_folder):
        for fn in files:
            if fn.lower().endswith(mask_exts):
                mask_paths.append(os.path.join(root, fn))
    mask_paths.sort()

    if not mask_paths:
        raise FileNotFoundError(f"No mask files found in: {mask_folder}")

    per_mask_values: Dict[str, Set[int]] = {}
    all_values: Set[int] = set()

    for p in mask_paths:
        arr = _load_mask_as_array(p, mode=mode)
        uniq = np.unique(arr).astype(np.int64)
        values = set(int(v) for v in uniq)

        if not include_background:
            for bv in background_values:
                values.discard(int(bv))

        per_mask_values[p] = values
        all_values |= values

    # Build columns
    if class_map:
        # Only keep values that appear OR that are in the map? Here: all appearing values.
        # Add any mapped values that appear (even if not in all_values after bg filtering).
        values_sorted = sorted(all_values)
        class_names = [class_map.get(v, f"class_{v}") for v in values_sorted]
        # Handle potential duplicate class names by making them unique
        seen = {}
        unique_class_names = []
        for name in class_names:
            if name not in seen:
                seen[name] = 1
                unique_class_names.append(name)
            else:
                seen[name] += 1
                unique_class_names.append(f"{name}__{seen[name]}")
        col_for_value = dict(zip(values_sorted, unique_class_names))
        columns = ["mask"] + unique_class_names
    else:
        values_sorted = sorted(all_values)
        columns = ["mask"] + [f"class_{v}" for v in values_sorted]
        col_for_value = {v: f"class_{v}" for v in values_sorted}

    # Write CSV (one row per mask; 1/0 for presence)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for p in mask_paths:
            row = {c: 0 for c in columns}
            row["mask"] = os.path.relpath(p, mask_folder)

            for v in per_mask_values[p]:
                col = col_for_value.get(v)
                if col is not None:
                    row[col] = 1

            writer.writerow(row)

    print(f"Done.\nMasks scanned: {len(mask_paths)}\nClasses (columns): {len(columns)-1}\nOutput: {output_csv}")


if __name__ == "__main__":
    # === EDIT THESE ===
    MASK_FOLDER = r"path-to-mask-folder"
    OUTPUT_CSV = r"path-to-output-csv.csv"

    # OPTIONAL: provide a mapping from pixel value -> class name
    # For grayscale ID masks (common):
    # CLASS_MAP = {0: "background", 1: "sky", 2: "vegetation", 3: "ground"}
    #
    # For color masks, map ENCODED RGB ints:
    # Example: pure red (255,0,0) becomes (255<<16 | 0<<8 | 0) = 16711680
    # CLASS_MAP = {0: "background", 16711680: "sky"}  # etc.
    class_map = {
        "background": 0,
        "snow_ice": 1,
        "low_lying_shrub": 2,
        "graminoid": 3,
        "forb": 4,
        "fern": 5,
        "lichen_bryophyte": 6,
        "ground_cover_other": 7,
        "spruce": 8,
        "pine": 9,
        "birch": 10,
        "ignore": 255,
    }

    # Convert to (id -> name) as required by the script
    ID_TO_CLASS = {v: k for k, v in class_map.items()}

# Convert to (id -> name) as required by the script
list_classes_in_masks_to_csv(
    mask_folder=MASK_FOLDER,
    output_csv=OUTPUT_CSV,
    class_map=ID_TO_CLASS,
    mode="grayscale",          # correct for ID masks
    include_background=False,  # usually desired
    background_values=(0, 255) # background + ignore
)

