
# class_checker_presence_and_pixelpct.py
# Edit ONLY the 4 PATHS below, then press Run in PyCharm.

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# =========================
# EDIT THESE PATHS
# =========================
INDEX_CSV = Path(r"path-to-index.csv")
IMAGE_ROOT = Path(r"path-to-image-directory")
MASK_ROOT = Path(r"path-to-mask-directory")
OUT_XLSX = Path(r"path-to-output.xlsx")

# If your validation split is named "test", change this to "test"
VAL_SPLIT_NAME = "test"
# =========================


def normalize_split(x: str) -> str:
    s = str(x).strip().lower()
    if s in ("train", "training", "tr"):
        return "train"
    if s in ("val", "valid", "validation", "dev"):
        return "val"
    if s in ("test", "testing", "te"):
        return "test"
    return s


def infer_cols(df: pd.DataFrame):
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(options):
        for o in options:
            if o in cols:
                return cols[o]
        return None

    image_col = pick(["image_name", "image", "img", "filename", "file"])
    mask_col  = pick(["mask_name", "mask", "label", "annotation"])
    split_col = pick(["partition", "split", "set", "subset", "usage"])

    if mask_col is None or split_col is None:
        raise ValueError(
            "Index CSV must contain at least: mask_name (or mask) and partition (or split)."
        )

    return image_col, mask_col, split_col


def load_label_mask(mask_path: Path) -> np.ndarray:
    """Single-channel class-id mask expected."""
    with Image.open(mask_path) as im:
        arr = np.array(im)

    if arr.ndim == 3:
        # collapse if RGB channels identical
        if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"Mask is multi-channel with differing channels (expected single-channel IDs): {mask_path}"
            )

    return arr.astype(np.int64)


def main():
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"INDEX_CSV not found: {INDEX_CSV}")
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"IMAGE_ROOT not found: {IMAGE_ROOT}")
    if not MASK_ROOT.exists():
        raise FileNotFoundError(f"MASK_ROOT not found: {MASK_ROOT}")

    df = pd.read_csv(INDEX_CSV)
    image_col, mask_col, split_col = infer_cols(df)

    df["_split"] = df[split_col].apply(normalize_split)

    # Build full paths from filenames
    if image_col is not None:
        df["_image_path"] = df[image_col].astype(str).apply(lambda n: (IMAGE_ROOT / n).resolve())
    else:
        df["_image_path"] = ""

    df["_mask_path"] = df[mask_col].astype(str).apply(lambda n: (MASK_ROOT / n).resolve())

    # Check masks exist
    exists_mask = df["_mask_path"].apply(lambda p: Path(p).exists())
    if not exists_mask.all():
        missing = df.loc[~exists_mask, "_mask_path"].astype(str).head(20).tolist()
        raise FileNotFoundError(
            f"{(~exists_mask).sum()} masks not found. First up to 20:\n" + "\n".join(missing)
        )

    # Determine all class IDs across all masks
    all_class_ids = set()
    for mp in df["_mask_path"]:
        m = load_label_mask(Path(mp))
        all_class_ids.update(np.unique(m).tolist())
    class_ids = np.array(sorted(all_class_ids), dtype=np.int64)

    # Split counts
    split_counts = df["_split"].value_counts().to_dict()

    # Aggregators per split/class
    splits = ["train", "val", "test"]
    images_with_class = {s: {int(c): 0 for c in class_ids} for s in splits}
    class_pixels = {s: {int(c): 0 for c in class_ids} for s in splits}
    total_pixels = {s: 0 for s in splits}

    # For means
    sum_pixel_pct_overall = {s: {int(c): 0.0 for c in class_ids} for s in splits}   # absent treated as 0
    sum_pixel_pct_present = {s: {int(c): 0.0 for c in class_ids} for s in splits}   # only when present
    n_present = {s: {int(c): 0 for c in class_ids} for s in splits}

    # Per-image validation output
    val_key = normalize_split(VAL_SPLIT_NAME)
    per_image_rows = []

    for _, row in df.iterrows():
        split = row["_split"]
        if split not in splits:
            continue

        mp = Path(row["_mask_path"])
        m = load_label_mask(mp)
        flat = m.ravel()
        npx = int(flat.size)
        total_pixels[split] += npx

        # counts for all classes (fast bincount)
        bc = np.bincount(flat, minlength=int(class_ids.max()) + 1)

        present_set = set(np.unique(m).tolist())

        # Update per-class aggregates
        for cid in class_ids:
            cid = int(cid)
            cnt = int(bc[cid]) if cid < len(bc) else 0

            class_pixels[split][cid] += cnt

            pct = (cnt / npx) if npx else 0.0
            sum_pixel_pct_overall[split][cid] += pct

            if cid in present_set:
                images_with_class[split][cid] += 1
                sum_pixel_pct_present[split][cid] += pct
                n_present[split][cid] += 1

        # Per-image row (validation split only)
        if split == val_key:
            out = {
                "image_path": str(row["_image_path"]) if image_col else "",
                "mask_path": str(mp),
                "split": split,
                "total_pixels": npx,
            }
            for cid in class_ids:
                cid = int(cid)
                cnt = int(bc[cid]) if cid < len(bc) else 0
                out[f"class_{cid}_present"] = 1 if cnt > 0 else 0
                out[f"class_{cid}_pixel_pct"] = (cnt / npx) if npx else 0.0
            per_image_rows.append(out)

    # Summary sheet
    summary_rows = []
    for split in splits:
        n_items = int(split_counts.get(split, 0))
        if n_items == 0:
            continue

        for cid in class_ids:
            cid = int(cid)
            iwc = images_with_class[split][cid]
            tp = total_pixels[split]
            cp = class_pixels[split][cid]

            summary_rows.append({
                "split": split,
                "class_id": cid,

                # image-level presence
                "images_in_split": n_items,
                "images_with_class": int(iwc),
                "presence_pct": (iwc / n_items) if n_items else 0.0,

                # pixel-level counts within this split
                "total_pixels_in_split": int(tp),
                "class_pixels_in_split": int(cp),
                "pixel_pct_in_split": (cp / tp) if tp else 0.0,

                # mean per-image pixel percentages
                "mean_pixel_pct_overall": (
                        sum_pixel_pct_overall[split][cid] / n_items
                ) if n_items else 0.0,

                "mean_pixel_pct_when_present": (
                        sum_pixel_pct_present[split][cid] / n_present[split][cid]
                ) if n_present[split][cid] else 0.0,

                # clearer duplicate names for reporting
                "class_pixel_count_for_this_split": int(cp),
                "class_pixel_percentage_for_this_split": (cp / tp) if tp else 0.0,
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(["split", "class_id"])
    per_image_df = pd.DataFrame(per_image_rows)

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        per_image_df.to_excel(writer, sheet_name="per_image_val", index=False)

    print(f"Wrote: {OUT_XLSX}")


if __name__ == "__main__":
    main()
