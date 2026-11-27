
# class_checker_presence_and_pixelpct_WHOLE.py
# Edit ONLY the 4 PATHS below, then press Run.

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
# =========================


def infer_cols(df: pd.DataFrame):
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(options):
        for o in options:
            if o in cols:
                return cols[o]
        return None

    image_col = pick(["image_name", "image", "img", "filename", "file"])
    mask_col = pick(["mask_name", "mask", "label", "annotation"])

    if mask_col is None:
        raise ValueError("Index CSV must contain at least: mask_name or mask.")

    return image_col, mask_col


def load_label_mask(mask_path: Path) -> np.ndarray:
    """Single-channel class-id mask expected."""
    with Image.open(mask_path) as im:
        arr = np.array(im)

    if arr.ndim == 3:
        if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"Mask is multi-channel with differing channels: {mask_path}"
            )

    return arr.astype(np.int64)


def main():
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"INDEX_CSV not found: {INDEX_CSV}")
    if not MASK_ROOT.exists():
        raise FileNotFoundError(f"MASK_ROOT not found: {MASK_ROOT}")

    df = pd.read_csv(INDEX_CSV)
    image_col, mask_col = infer_cols(df)

    if image_col is not None:
        df["_image_path"] = df[image_col].astype(str).apply(lambda n: IMAGE_ROOT / n)
    else:
        df["_image_path"] = ""

    df["_mask_path"] = df[mask_col].astype(str).apply(lambda n: MASK_ROOT / n)

    exists_mask = df["_mask_path"].apply(lambda p: Path(p).exists())
    if not exists_mask.all():
        missing = df.loc[~exists_mask, "_mask_path"].astype(str).head(20).tolist()
        raise FileNotFoundError(
            f"{(~exists_mask).sum()} masks not found. First up to 20:\n"
            + "\n".join(missing)
        )

    # Determine class IDs across the whole dataset
    all_class_ids = set()
    for mp in df["_mask_path"]:
        m = load_label_mask(Path(mp))
        all_class_ids.update(np.unique(m).tolist())

    class_ids = np.array(sorted(all_class_ids), dtype=np.int64)

    n_images = len(df)
    images_with_class = {int(c): 0 for c in class_ids}
    class_pixels = {int(c): 0 for c in class_ids}
    total_pixels = 0

    sum_pixel_pct_overall = {int(c): 0.0 for c in class_ids}
    sum_pixel_pct_present = {int(c): 0.0 for c in class_ids}
    n_present = {int(c): 0 for c in class_ids}

    per_image_rows = []

    for _, row in df.iterrows():
        mp = Path(row["_mask_path"])
        m = load_label_mask(mp)

        flat = m.ravel()
        npx = int(flat.size)
        total_pixels += npx

        bc = np.bincount(flat, minlength=int(class_ids.max()) + 1)

        out = {
            "image_path": str(row["_image_path"]) if image_col else "",
            "mask_path": str(mp),
            "total_pixels": npx,
        }

        for cid in class_ids:
            cid = int(cid)
            cnt = int(bc[cid]) if cid < len(bc) else 0
            pct = (cnt / npx) if npx else 0.0

            class_pixels[cid] += cnt
            sum_pixel_pct_overall[cid] += pct

            if cnt > 0:
                images_with_class[cid] += 1
                sum_pixel_pct_present[cid] += pct
                n_present[cid] += 1

            out[f"class_{cid}_present"] = 1 if cnt > 0 else 0
            out[f"class_{cid}_pixel_pct"] = pct

        per_image_rows.append(out)

    summary_rows = []

    for cid in class_ids:
        cid = int(cid)
        iwc = images_with_class[cid]
        cp = class_pixels[cid]

        summary_rows.append({
            "class_id": cid,
            "images_total": n_images,
            "images_with_class": int(iwc),
            "presence_pct": (iwc / n_images) if n_images else 0.0,
            "total_pixels": int(total_pixels),
            "class_pixels": int(cp),
            "pixel_pct_total": (cp / total_pixels) if total_pixels else 0.0,
            "mean_pixel_pct_overall": (
                sum_pixel_pct_overall[cid] / n_images
            ) if n_images else 0.0,
            "mean_pixel_pct_when_present": (
                sum_pixel_pct_present[cid] / n_present[cid]
            ) if n_present[cid] else 0.0,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("class_id")
    per_image_df = pd.DataFrame(per_image_rows)

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary_whole", index=False)
        per_image_df.to_excel(writer, sheet_name="per_image_all", index=False)

    print(f"Wrote: {OUT_XLSX}")


if __name__ == "__main__":
    main()
