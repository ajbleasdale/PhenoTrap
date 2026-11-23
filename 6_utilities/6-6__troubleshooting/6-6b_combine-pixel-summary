
# ============================================================
# Combine class-separated pixel summary into one row per image
# ============================================================

from pathlib import Path
import pandas as pd
import re


# ============================================================
# SETTINGS
# ============================================================

INPUT_CSV = Path(
    r"path-to-pixel-summary.csv"
)

OUTPUT_CSV = Path(
    r"path-to-pixel-summary-fixed.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print(f"Input rows: {len(df):,}")


# ============================================================
# REMOVE _class_X SUFFIX FROM FILENAMES
# ============================================================

# Example:
# BE-LE_1068_2024-04-02_120000_class_1.png
# ->
# BE-LE_1068_2024-04-02_120000.png

df["filename"] = df["filename"].str.replace(
    r"_class_\d+(?=\.[^.]+$)",
    "",
    regex=True
)


# ============================================================
# IDENTIFY PIXEL COLUMNS
# ============================================================

pixel_cols = [
    col for col in df.columns
    if col.endswith("_pixels")
    and col not in ["background_pixels", "total_class_pixels"]
]

print("\nClass pixel columns:")
for col in pixel_cols:
    print(f"  {col}")


# ============================================================
# COMBINE ALL CLASS ROWS FOR EACH IMAGE
# ============================================================

# Sum pixel counts for all class files belonging to same image.
# Date is retained automatically because filename + date
# identify the original image.

combined = (
    df.groupby(["filename", "date"], as_index=False)[pixel_cols]
    .sum()
)


# ============================================================
# CALCULATE TOTAL CLASS PIXELS
# ============================================================

# Background is deliberately excluded.

combined["total_class_pixels"] = combined[pixel_cols].sum(axis=1)


# ============================================================
# CALCULATE PERCENTAGES
# ============================================================

for col in pixel_cols:

    # e.g. sky_pixels -> sky_percent
    percent_col = col.replace("_pixels", "_percent")

    combined[percent_col] = (
        combined[col]
        / combined["total_class_pixels"]
        * 100
    )

# Protect against images with zero classified pixels
percent_cols = [
    col.replace("_pixels", "_percent")
    for col in pixel_cols
]

combined[percent_cols] = combined[percent_cols].fillna(0)


# ============================================================
# OPTIONAL: ROUND PERCENTAGES
# ============================================================

combined[percent_cols] = combined[percent_cols].round(4)


# ============================================================
# COLUMN ORDER
# ============================================================

combined = combined[
    ["filename", "date"]
    + pixel_cols
    + ["total_class_pixels"]
    + percent_cols
]


# ============================================================
# SAVE
# ============================================================

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

combined.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("Finished")
print("========================================")

print(f"Original class rows: {len(df):,}")
print(f"Combined images:     {len(combined):,}")
print(f"Output: {OUTPUT_CSV}")

print("\nFirst rows:")
print(combined.head())
