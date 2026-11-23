
# ==========================================================
# WHOLE-IMAGE GREENNESS (GCC) -> CSV
# ==========================================================
#
# For every whole camera-trap image:
#
#   1. Read the RGB image
#   2. Retain all valid pixels
#   3. Calculate GCC independently for every pixel:
#
#          GCC = G / (R + G + B)
#
#   4. Average pixel-level GCC values to obtain one
#      whole-image GCC value
#
# Output:
#   One row per image
#
# ==========================================================

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ==========================================================
# SETTINGS
# ==========================================================

# Folder containing the ORIGINAL whole images
INPUT_DIR = Path(
    r"path-to-image-dataset"
)

# Output CSV
OUTPUT_CSV = Path(
    r"path-tooutput-csv.csv"
)


# Search inside subfolders too?
SEARCH_SUBFOLDERS = True


# ==========================================================
# IMAGE EXTENSIONS
# ==========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
}


# ==========================================================
# OPTIONAL BRIGHTNESS FILTER
# ==========================================================
#
# False = retain every non-black pixel.
#
# Keep this False if you want the calculation to match
# the class-level calculation we were using.
# ==========================================================

USE_BRIGHTNESS_FILTER = False

MIN_BRIGHTNESS = 10


# ==========================================================
# CALCULATE WHOLE-IMAGE GREENNESS
# ==========================================================

def calculate_greenness_metrics(image_path):

    # ------------------------------------------------------
    # READ IMAGE
    # ------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        return None


    # OpenCV reads BGR, so convert to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float64)


    # ------------------------------------------------------
    # RGB CHANNELS
    # ------------------------------------------------------

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]


    # ------------------------------------------------------
    # VALID PIXELS
    # ------------------------------------------------------

    rgb_sum = R + G + B

    # Exclude completely black pixels because:
    #
    # R + G + B = 0
    #
    # would make GCC undefined.

    valid = rgb_sum > 0


    # ------------------------------------------------------
    # OPTIONAL BRIGHTNESS FILTER
    # ------------------------------------------------------

    brightness = (
        R + G + B
    ) / 3.0


    if USE_BRIGHTNESS_FILTER:

        valid = (
            valid
            &
            (brightness >= MIN_BRIGHTNESS)
        )


    # ------------------------------------------------------
    # RETAIN VALID PIXELS
    # ------------------------------------------------------

    Rv = R[valid]
    Gv = G[valid]
    Bv = B[valid]


    if len(Rv) == 0:

        print(
            f"No valid pixels: {image_path}"
        )

        return None


    # ======================================================
    # GCC
    # ======================================================
    #
    # GCC is calculated PER PIXEL first:
    #
    #       G
    #  -----------
    #   R + G + B
    #
    # and then averaged across the whole image.
    #
    # ======================================================

    denominator = (
        Rv + Gv + Bv
    )


    gcc_pixels = (
        Gv / denominator
    )


    GCC = np.mean(
        gcc_pixels
    )


    # ======================================================
    # OTHER USEFUL IMAGE METRICS
    # ======================================================

    RCC = np.mean(
        Rv / denominator
    )

    BCC = np.mean(
        Bv / denominator
    )


    mean_R = np.mean(Rv)
    mean_G = np.mean(Gv)
    mean_B = np.mean(Bv)


    mean_brightness = np.mean(
        (Rv + Gv + Bv) / 3.0
    )


    # ======================================================
    # RETURN RESULTS
    # ======================================================

    return {

        "valid_pixel_count":
            len(Rv),

        "mean_R":
            mean_R,

        "mean_G":
            mean_G,

        "mean_B":
            mean_B,

        "brightness":
            mean_brightness,

        "GCC":
            GCC,

        "RCC":
            RCC,

        "BCC":
            BCC,
    }


# ==========================================================
# FIND IMAGES
# ==========================================================

if SEARCH_SUBFOLDERS:

    image_files = sorted([
        path
        for path in INPUT_DIR.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ])

else:

    image_files = sorted([
        path
        for path in INPUT_DIR.iterdir()
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ])


if not image_files:

    raise ValueError(
        f"No images found in:\n{INPUT_DIR}"
    )


print(
    f"\nImages found: {len(image_files)}"
)


# ==========================================================
# PROCESS IMAGES
# ==========================================================

results = []


for number, image_path in enumerate(
    image_files,
    start=1
):

    metrics = calculate_greenness_metrics(
        image_path
    )


    if metrics is None:
        continue


    row = {

        "filename":
            image_path.name,

        "relative_path":
            str(
                image_path.relative_to(
                    INPUT_DIR
                )
            ),

        **metrics,
    }


    results.append(row)


    if number % 100 == 0:

        print(
            f"Processed "
            f"{number}/"
            f"{len(image_files)}"
        )


# ==========================================================
# CREATE DATAFRAME
# ==========================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise ValueError(
        "No valid images were processed."
    )


# ==========================================================
# EXTRACT DATE FROM FILENAME
# ==========================================================
#
# Recognises:
#
# YYYY-MM-DD
# YYYY_MM_DD
# YYYYMMDD
#
# ==========================================================

date_string = (
    results_df["filename"]
    .str.extract(
        r"(\d{4}[-_]?\d{2}[-_]?\d{2})",
        expand=False
    )
)


date_string = (
    date_string
    .str.replace(
        "_",
        "-",
        regex=False
    )
)


# ----------------------------------------------------------
# YYYY-MM-DD
# ----------------------------------------------------------

results_df["date"] = pd.to_datetime(
    date_string,
    format="%Y-%m-%d",
    errors="coerce"
)


# ----------------------------------------------------------
# YYYYMMDD
# ----------------------------------------------------------

missing_date = (
    results_df["date"].isna()
)


if missing_date.any():

    compact_date = (
        results_df.loc[
            missing_date,
            "filename"
        ]
        .str.extract(
            r"(\d{8})",
            expand=False
        )
    )


    results_df.loc[
        missing_date,
        "date"
    ] = pd.to_datetime(
        compact_date,
        format="%Y%m%d",
        errors="coerce"
    )


# ==========================================================
# REORDER COLUMNS
# ==========================================================

column_order = [

    "filename",
    "date",
    "relative_path",

    "valid_pixel_count",

    "mean_R",
    "mean_G",
    "mean_B",

    "brightness",

    "GCC",
    "RCC",
    "BCC",
]


results_df = results_df[
    column_order
]


# ==========================================================
# SORT BY DATE
# ==========================================================

results_df = results_df.sort_values(
    [
        "date",
        "filename"
    ],
    na_position="last"
)


# ==========================================================
# SAVE CSV
# ==========================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)

print("FINISHED")

print("=" * 60)


print(
    f"Images processed: "
    f"{len(results_df)}"
)


print(
    f"Images without extracted date: "
    f"{results_df['date'].isna().sum()}"
)


print(
    f"Mean whole-image GCC: "
    f"{results_df['GCC'].mean():.4f}"
)


print(
    f"\nCSV saved to:\n"
    f"{OUTPUT_CSV}"
)
