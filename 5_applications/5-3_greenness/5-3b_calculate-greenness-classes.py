
# ==========================================================
# IMAGE / CLASS GREENNESS METRICS -> CSV
# ==========================================================
#
# For every image in every class folder:
#
#   1. Read RGB image
#   2. Identify valid pixels
#   3. Calculate colour indices PER PIXEL
#   4. Average pixel values to obtain one value per image
#
# GCC calculation:
#
#   GCC_pixel = G / (R + G + B)
#
#   GCC_image = mean(GCC_pixel)
#
# ==========================================================

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ==========================================================
# SETTINGS
# ==========================================================

# Parent directory containing the class folders
INPUT_DIR = Path(
    r"path-to-organised-images-by-class"
)

# Output CSV
OUTPUT_CSV = Path(
    r"path-to-output.csv"
)


# ----------------------------------------------------------
# IMAGE EXTENSIONS
# ----------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
}


# ----------------------------------------------------------
# OPTIONAL BRIGHTNESS FILTER
# ----------------------------------------------------------
#
# False:
#   all non-black pixels are retained
#
# True:
#   pixels below MIN_BRIGHTNESS are excluded
#
# Leave False if you did not previously use a brightness
# threshold.
# ----------------------------------------------------------

USE_BRIGHTNESS_FILTER = False

MIN_BRIGHTNESS = 10


# ==========================================================
# FUNCTIONS
# ==========================================================

def calculate_greenness_metrics(image_path):

    # ------------------------------------------------------
    # READ IMAGE
    # ------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        return None


    # OpenCV reads BGR.
    # Convert explicitly to RGB.

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    image = image.astype(
        np.float64
    )


    # ------------------------------------------------------
    # SPLIT RGB CHANNELS
    # ------------------------------------------------------

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]


    # ------------------------------------------------------
    # VALID PIXELS
    # ------------------------------------------------------

    rgb_sum = (
        R + G + B
    )


    # Prevent division by zero.
    # Completely black pixels are excluded.

    valid = rgb_sum > 0


    # Optional additional brightness filtering

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
    # EXTRACT VALID PIXELS
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
    # CHROMATIC COORDINATES
    # ======================================================
    #
    # IMPORTANT:
    #
    # GCC is calculated independently for every pixel:
    #
    #     G / (R + G + B)
    #
    # and THEN averaged.
    #
    # We are NOT calculating:
    #
    # mean(G) / (mean(R) + mean(G) + mean(B))
    #
    # ======================================================

    denominator = (
        Rv + Gv + Bv
    )


    gcc_pixels = (
        Gv / denominator
    )

    rcc_pixels = (
        Rv / denominator
    )

    bcc_pixels = (
        Bv / denominator
    )


    GCC = np.mean(
        gcc_pixels
    )

    RCC = np.mean(
        rcc_pixels
    )

    BCC = np.mean(
        bcc_pixels
    )


    # ======================================================
    # MEAN RGB
    # ======================================================

    mean_R = np.mean(Rv)
    mean_G = np.mean(Gv)
    mean_B = np.mean(Bv)


    # ======================================================
    # MEAN BRIGHTNESS
    # ======================================================

    mean_brightness = np.mean(
        (Rv + Gv + Bv) / 3.0
    )


    # ======================================================
    # EXCESS GREEN (ExG)
    # ======================================================
    #
    # ExG = 2G - R - B
    #
    # Calculated per pixel and then averaged.
    # ======================================================

    exg_pixels = (
        2 * Gv
        - Rv
        - Bv
    )

    ExG = np.mean(
        exg_pixels
    )


    # ======================================================
    # GRVI
    # ======================================================
    #
    # GRVI = (G - R) / (G + R)
    #
    # ======================================================

    grvi_denominator = (
        Gv + Rv
    )

    valid_grvi = (
        grvi_denominator != 0
    )


    if np.any(valid_grvi):

        grvi_pixels = (
            (
                Gv[valid_grvi]
                -
                Rv[valid_grvi]
            )
            /
            grvi_denominator[valid_grvi]
        )

        GRVI = np.mean(
            grvi_pixels
        )

    else:

        GRVI = np.nan


    # ======================================================
    # VARI
    # ======================================================
    #
    # VARI = (G - R) / (G + R - B)
    #
    # ======================================================

    vari_denominator = (
        Gv + Rv - Bv
    )

    valid_vari = (
        vari_denominator != 0
    )


    if np.any(valid_vari):

        vari_pixels = (
            (
                Gv[valid_vari]
                -
                Rv[valid_vari]
            )
            /
            vari_denominator[valid_vari]
        )

        VARI = np.mean(
            vari_pixels
        )

    else:

        VARI = np.nan


    # ======================================================
    # GREEN-DOMINANT PIXELS
    # ======================================================
    #
    # Pixel considered green dominant when:
    #
    # G > R
    # AND
    # G > B
    #
    # ======================================================

    green_dominant = (
        (Gv > Rv)
        &
        (Gv > Bv)
    )


    green_dominant_percent = (
        np.mean(green_dominant)
        * 100
    )


    # ======================================================
    # RETURN METRICS
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

        "ExG":
            ExG,

        "GRVI":
            GRVI,

        "VARI":
            VARI,

        "green_dominant_percent":
            green_dominant_percent,
    }


# ==========================================================
# FIND CLASS FOLDERS
# ==========================================================

class_folders = sorted([
    folder
    for folder in INPUT_DIR.iterdir()
    if folder.is_dir()
])


if not class_folders:

    raise ValueError(
        f"No class folders found in:\n{INPUT_DIR}"
    )


print("\nClass folders found:")

for folder in class_folders:
    print(
        f"  {folder.name}"
    )


# ==========================================================
# PROCESS IMAGES
# ==========================================================

results = []


for class_folder in class_folders:

    print("\n" + "=" * 60)

    print(
        f"Processing class: "
        f"{class_folder.name}"
    )

    print("=" * 60)


    image_files = sorted([
        path
        for path in class_folder.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ])


    print(
        f"Images found: "
        f"{len(image_files)}"
    )


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

            "class_folder":
                class_folder.name,

            **metrics,
        }


        results.append(
            row
        )


        if number % 100 == 0:

            print(
                f"  Processed "
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
# This attempts to find a date automatically in filenames.
#
# It handles common formats such as:
#
# 2024-06-15
# 2024_06_15
# 20240615
#
# If your filenames use another format, this section may
# need changing.
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


# First try YYYY-MM-DD

results_df["date"] = pd.to_datetime(
    date_string,
    format="%Y-%m-%d",
    errors="coerce"
)


# Try YYYYMMDD for anything that failed

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
    "class_folder",

    "valid_pixel_count",

    "mean_R",
    "mean_G",
    "mean_B",

    "brightness",

    "GCC",
    "RCC",
    "BCC",

    "ExG",
    "GRVI",
    "VARI",

    "green_dominant_percent",
]


results_df = results_df[
    column_order
]


# ==========================================================
# SORT
# ==========================================================

results_df = results_df.sort_values(
    [
        "class_folder",
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
    f"Classes processed: "
    f"{results_df['class_folder'].nunique()}"
)

print(
    f"Images without extracted date: "
    f"{results_df['date'].isna().sum()}"
)

print(
    f"\nCSV saved to:\n"
    f"{OUTPUT_CSV}"
)
