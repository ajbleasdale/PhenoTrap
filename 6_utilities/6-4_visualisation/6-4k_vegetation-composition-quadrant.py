
# ============================================================
# Camera-trap vegetation composition figure
#
# Creates a 2 x 2 figure containing:
#   - Camera-trap image
#   - Vertical 100% stacked vegetation composition bar
#   - Shannon diversity index (H')
#
# The stacked bar and H' are calculated from the SAME
# vegetation-only proportional composition.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from PIL import Image


# ============================================================
# USER SETTINGS
# ============================================================

# Two pixel-summary CSV files
PIXEL_SUMMARY_CSVS = [
    r"path-to-pixel-summary-1",
    r"path-to-pixel-summary-2",
]

# Folders containing the original camera-trap images.
# Add as many folders as required.
IMAGE_DIRS = [
    r"G:\Datasets\Sweden\SE-NM",
    r"G:\Datasets\Belgium\BE-LE",
]

# Four images to display.
# They can come from either CSV / image directory.

#SELECTED_IMAGES = [
#    r"G:\Datasets\Sweden\SE-NM\SE-NM_1_2017-06-23_120000.JPG",
#    r"G:\Datasets\Belgium\BE-LE\BE-LE_729_2022-06-23_120000.jpg",
#    r"G:\Datasets\Sweden\SE-NM\SE-NM_115_2017-06-23_120000.JPG",
#    r"G:\Datasets\Belgium\BE-LE\BE-LE_717_2022-06-23_120000.jpg",
#]

#SELECTED_IMAGES = [
#    r"G:\Datasets\Sweden\SE-NM\SE-NM_174_2017-06-05_120000.JPG",
#    r"G:\Datasets\Belgium\BE-LE\BE-LE_961_2023-09-12_120000.png",
#    r"G:\Datasets\Sweden\SE-NM\SE-NM_251_2017-03-14_120000.JPG",
#    r"G:\Datasets\Belgium\BE-LE\BE-LE_1118_2024-07-24_120000.png",
#]
#

SELECTED_IMAGES = [
    r"G:\Datasets\Sweden\SE-NM\SE-NM_113_2017-04-12_120000.JPG",
    r"G:\Datasets\Belgium\BE-LE\BE-LE_138_2019-04-02_120000.jpg",
    r"G:\Datasets\Sweden\SE-NM\SE-NM_113_2017-06-15_120000.JPG",
    r"G:\Datasets\Belgium\BE-LE\BE-LE_138_2019-05-09_120000.jpg",
]



# Output
OUTPUT_PATH = r"path-to-output-figure.png"

# Figure size
FIGSIZE = (12, 9)

# Output resolution
DPI = 600


# ============================================================
# VEGETATION CLASSES
# ============================================================

# These are the classes included in BOTH:
#   1. the stacked composition bar
#   2. Shannon H'

VEGETATION_CLASSES = [
    "ericaceous_shrub",
    "briar_seedling",
    "graminoid",
    "forb",
    "fern",
    "cryptogam",
    "vine",
    "broadleaf_evergreen",
    "conifer",
    "broadleaf_deciduous",
]


# ============================================================
# CLASS DISPLAY NAMES
# ============================================================

CLASS_LABELS = {
    "ericaceous_shrub": "Ericaceous shrub",
    "briar_seedling": "Briar / seedling",
    "graminoid": "Graminoid",
    "forb": "Forb",
    "fern": "Fern",
    "cryptogam": "Cryptogam",
    "vine": "Vine",
    "broadleaf_evergreen": "Broadleaf evergreen",
    "conifer": "Conifer",
    "broadleaf_deciduous": "Broadleaf deciduous",
}


# ============================================================
# CLASS COLOURS
# ============================================================

# Change these if you already have standard segmentation
# colours that you want to use throughout the paper.

CLASS_COLOURS = {
    "sky": "#A3D6F3",                    # light blue
    "snow_ice": "#C8C8C8",               # light grey
    "bare": "#9D2C00",                   # dark rust/brown
    "cryptogam": "#7E4794",              # purple
    "graminoid": "#D4A72C",              # ochre/gold
    "forb": "#F0C571",                   # pale yellow
    "fern": "#2E7D32",                   # strong green
    "ericaceous_shrub": "#D95F5F",       # coral/red
    "vine": "#C44E9B",                   # magenta
    "briar_seedling": "#285F84",         # medium blue
    "broadleaf_deciduous": "#7CB342",     # yellow-green
    "broadleaf_evergreen": "#00D7D2",     # teal/cyan
    "conifer": "#174A3F",                 # very dark green
}



# ============================================================
# FONT SETTINGS
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.linewidth"] = 0.8


# ============================================================
# LOAD PIXEL SUMMARY DATA
# ============================================================

def load_pixel_summaries(csv_paths):

    frames = []

    for csv_path in csv_paths:

        csv_path = Path(csv_path)

        print(f"Loading: {csv_path}")

        df = pd.read_csv(csv_path)

        # Record where each row came from
        df["_source_csv"] = str(csv_path)

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    print()
    print(f"Total rows loaded: {len(combined):,}")

    return combined


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(filename, image_dirs):
    """
    Find the original image regardless of file extension.

    Example:
        CSV:    SE-NM_1_2017-03-26_120000.png
        Folder: SE-NM_1_2017-03-26_120000.JPG

    These will be treated as the same image.
    """

    target_stem = Path(filename).stem.lower()

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
    }

    for directory in image_dirs:

        directory = Path(directory)

        # Search recursively through directory/subdirectories
        for path in directory.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in image_extensions:
                continue

            if path.stem.lower() == target_stem:
                return path

    raise FileNotFoundError(
        f"Could not find an image matching:\n"
        f"{filename}\n"
        f"(searched by filename without extension)"
    )

# ============================================================
# GET VEGETATION COMPOSITION
# ============================================================

def get_vegetation_composition(row):

    values = []

    for vegetation_class in VEGETATION_CLASSES:

        column = f"{vegetation_class}_percent"

        if column not in row.index:
            raise KeyError(
                f"Missing column: {column}"
            )

        value = row[column]

        # Treat missing values as zero
        if pd.isna(value):
            value = 0

        values.append(float(value))

    values = np.array(values, dtype=float)

    total_vegetation = values.sum()

    if total_vegetation <= 0:
        return np.zeros_like(values)

    # --------------------------------------------------------
    # Renormalise vegetation to 100%
    # --------------------------------------------------------

    proportions = values / total_vegetation

    return proportions


# ============================================================
# SHANNON DIVERSITY
# ============================================================

def calculate_shannon(proportions):

    # Remove zero classes because ln(0) is undefined
    p = proportions[proportions > 0]

    if len(p) == 0:
        return np.nan

    H = -np.sum(
        p * np.log(p)
    )

    return H

# ============================================================
# GET IMAGE ASPECT RATIO
# ============================================================

def get_image_aspect(image_path):
    """
    Return image width / height.
    """

    with Image.open(image_path) as img:
        width, height = img.size

    return width / height


# ============================================================
# PLOT ONE PANEL
# ============================================================

def plot_panel(
    fig,
    subplot_spec,
    image_path,
    proportions,
    shannon,
    panel_letter
):

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    with Image.open(image_path) as img:
        image = np.array(
            img.convert("RGB")
        )

    image_height, image_width = image.shape[:2]


    # ========================================================
    # SUB-GRID
    #
    # IMAGE | STACKED BAR
    # ========================================================

    inner = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=subplot_spec,
        width_ratios=[10, 1],
        wspace=0.05,
    )

    ax_image = fig.add_subplot(inner[0])
    ax_bar = fig.add_subplot(inner[1])


    # ========================================================
    # IMAGE
    # ========================================================

    ax_image.imshow(
        image,
        aspect="equal"
    )

    # Use complete image extent
    ax_image.set_xlim(
        -0.5,
        image_width - 0.5
    )

    ax_image.set_ylim(
        image_height - 0.5,
        -0.5
    )

    # Preserve original aspect ratio
    ax_image.set_aspect(
        "equal",
        adjustable="box"
    )

    # Centre image in available space
    ax_image.set_anchor("C")

    ax_image.set_xticks([])
    ax_image.set_yticks([])


    # ========================================================
    # IMAGE FILENAME
    # ========================================================

    image_name = Path(image_path).stem

    ax_image.text(
        0.5,
        -0.04,
        image_name,
        transform=ax_image.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        clip_on=False,
    )


    # ========================================================
    # IMAGE BORDER
    # ========================================================

    for spine in ax_image.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")


    # ========================================================
    # PANEL LETTER
    # ========================================================

    ax_image.text(
        0.01,
        0.99,
        panel_letter,
        transform=ax_image.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            pad=2,
        ),
    )


    # ========================================================
    # STACKED BAR
    # ========================================================

    bottom = 0

    for vegetation_class, proportion in zip(
        VEGETATION_CLASSES,
        proportions
    ):

        percent = proportion * 100

        if percent <= 0:
            continue

        ax_bar.bar(
            0,
            percent,
            bottom=bottom,
            width=0.65,
            color=CLASS_COLOURS[vegetation_class],
            edgecolor="white",
            linewidth=0.5,
        )

        bottom += percent


    # ========================================================
    # BAR AXIS
    # ========================================================

    ax_bar.set_ylim(0, 100)
    ax_bar.set_xlim(-0.6, 0.6)

    ax_bar.set_xticks([])

    ax_bar.set_yticks(
        [0, 25, 50, 75, 100]
    )

    ax_bar.set_yticklabels(
        ["0", "25", "50", "75", "100"]
    )

    ax_bar.yaxis.tick_right()

    ax_bar.tick_params(
        axis="y",
        labelsize=10,
        length=3,
    )

    ax_bar.set_ylabel(
        "Vegetation composition (%)",
        fontsize=12,
        rotation=270,
        labelpad=12,
    )

    ax_bar.yaxis.set_label_position(
        "right"
    )


    # ========================================================
    # SHANNON H'
    # ========================================================

    if np.isnan(shannon):
        h_text = "H′ = NA"
    else:
        h_text = f"H′ = {shannon:.2f}"

    ax_bar.text(
        0.5,
        -0.05,
        h_text,
        transform=ax_bar.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )


    # ========================================================
    # CLEAN BAR SPINES
    # ========================================================

    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["bottom"].set_visible(False)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD CSV FILES
    # ========================================================

    data = load_pixel_summaries(
        PIXEL_SUMMARY_CSVS
    )

    if "filename" not in data.columns:
        raise KeyError(
            "The CSV files must contain a 'filename' column."
        )


    # ========================================================
    # FIND THE FOUR IMAGE FILES
    # ========================================================

    resolved_image_paths = [
        find_image(
            filename,
            IMAGE_DIRS
        )
        for filename in SELECTED_IMAGES
    ]


    # ========================================================
    # CALCULATE IMAGE ASPECT RATIOS
    # ========================================================

    aspect_ratios = [
        get_image_aspect(path)
        for path in resolved_image_paths
    ]

    print()
    print("Image aspect ratios:")

    for path, aspect in zip(
        resolved_image_paths,
        aspect_ratios
    ):

        print(
            f"{Path(path).name}: "
            f"{aspect:.3f}"
        )


    # ========================================================
    # CALCULATE COLUMN WIDTHS
    #
    # Images 0 and 2 = left column
    # Images 1 and 3 = right column
    #
    # Widths are based on image aspect ratios so that the
    # displayed image HEIGHTS remain comparable.
    # ========================================================

    left_aspect = np.mean([
        aspect_ratios[0],
        aspect_ratios[2],
    ])

    right_aspect = np.mean([
        aspect_ratios[1],
        aspect_ratios[3],
    ])

    print()
    print(
        f"Left column aspect: "
        f"{left_aspect:.3f}"
    )

    print(
        f"Right column aspect: "
        f"{right_aspect:.3f}"
    )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=FIGSIZE
    )


    # ========================================================
    # OUTER 2 x 2 GRID
    # ========================================================

    outer = GridSpec(
        2,
        2,
        figure=fig,

        # Allow the two columns to have different widths
        # while preserving equal displayed image heights
        width_ratios=[
            left_aspect,
            right_aspect,
        ],

        height_ratios=[
            1,
            1,
        ],

        wspace=0.18,
        hspace=0.16,
    )


    # ========================================================
    # PANEL LETTERS
    # ========================================================

    panel_letters = [
        "A",
        "B",
        "C",
        "D",
    ]


    # ========================================================
    # PREPARE CSV FILENAME STEMS ONCE
    # ========================================================

    csv_stems = (
        data["filename"]
        .astype(str)
        .apply(
            lambda x: Path(x).stem.lower()
        )
    )


    # ========================================================
    # PROCESS FOUR IMAGES
    # ========================================================

    for i, filename in enumerate(
        SELECTED_IMAGES
    ):

        print()
        print("=" * 60)
        print(f"Processing: {filename}")


        # ----------------------------------------------------
        # Match selected image to CSV by filename stem
        # ----------------------------------------------------

        selected_name = Path(filename).name

        target_stem = (
            Path(selected_name)
            .stem
            .lower()
        )

        matches = data[
            csv_stems == target_stem
        ]


        # ----------------------------------------------------
        # Check match
        # ----------------------------------------------------

        if len(matches) == 0:

            print()
            print(
                "Could not find matching CSV row."
            )

            print(
                f"Selected image: {filename}"
            )

            print(
                f"Looking for stem: {target_stem}"
            )

            raise ValueError(
                f"Could not find '{target_stem}' "
                f"in either pixel-summary CSV."
            )


        if len(matches) > 1:

            print(
                f"WARNING: {target_stem} occurs "
                f"{len(matches)} times."
            )

            print(
                "Using first matching row."
            )


        row = matches.iloc[0]


        # ----------------------------------------------------
        # Actual image path
        # ----------------------------------------------------

        image_path = (
            resolved_image_paths[i]
        )

        print(
            f"Image: {image_path}"
        )

        print(
            f"CSV: {row['_source_csv']}"
        )


        # ----------------------------------------------------
        # Vegetation composition
        # ----------------------------------------------------

        proportions = (
            get_vegetation_composition(
                row
            )
        )


        # ----------------------------------------------------
        # Shannon H'
        # ----------------------------------------------------

        shannon = calculate_shannon(
            proportions
        )

        print(
            f"H' = {shannon:.4f}"
        )


        # ----------------------------------------------------
        # Plot panel
        # ----------------------------------------------------

        plot_panel(
            fig=fig,
            subplot_spec=outer[i],
            image_path=image_path,
            proportions=proportions,
            shannon=shannon,
            panel_letter=panel_letters[i],
        )


    # ========================================================
    # SHARED LEGEND
    # ========================================================

    legend_handles = [

        Patch(
            facecolor=CLASS_COLOURS[c],
            edgecolor="none",
            label=CLASS_LABELS[c],
        )

        for c in VEGETATION_CLASSES
    ]


    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(
            0.5,
            0.01
        ),
    )


    # ========================================================
    # FINAL LAYOUT
    # ========================================================

    plt.subplots_adjust(
        left=0.03,
        right=0.96,
        top=0.98,
        bottom=0.12,
    )


    # ========================================================
    # SAVE
    # ========================================================

    output = Path(
        OUTPUT_PATH
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )


    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 60)
    print("Figure complete")
    print("Saved to:")
    print(output)
    print("=" * 60)

    plt.show()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
