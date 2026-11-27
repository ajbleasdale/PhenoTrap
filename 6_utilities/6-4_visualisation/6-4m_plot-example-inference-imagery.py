
# ============================================================
# CNN SEGMENTATION EXAMPLE FIGURE
#
# 6 rows:
#   - 3 SE-NM examples
#   - 3 BE-LE examples
#
# Columns:
#   1. Original camera-trap image
#   2. CNN segmentation / annotation
#   3. FWIoU and mIoU values
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np
from matplotlib.patches import Patch

# ============================================================
# USER SETTINGS
# ============================================================

OUTPUT_PATH = (
    r"path-to-output.png"
)

DPI = 600

FIGSIZE = (12, 16)


# ============================================================
# IMAGE PAIRS
#
# Replace these paths and metric values with your own.
# ============================================================

EXAMPLES = [

    # --------------------------------------------------------
    # SE-NM
    # --------------------------------------------------------

    {
        "dataset": "SE-NM",
        "rgb": r"\path-to-input-rgb-image-1.jpg",
        "cnn": r"path-to-predicted-mask-1.png",
        "fwiou": 0.976, #change all these to true results
        "miou": 0.717,
    },

    {
        "dataset": "SE-NM",
        "rgb": r"path-to-input-rgb-image-2.jpg",
        "cnn": r"path-to-predicted-mask-2.png",
        "fwiou": 0.792,
        "miou": 0.438,
    },

    {
        "dataset": "SE-NM",
        "rgb": r"\path-to-input-rgb-image-3.jpg",



        "cnn": r"path-to-predicted-mask-3.png",
        "fwiou": 0.408,
        "miou": 0.277,
    },


    # --------------------------------------------------------
    # BE-LE
    # --------------------------------------------------------

    {
        "dataset": "BE-LE",
        "rgb": r"\path-to-input-rgb-image-4.jpg",


        "cnn": r"path-to-predicted-mask-4.png",
        "fwiou": 0.961,
        "miou": 0.710,
    },

    {
        "dataset": "BE-LE",
        "rgb": r"\path-to-input-rgb-image-5.jpg",



        "cnn": r"path-to-predicted-mask-5.png",
        "fwiou": 0.722,
        "miou": 0.441,
    },

    {
        "dataset": "BE-LE",
        "rgb": r"\path-to-input-rgb-image-6.jpg",


        "cnn": r"path-to-predicted-mask-6.png",
        "fwiou": 0.712,
        "miou": 0.223,
    },
]

# ============================================================
# SEGMENTATION CLASSES
# ============================================================

SEGMENTATION_CLASSES = [
    "sky",
    "snow_ice",
    "bare",
    "cryptogam",
    "graminoid",
    "forb",
    "fern",
    "ericaceous_shrub",
    "vine",
    "briar_seedling",
    "broadleaf_deciduous",
    "broadleaf_evergreen",
    "conifer",
]


# ============================================================
# CLASS COLOURS
# ============================================================

CLASS_COLOURS = {
    "sky": "#9BD5EF",
    "snow_ice": "#c8c8c8",
    "bare": "#9d2c00",
    "cryptogam": "#7E4794",
    "graminoid": "#A6A83A",
    "forb": "#f0c571",
    "fern": "#26734D",
    "ericaceous_shrub": "#e25759",
    "vine": "#D65A9E",
    "briar_seedling": "#59a89c",
    "broadleaf_deciduous": "#79C267",
    "broadleaf_evergreen": "#257A70",
    "conifer": "#174A3F",
}


# ============================================================
# CLASS LABELS
# ============================================================

CLASS_LABELS = {
    "sky": "Sky",
    "snow_ice": "Snow / ice",
    "bare": "Bare",
    "cryptogam": "Cryptogam",
    "graminoid": "Graminoid",
    "forb": "Forb",
    "fern": "Fern",
    "ericaceous_shrub": "Ericaceous shrub",
    "vine": "Vine",
    "briar_seedling": "Briar / seedling",
    "broadleaf_deciduous": "Broadleaf deciduous",
    "broadleaf_evergreen": "Broadleaf evergreen",
    "conifer": "Conifer",
}

# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find image:\n{path}"
        )

    with Image.open(path) as img:
        return img.convert("RGB").copy()


# ============================================================
# FORMAT IMAGE AXIS
# ============================================================

def format_image_axis(ax):

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD ALL IMAGES
    # ========================================================

    loaded_examples = []

    for example in EXAMPLES:

        rgb = load_image(
            example["rgb"]
        )

        cnn = load_image(
            example["cnn"]
        )

        loaded_examples.append({
            **example,
            "rgb_image": rgb,
            "cnn_image": cnn,
        })


    # ========================================================
    # CHECK IMAGE DIMENSIONS
    # ========================================================

    print()
    print("Image dimensions:")
    print()

    for example in loaded_examples:

        rgb_width, rgb_height = (
            example["rgb_image"].size
        )

        cnn_width, cnn_height = (
            example["cnn_image"].size
        )

        print(
            f"{example['dataset']}: "
            f"RGB {rgb_width} x {rgb_height} | "
            f"CNN {cnn_width} x {cnn_height}"
        )


    # ========================================================
    # CALCULATE ROW HEIGHTS
    #
    # All images will have the SAME displayed width.
    #
    # Therefore:
    #
    # displayed height ∝ image height / image width
    #
    # Each row receives a height based on its image's
    # natural height-to-width ratio.
    # ========================================================

    row_height_ratios = []

    for example in loaded_examples:

        rgb_width, rgb_height = (
            example["rgb_image"].size
        )

        cnn_width, cnn_height = (
            example["cnn_image"].size
        )

        rgb_ratio = (
            rgb_height / rgb_width
        )

        cnn_ratio = (
            cnn_height / cnn_width
        )

        # Normally RGB and CNN should have the same
        # dimensions/aspect ratio.
        #
        # Use the larger ratio so neither image is clipped.
        row_ratio = max(
            rgb_ratio,
            cnn_ratio
        )

        row_height_ratios.append(
            row_ratio
        )


    print()
    print("Row height ratios:")

    for i, ratio in enumerate(
        row_height_ratios,
        start=1
    ):
        print(
            f"Row {i}: {ratio:.3f}"
        )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=FIGSIZE
    )


    # ========================================================
    # GRID
    #
    # Column 1 = Original
    # Column 2 = CNN
    # Column 3 = Performance
    #
    # Columns 1 and 2 are EXACTLY the same width.
    #
    # Row heights vary according to image aspect ratio.
    # ========================================================

    gs = GridSpec(
        nrows=len(loaded_examples),
        ncols=3,
        figure=fig,

        width_ratios=[
            1,
            1,
            0.28,
        ],

        height_ratios=row_height_ratios,

        # Horizontal space between columns
        wspace=0.08,

        # Vertical space between rows
        hspace=0.12,
    )


    # ========================================================
    # STORE AXES
    # ========================================================

    rgb_axes = []
    cnn_axes = []
    metric_axes = []


    # ========================================================
    # PLOT ROWS
    # ========================================================

    for row, example in enumerate(
        loaded_examples
    ):

        ax_rgb = fig.add_subplot(
            gs[row, 0]
        )

        ax_cnn = fig.add_subplot(
            gs[row, 1]
        )

        ax_metrics = fig.add_subplot(
            gs[row, 2]
        )

        rgb_axes.append(
            ax_rgb
        )

        cnn_axes.append(
            ax_cnn
        )

        metric_axes.append(
            ax_metrics
        )


        # ====================================================
        # ORIGINAL RGB IMAGE
        # ====================================================

        rgb_image = (
            example["rgb_image"]
        )

        rgb_width, rgb_height = (
            rgb_image.size
        )

        ax_rgb.imshow(
            rgb_image,
            aspect="auto",
        )

        # Fill the complete subplot width and height.
        #
        # Because the GridSpec row height was calculated
        # from the original image aspect ratio, the image
        # retains its correct proportions.
        ax_rgb.set_xlim(
            -0.5,
            rgb_width - 0.5
        )

        ax_rgb.set_ylim(
            rgb_height - 0.5,
            -0.5
        )

        format_image_axis(
            ax_rgb
        )


        # ====================================================
        # CNN SEGMENTATION
        # ====================================================

        cnn_image = (
            example["cnn_image"]
        )

        cnn_width, cnn_height = (
            cnn_image.size
        )

        ax_cnn.imshow(
            cnn_image,
            aspect="auto",
        )

        ax_cnn.set_xlim(
            -0.5,
            cnn_width - 0.5
        )

        ax_cnn.set_ylim(
            cnn_height - 0.5,
            -0.5
        )

        format_image_axis(
            ax_cnn
        )

        # ====================================================
        # IMAGE NAME
        # ====================================================

        image_name = Path(
            example["rgb"]
        ).stem

        ax_rgb.text(
            -0.035,
            0.5,
            image_name,
            transform=ax_rgb.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
            rotation=90,
        )

        # ====================================================
        # PERFORMANCE METRICS
        # ====================================================

        ax_metrics.axis(
            "off"
        )

        metric_text = (
            f"FWIoU = {example['fwiou']:.3f}\n"
            f"mIoU = {example['miou']:.3f}"
        )

        ax_metrics.text(
            0.02,
            0.5,
            metric_text,
            transform=ax_metrics.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            linespacing=1.5,
        )


    # ========================================================
    # COLUMN TITLES
    # ========================================================

    rgb_axes[0].set_title(
        "Original image",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    cnn_axes[0].set_title(
        "CNN segmentation",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    metric_axes[0].set_title(
        "Performance",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )


    # ========================================================
    # SHARED CLASS LEGEND
    # ========================================================

    legend_handles = [

        Patch(
            facecolor=CLASS_COLOURS[c],
            edgecolor="none",
            label=CLASS_LABELS[c],
        )

        for c in SEGMENTATION_CLASSES
    ]


    # ========================================================
    # LEGEND UNDER ENTIRE FIGURE
    # ========================================================

    fig.legend(
        handles=legend_handles,
        loc="lower center",

        # Change this if you want more/fewer items per row
        ncol=5,

        frameon=False,
        fontsize=12,

        bbox_to_anchor=(
            0.5,
            0.015
        ),

        columnspacing=1.4,
        handletextpad=0.5,
    )


    # ========================================================
    # FINAL LAYOUT
    # ========================================================

    plt.subplots_adjust(
        left=0.07,
        right=0.97,
        top=0.96,

        # Space for legend
        bottom=0.11,
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
