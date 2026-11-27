
# ============================================================
# 8-IMAGE FIGURE
#
# 4 columns x 2 rows
# Same displayed image width
# Natural image heights retained
# Top row = A
# Bottom row = B
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


# ============================================================
# USER SETTINGS
# ============================================================

IMAGES = [

    # -------------------------
    # ROW A
    # -------------------------
    r"path-to-image-1.jpg",
    r"path-to-image-2.jpg",
    r"path-to-image-3.jpg",
    r"path-to-image-4.jpg",

    # -------------------------
    # ROW B
    # -------------------------
    r"path-to-image-5.jpg",
    r"path-to-image-6.jpg",
    r"path-to-image-7.jpg",
    r"path-to-image-8.jpg",
]


OUTPUT_PATH = (
    r"path-to-output-figure.png"
)

FIGSIZE = (14, 8)

DPI = 600


# ============================================================
# FONT SETTINGS
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8


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

    # --------------------------------------------------------
    # Check number of images
    # --------------------------------------------------------

    if len(IMAGES) != 8:
        raise ValueError(
            "Exactly 8 images are required."
        )


    # ========================================================
    # LOAD ALL IMAGES
    # ========================================================

    loaded_images = [
        load_image(path)
        for path in IMAGES
    ]


    # ========================================================
    # PRINT IMAGE DIMENSIONS
    # ========================================================

    print()
    print("Image dimensions:")
    print()

    for path, image in zip(
        IMAGES,
        loaded_images
    ):

        width, height = image.size

        print(
            f"{Path(path).name}: "
            f"{width} x {height}"
        )


    # ========================================================
    # CALCULATE ROW HEIGHTS
    #
    # All images have the same displayed WIDTH.
    #
    # Height therefore depends on:
    #
    # image height / image width
    #
    # One row height must accommodate all four images.
    # ========================================================

    row_a_ratios = [
        image.height / image.width
        for image in loaded_images[0:4]
    ]

    row_b_ratios = [
        image.height / image.width
        for image in loaded_images[4:8]
    ]

    row_a_height = max(
        row_a_ratios
    )

    row_b_height = max(
        row_b_ratios
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
    # Four EXACTLY equal-width columns.
    #
    # Row heights can differ.
    # ========================================================

    gs = GridSpec(
        nrows=2,
        ncols=4,
        figure=fig,

        width_ratios=[
            1,
            1,
            1,
            1,
        ],

        height_ratios=[
            row_a_height,
            row_b_height,
        ],

        wspace=0.01,
        hspace=0.01,
    )


    # ========================================================
    # PLOT IMAGES
    # ========================================================

    axes = []

    for i, image in enumerate(
        loaded_images
    ):

        row = i // 4
        col = i % 4

        ax = fig.add_subplot(
            gs[row, col]
        )

        axes.append(
            ax
        )

        # ----------------------------------------------------
        # Plot image
        # ----------------------------------------------------

        ax.imshow(
            image,
            aspect="equal",
        )

        # ----------------------------------------------------
        # Force SAME displayed width while preserving the
        # natural image height-to-width ratio.
        # ----------------------------------------------------

        ax.set_box_aspect(
            image.height / image.width
        )

        # Pull the two rows towards each other
        if row == 0:
            ax.set_anchor("S")  # top row anchored downward
        else:
            ax.set_anchor("N")  # bottom row anchored upward

        format_image_axis(
            ax
        )

    # ========================================================
    # ROW LABEL A — INSIDE FIRST IMAGE
    # ========================================================

    axes[0].text(
        0.05,
        0.95,
        "A",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="black",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            pad=4,
        ),
    )

    # ========================================================
    # ROW LABEL B — INSIDE FIRST IMAGE
    # ========================================================

    axes[4].text(
        0.05,
        0.95,
        "B",
        transform=axes[4].transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="black",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            pad=4,
        ),
    )

    # ========================================================
    # FINAL LAYOUT
    # ========================================================

    plt.subplots_adjust(
        left=0.04,
        right=0.99,
        top=0.98,
        bottom=0.02,
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
