
# ============================================================
# RGB / ANNOTATION / PREDICTION COMPARISON
#
# Layout:
#
#                  SE-NM              BE-LE
#
# Original RGB     image              image
# Annotation       image              image
# Prediction       image              image
#
# Shared semantic-class legend underneath
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

# ------------------------------------------------------------
# SE-NM
# ------------------------------------------------------------

SE_NM_RGB = r"path-to-rgb-image-1.png"
SE_NM_ANNOTATION = r"path-to-annotation-mask-1.png""
SE_NM_PREDICTION = r"path-to-predicted-mask-1.png""


# ------------------------------------------------------------
# BE-LE
# ------------------------------------------------------------

BE_LE_RGB = r"\path-to-rgb-image-2.png"
BE_LE_ANNOTATION = r"path-to-annotation-mask-2.png"
BE_LE_PREDICTION = r"path-to-predicted-mask-2.png"


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_PATH = r"path-to-ouput-figure.png"

DPI = 600

FIGSIZE = (12, 12)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

# Show panel letters?
SHOW_PANEL_LETTERS = False

# Show row labels on left?
SHOW_ROW_LABELS = True

# Font sizes
SITE_TITLE_SIZE = 18
ROW_LABEL_SIZE = 18
PANEL_LETTER_SIZE = 18
LEGEND_FONT_SIZE = 18


# ============================================================
# SEMANTIC CLASSES
# ============================================================

# IMPORTANT:
# Change these colours to EXACTLY match the colours used
# in your annotation and prediction masks/overlays.

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
# FONT
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10


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
# IMAGE PATHS
# ============================================================

image_paths = [

    # Row 1 — Original RGB
    [
        SE_NM_RGB,
        BE_LE_RGB,
    ],

    # Row 2 — Annotation
    [
        SE_NM_ANNOTATION,
        BE_LE_ANNOTATION,
    ],

    # Row 3 — Prediction
    [
        SE_NM_PREDICTION,
        BE_LE_PREDICTION,
    ],
]


# ============================================================
# LABELS
# ============================================================

column_titles = [
    "SE-NM_84",
    "BE-LE_1087",
]

row_labels = [
    "Original RGB",
    "Annotation",
    "Prediction",
]

#panel_letters = [
#    ["a", "b"],
#    ["c", "d"],
#    ["e", "f"],
#]


# ============================================================
# CREATE FIGURE
# ============================================================

# Load all six images first
images = [
    [
        load_image(SE_NM_RGB),
        load_image(BE_LE_RGB),
    ],
    [
        load_image(SE_NM_ANNOTATION),
        load_image(BE_LE_ANNOTATION),
    ],
    [
        load_image(SE_NM_PREDICTION),
        load_image(BE_LE_PREDICTION),
    ],
]


# ============================================================
# CALCULATE COLUMN WIDTHS
# ============================================================

# Use RGB image dimensions to determine the natural
# width required for each column when image heights are equal.

se_width, se_height = images[0][0].size
be_width, be_height = images[0][1].size

se_aspect = se_width / se_height
be_aspect = be_width / be_height

print(f"SE-NM dimensions: {se_width} x {se_height}")
print(f"BE-LE dimensions: {be_width} x {be_height}")

print(f"SE-NM aspect ratio: {se_aspect:.3f}")
print(f"BE-LE aspect ratio: {be_aspect:.3f}")


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=FIGSIZE
)

gs = fig.add_gridspec(
    nrows=3,
    ncols=2,

    # IMPORTANT:
    # columns now have widths proportional to their
    # original image aspect ratios
    width_ratios=[
        se_aspect,
        be_aspect,
    ],

    # Equal row heights
    height_ratios=[
        1,
        1,
        1,
    ],

    # Small gaps
    wspace=0.03,
    hspace=0.03,
)


axes = np.empty(
    (3, 2),
    dtype=object
)


# ============================================================
# PLOT SIX IMAGES
# ============================================================

for row in range(3):

    for col in range(2):

        ax = fig.add_subplot(
            gs[row, col]
        )

        axes[row, col] = ax

        image = images[row][col]

        # ----------------------------------------------------
        # Plot ONCE
        # ----------------------------------------------------

        ax.imshow(
            image,
            aspect="equal"
        )

        # ----------------------------------------------------
        # No ticks
        # ----------------------------------------------------

        ax.set_xticks([])
        ax.set_yticks([])

        # ----------------------------------------------------
        # Make axes fit image exactly
        # ----------------------------------------------------

        width, height = image.size

        ax.set_xlim(
            -0.5,
            width - 0.5
        )

        ax.set_ylim(
            height - 0.5,
            -0.5
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        # ----------------------------------------------------
        # Centre image
        # ----------------------------------------------------

        ax.set_anchor("C")

        # ----------------------------------------------------
        # Border
        # ----------------------------------------------------

        for spine in ax.spines.values():

            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("black")


# ============================================================
# COLUMN TITLES
# ============================================================

for col, title in enumerate(column_titles):

    axes[0, col].set_title(
        title,
        fontsize=SITE_TITLE_SIZE,
        fontweight="bold",
        pad=8,
    )


# ============================================================
# ROW LABELS
# ============================================================

if SHOW_ROW_LABELS:

    for row, label in enumerate(row_labels):

        axes[row, 0].set_ylabel(
            label,
            fontsize=ROW_LABEL_SIZE,
            fontweight="bold",
            rotation=90,
            labelpad=10,
        )


# ============================================================
# COLUMN TITLES
# ============================================================

for col, title in enumerate(column_titles):

    axes[0, col].set_title(
        title,
        fontsize=SITE_TITLE_SIZE,
        fontweight="bold",
        pad=10,
    )


# ============================================================
# ROW LABELS
# ============================================================

if SHOW_ROW_LABELS:

    for row, label in enumerate(row_labels):

        axes[row, 0].set_ylabel(
            label,
            fontsize=ROW_LABEL_SIZE,
            fontweight="bold",
            rotation=90,
            labelpad=12,
        )


# ============================================================
# LEGEND
# ============================================================

legend_handles = [

    Patch(
        facecolor=colour,
        edgecolor="black",
        linewidth=0.4,
        label=class_name,
    )

    for class_name, colour in CLASS_COLOURS.items()
]


fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=5,
    frameon=False,
    fontsize=LEGEND_FONT_SIZE,
    bbox_to_anchor=(0.5, 0.015),
)


# ============================================================
# SPACING
# ============================================================

plt.subplots_adjust(
    left=0.10,
    right=0.98,
    top=0.95,
    bottom=0.13,
    wspace=0.05,
    hspace=0.08,
)


# ============================================================
# SAVE
# ============================================================

output = Path(OUTPUT_PATH)

output.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    output,
    dpi=DPI,
    bbox_inches="tight",
)

print()
print("=" * 60)
print("Figure complete")
print(f"Saved to: {output}")
print("=" * 60)

plt.show()
