
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Use Arial throughout the figure
plt.rcParams["font.family"] = "Arial"
# =========================================================
# OUTPUT
# =========================================================

OUTPUT_PATH = Path(r"path-to-legend.png")


# =========================================================
# CLASS COLOURS
# =========================================================

CLASS_COLOURS = {
    "background": "#000000",
    "sky": "#9BD5EF",
    "snow_ice": "#c8c8c8",
    "bare": "#9d2c00",
    "ericaceous_shrub": "#e25759",
    "briar_seedling": "#59a89c",
    "graminoid": "#A6A83A",
    "forb": "#f0c571",
    "fern": "#26734D",
    "cryptogam": "#7E4794",
    "vine": "#D65A9E",
    "broadleaf_evergreen": "#257A70",
    "conifer": "#174A3F",
    "broadleaf_deciduous": "#79C267",
    "ignore": "#FFFFFF",
}


# =========================================================
# DISPLAY NAMES
# =========================================================

CLASS_LABELS = {
    "background": "Background",
    "sky": "Sky",
    "snow_ice": "Snow and Ice",
    "bare": "Bare",
    "ericaceous_shrub": "Ericaceous Shrub",
    "briar_seedling": "Briar/Seedling",
    "graminoid": "Graminoid",
    "forb": "Forb",
    "fern": "Fern",
    "cryptogam": "Cryptogam",
    "vine": "Vine",
    "broadleaf_evergreen": "Broadleaf Evergreen",
    "conifer": "Conifer",
    "broadleaf_deciduous": "Broadleaf Deciduous",
    "ignore": "Ignore",
}


# =========================================================
# CLASSES TO INCLUDE
# =========================================================
# Add/remove/reorder classes here.
# Only classes in this list will appear in the legend.

INCLUDE_CLASSES = [
    "sky",
    "snow_ice",
     "bare",
    "ericaceous_shrub",
   # "briar_seedling",
    "graminoid",
  # "forb",
   # "fern",
    "cryptogam",
   # "vine",
    "broadleaf_deciduous",
    "broadleaf_evergreen",
    "conifer",
    "ignore"
]

# Example for a smaller legend:
#
# INCLUDE_CLASSES = [
#     "bare",
#     "ericaceous_shrub",
#     "broadleaf_deciduous",
#     "conifer",
# ]


# =========================================================
# LEGEND SETTINGS
# =========================================================

N_COLUMNS = 1
FONT_SIZE = 16

# Increase/decrease these to alter spacing
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 2.5


# =========================================================
# CREATE LEGEND
# =========================================================

def create_legend():

    handles = []

    for class_name in INCLUDE_CLASSES:

        if class_name not in CLASS_COLOURS:
            raise ValueError(
                f"No colour defined for '{class_name}'"
            )

        if class_name not in CLASS_LABELS:
            raise ValueError(
                f"No display label defined for '{class_name}'"
            )

        handles.append(
            Patch(
                facecolor=CLASS_COLOURS[class_name],
                edgecolor="black",
                label=CLASS_LABELS[class_name]
            )
        )

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
    )

    ax.axis("off")

    ax.legend(
        handles=handles,
        loc="center",
        ncol=N_COLUMNS,
        frameon=False,
        fontsize=FONT_SIZE
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        transparent=True
    )

    plt.close(fig)

    print(f"Saved legend to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    create_legend()
