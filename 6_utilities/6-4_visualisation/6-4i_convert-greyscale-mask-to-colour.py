
from pathlib import Path

import numpy as np
from PIL import Image


# =========================================================
# INPUT / OUTPUT
# =========================================================

INPUT_MASK = Path(r"path-to-input-mask.png")
OUTPUT_MASK = Path(r"path-to-output-mask.png")

# =========================================================
# CLASS MAP
# =========================================================

CLASS_MAP = {
    "background": 0,
    "sky": 1,
    "snow_ice": 2,
    "bare": 3,
    "ericaceous_shrub": 4,
    "briar_seedling": 5,
    "graminoid": 6,
    "forb": 7,
    "fern": 8,
    "cryptogam": 9,
    "vine": 10,
    "broadleaf_evergreen": 11,
    "conifer": 12,
    "broadleaf_deciduous": 13,
    "ignore": 255,
}


# =========================================================
# COLOURS
# =========================================================

CLASS_COLOURS = {
    "sky": "#56B4E9",                    # light blue
    "snow_ice": "#C8C8C8",               # light grey
    "bare": "#9D2C00",                   # dark rust/brown

    "cryptogam": "#7E4794",              # purple
    "graminoid": "#D4A72C",              # ochre/gold
    "forb": "#F0C571",                   # pale yellow
    "fern": "#2E7D32",                   # strong green
    "ericaceous_shrub": "#D95F5F",       # coral/red
    "vine": "#C44E9B",                   # magenta
    "briar_seedling": "#3F8FC4",         # medium blue

    "broadleaf_deciduous": "#7CB342",     # yellow-green
    "broadleaf_evergreen": "#00A6A6",     # teal/cyan
    "conifer": "#174A3F",                 # very dark green


    # Choose suitable display colours for these:
    "background": "#000000",
    "ignore": "#FFFFFF",
}


def hex_to_rgb(hex_colour):
    """Convert '#RRGGBB' to an (R, G, B) tuple."""
    hex_colour = hex_colour.lstrip("#")
    return tuple(
        int(hex_colour[i:i + 2], 16)
        for i in (0, 2, 4)
    )


def colourise_mask(input_path, output_path):
    # Read original class mask
    mask = np.array(Image.open(input_path))

    # Ensure mask is single channel
    if mask.ndim != 2:
        raise ValueError(
            f"Expected a single-channel mask, "
            f"but received shape {mask.shape}"
        )

    # Create empty RGB image
    rgb_mask = np.zeros(
        (mask.shape[0], mask.shape[1], 3),
        dtype=np.uint8
    )

    # Apply colour for each class
    for class_name, class_value in CLASS_MAP.items():

        colour = hex_to_rgb(
            CLASS_COLOURS[class_name]
        )

        rgb_mask[mask == class_value] = colour

    # Check for unexpected values
    expected_values = set(CLASS_MAP.values())
    actual_values = set(np.unique(mask))

    unexpected = actual_values - expected_values

    if unexpected:
        print(
            "Warning: unexpected class values found:",
            sorted(unexpected)
        )

    # Save RGB visualisation
    Image.fromarray(rgb_mask, mode="RGB").save(
        output_path
    )

    print(f"Saved colour mask to:\n{output_path}")


if __name__ == "__main__":
    colourise_mask(
        INPUT_MASK,
        OUTPUT_MASK
    )
