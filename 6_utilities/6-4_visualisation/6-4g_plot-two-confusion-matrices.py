import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# CONFIG
# =============================================================================

# Input files
EXCEL_PATH_1 = r"path-to-confusion-matrix-1.xlsx"
EXCEL_PATH_2 = r"path-to-confusion-matrix-2.xlsx"


SHEET_NAME_1 = 0
SHEET_NAME_2 = 0

# Names shown above each confusion matrix
PANEL_1_TITLE = "(b) SE-NM"
PANEL_2_TITLE = "(a) BE-LE"

# Column names
GT_COL = "ground_truth_class"
ROW_TOTAL_COL = "row_total_gt_pixels"
PRED_PREFIX = "pred_"

# Choose:
# "percent" = percentage confusion matrix
# "count"   = estimated pixel counts
HEATMAP_MODE = "percent"


# =============================================================================
# OUTPUT
# =============================================================================

SHOW_FIGURE = True
SAVE_HTML = False
SAVE_PNG = True

OUTPUT_HTML = r"path-to-output-confusion-matrix.html"

OUTPUT_PNG = r"path-to-output-confusion-matrix.png"


# =============================================================================
# FIGURE SETTINGS
# =============================================================================

FIG_WIDTH = 2000
FIG_HEIGHT = 900

FONT_FAMILY = "Arial"

TITLE_SIZE = 24
PANEL_TITLE_SIZE = 24
AXIS_TITLE_SIZE = 22
TICK_SIZE = 18
CELL_TEXT_SIZE = 18
COLOURBAR_TITLE_SIZE = 18
COLOURBAR_TICK_SIZE = 18

# Space between confusion matrices
HORIZONTAL_SPACING = 0.12


# =============================================================================
# COLOUR SCALE
# =============================================================================

COLOUR_SCALE = "ylgnbu_r"

# Because these are percentages, force both matrices to use the same scale.
PERCENT_MIN = 0
PERCENT_MAX = 100

# Colour used where no ground-truth pixels exist
NO_GT_COLOUR = "lightgrey"


# =============================================================================
# LABEL OPTIONS
# =============================================================================

REMOVE_GT_PREFIX = True
REMOVE_PRED_PREFIX = True


# =============================================================================
# FUNCTION TO LOAD AND BUILD ONE CONFUSION MATRIX
# =============================================================================

def build_confusion_matrix(excel_path, sheet_name=0):

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    df = pd.read_excel(
        excel_path,
        sheet_name=sheet_name
    )

    # Check required columns
    required_cols = [
        GT_COL,
        ROW_TOTAL_COL
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in:\n{excel_path}"
            )

    # Find prediction columns
    pred_cols = [
        c for c in df.columns
        if c.startswith(PRED_PREFIX)
    ]

    if not pred_cols:
        raise ValueError(
            f"No columns beginning with '{PRED_PREFIX}' found in:\n"
            f"{excel_path}"
        )

    # Keep only required columns
    work_df = df[
        [GT_COL, ROW_TOTAL_COL] + pred_cols
    ].copy()

    # Remove rows without GT class labels
    work_df = work_df.dropna(
        subset=[GT_COL]
    )

    # -------------------------------------------------------------------------
    # Convert values to numeric
    # -------------------------------------------------------------------------

    pred_percent = (
        work_df[pred_cols]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
    )

    row_totals = (
        pd.to_numeric(
            work_df[ROW_TOTAL_COL],
            errors="coerce"
        )
        .fillna(0)
    )

    # -------------------------------------------------------------------------
    # Build matrix
    # -------------------------------------------------------------------------

    if HEATMAP_MODE.lower() == "percent":

        matrix = pred_percent.copy()

    elif HEATMAP_MODE.lower() == "count":

        matrix = (
            pred_percent
            .div(100.0)
            .mul(
                row_totals,
                axis=0
            )
        )

        matrix = matrix.round()

    else:

        raise ValueError(
            "HEATMAP_MODE must be either "
            "'percent' or 'count'"
        )

    # -------------------------------------------------------------------------
    # Grey out rows with NO ground-truth pixels
    # -------------------------------------------------------------------------

    no_gt_mask = row_totals == 0



    # -------------------------------------------------------------------------
    # Row labels
    # -------------------------------------------------------------------------

    row_labels = (
        work_df[GT_COL]
        .astype(str)
        .tolist()
    )

    if REMOVE_GT_PREFIX:

        row_labels = [
            x.replace(
                "gt_",
                "",
                1
            )
            if x.startswith("gt_")
            else x
            for x in row_labels
        ]

    # -------------------------------------------------------------------------
    # Column labels
    # -------------------------------------------------------------------------

    col_labels = pred_cols.copy()

    if REMOVE_PRED_PREFIX:

        col_labels = [
            x.replace(
                "pred_",
                "",
                1
            )
            if x.startswith("pred_")
            else x
            for x in col_labels
        ]

    # Assign labels
    matrix.index = row_labels
    matrix.columns = col_labels

    return matrix, no_gt_mask



# =============================================================================
# BUILD BOTH MATRICES
# =============================================================================

matrix_1, no_gt_mask_1 = build_confusion_matrix(
    EXCEL_PATH_1,
    SHEET_NAME_1
)

matrix_2, no_gt_mask_2 = build_confusion_matrix(
    EXCEL_PATH_2,
    SHEET_NAME_2
)

# =============================================================================
# CHECK CLASS ORDER
# =============================================================================

# For a direct comparison, ideally both matrices should contain
# the same classes in the same order.

if list(matrix_1.columns) != list(matrix_2.columns):

    print(
        "WARNING: Prediction class order differs "
        "between the two files."
    )

if list(matrix_1.index) != list(matrix_2.index):

    print(
        "WARNING: Ground-truth class order differs "
        "between the two files."
    )


# =============================================================================
# CREATE SUBPLOTS
# =============================================================================

fig = make_subplots(
    rows=1,
    cols=2,

    subplot_titles=(
        PANEL_1_TITLE,
        PANEL_2_TITLE
    ),

    horizontal_spacing=HORIZONTAL_SPACING
)


# =============================================================================
# MATRIX 1
# =============================================================================

fig.add_trace(

    go.Heatmap(

        z=matrix_1.values,

        x=matrix_1.columns,
        y=matrix_1.index,

        text=matrix_1.values,

        texttemplate=(
            "%{text:.1f}"
            if HEATMAP_MODE == "percent"
            else "%{text:.0f}"
        ),

        textfont=dict(
            family=FONT_FAMILY,
            size=CELL_TEXT_SIZE
        ),

        coloraxis="coloraxis",

        hovertemplate=(
            "Ground truth: %{y}<br>"
            "Predicted: %{x}<br>"
            "Value: %{z:.1f}"
            "<extra></extra>"
        ),

        showscale=False

    ),

    row=1,
    col=1
)

# Grey mask: 1 = no GT data, NaN = normal cell
grey_mask_1 = np.full(matrix_1.shape, np.nan)

grey_mask_1[no_gt_mask_1, :] = 1

fig.add_trace(
    go.Heatmap(
        z=grey_mask_1,
        x=matrix_1.columns,
        y=matrix_1.index,

        colorscale=[
            [0, "lightgrey"],
            [1, "lightgrey"]
        ],

        showscale=False,

        # No text in grey cells
        text=None,

        hoverinfo="skip"
    ),
    row=1,
    col=1
)

# =============================================================================
# MATRIX 2
# =============================================================================

fig.add_trace(

    go.Heatmap(

        z=matrix_2.values,

        x=matrix_2.columns,
        y=matrix_2.index,

        text=matrix_2.values,

        texttemplate=(
            "%{text:.1f}"
            if HEATMAP_MODE == "percent"
            else "%{text:.0f}"
        ),

        textfont=dict(
            family=FONT_FAMILY,
            size=CELL_TEXT_SIZE
        ),

        coloraxis="coloraxis",

        hovertemplate=(
            "Ground truth: %{y}<br>"
            "Predicted: %{x}<br>"
            "Value: %{z:.1f}"
            "<extra></extra>"
        ),

        showscale=False

    ),

    row=1,
    col=2
)

grey_mask_2 = np.full(matrix_2.shape, np.nan)

grey_mask_2[no_gt_mask_2, :] = 1

fig.add_trace(
    go.Heatmap(
        z=grey_mask_2,
        x=matrix_2.columns,
        y=matrix_2.index,

        colorscale=[
            [0, "lightgrey"],
            [1, "lightgrey"]
        ],

        showscale=False,
        text=None,
        hoverinfo="skip"
    ),
    row=1,
    col=2
)

# =============================================================================
# SHARED COLOUR SCALE
# =============================================================================

if HEATMAP_MODE.lower() == "percent":

    fig.update_layout(

        coloraxis=dict(

            colorscale=COLOUR_SCALE,

            cmin=PERCENT_MIN,
            cmax=PERCENT_MAX,

            colorbar=dict(

                title=dict(
                    text="Percent",
                    font=dict(
                        size=COLOURBAR_TITLE_SIZE,
                        family=FONT_FAMILY
                    )
                ),

                tickfont=dict(
                    size=COLOURBAR_TICK_SIZE,
                    family=FONT_FAMILY
                ),

                ticksuffix="%",

                thickness=25,
                len=0.8
            )
        )
    )

else:

    # Shared scale based on both count matrices

    combined_max = np.nanmax([
        np.nanmax(matrix_1.values),
        np.nanmax(matrix_2.values)
    ])

    fig.update_layout(

        coloraxis=dict(

            colorscale=COLOUR_SCALE,

            cmin=0,
            cmax=combined_max,

            colorbar=dict(

                title=dict(
                    text="Pixel count",
                    font=dict(
                        size=COLOURBAR_TITLE_SIZE,
                        family=FONT_FAMILY
                    )
                ),

                tickfont=dict(
                    size=COLOURBAR_TICK_SIZE,
                    family=FONT_FAMILY
                ),

                thickness=25,
                len=0.8
            )
        )
    )


# =============================================================================
# GENERAL FIGURE FORMATTING
# =============================================================================

fig.update_layout(

    width=FIG_WIDTH,
    height=FIG_HEIGHT,

    font=dict(
        family=FONT_FAMILY,
        size=TICK_SIZE
    ),

    # NaN cells appear as this colour
    plot_bgcolor=NO_GT_COLOUR,

    margin=dict(
        l=100,
        r=130,
        t=80,
        b=160
    )
)


# =============================================================================
# SUBPLOT TITLES
# =============================================================================

for annotation in fig.layout.annotations:

    annotation.font = dict(
        family=FONT_FAMILY,
        size=PANEL_TITLE_SIZE
    )


# =============================================================================
# X AXES
# =============================================================================

# Remove individual x-axis titles
fig.update_xaxes(title_text=None)

# Add one shared x-axis title
fig.add_annotation(
    text="Predicted class",
    x=0.5,
    y=-0.20,
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        family=FONT_FAMILY,
        size=AXIS_TITLE_SIZE
    )
),


# =============================================================================
# Y AXES
# =============================================================================

fig.update_yaxes(

    title_font=dict(
        family=FONT_FAMILY,
        size=AXIS_TITLE_SIZE
    ),

    tickfont=dict(
        family=FONT_FAMILY,
        size=TICK_SIZE
    ),

    autorange="reversed"
)


# Add ground-truth label to both panels
fig.update_yaxes(
    title_text="Ground truth class",
    row=1,
    col=1
)

fig.update_yaxes(
    title_text="None",
    row=1,
    col=2
)


# =============================================================================
# SAVE HTML
# =============================================================================

if SAVE_HTML:

    outdir = os.path.dirname(
        OUTPUT_HTML
    )

    if outdir:
        os.makedirs(
            outdir,
            exist_ok=True
        )

    fig.write_html(
        OUTPUT_HTML
    )

    print(
        f"Saved HTML:\n{OUTPUT_HTML}"
    )

# =============================================================================
# FINAL AXIS FORMATTING
# =============================================================================

# Remove individual x-axis titles but keep vertical tick labels
fig.update_xaxes(
    title_text=None,
    tickangle=-90,
    tickfont=dict(
        family=FONT_FAMILY,
        size=TICK_SIZE
    )
)

# Only show the y-axis title on the left matrix
fig.update_yaxes(
    title_text="Ground truth class",
    row=1,
    col=1
)

fig.update_yaxes(
    title_text=None,
    row=1,
    col=2
)

# One shared x-axis title
fig.add_annotation(
    text="Predicted class",
    x=0.5,
    y=-0.20,
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(
        family=FONT_FAMILY,
        size=AXIS_TITLE_SIZE
    )
)

# =============================================================================
# SAVE PNG
# =============================================================================

if SAVE_PNG:

    outdir = os.path.dirname(
        OUTPUT_PNG
    )

    if outdir:
        os.makedirs(
            outdir,
            exist_ok=True
        )

    fig.write_image(
        OUTPUT_PNG,
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        scale=2
    )

    print(
        f"Saved PNG:\n{OUTPUT_PNG}"
    )


# =============================================================================
# SHOW
# =============================================================================

if SHOW_FIGURE:
    fig.show()
