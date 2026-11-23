
# ==========================================================
# OVERALL GREENNESS - MEAN ANNUAL GCC CURVE
# ==========================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter


# ==========================================================
# SETTINGS
# ==========================================================

CSV_FILE = Path(
    r"path-to-whole-image-greenness.csv"
)

OUTPUT_DIR = Path(
    r"path-to-output-directory")

DATE_COL = "date"
VALUE_COL = "GCC"

# ==========================================================
# SMOOTHING
# ==========================================================

APPLY_SAVGOL = True

SAVGOL_WINDOW = 75
SAVGOL_POLYORDER = 3


# ==========================================================
# ERROR BAND
# ==========================================================

SHOW_ERROR_BARS = True

ERROR_TYPE = "std"
# Options: "std", "sem", "none"

ERROR_ALPHA = 0.15


# ==========================================================
# FIGURE SETTINGS
# ==========================================================

FIG_WIDTH = 12
FIG_HEIGHT = 9
DPI = 300

LINE_COLOUR = "#0000a2"
LINE_WIDTH = 2.5

FONT_SIZE = 18
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 18
TICK_SIZE = 18
LEGEND_SIZE = 18

TITLE = "Mean annual greenness curve"

LEGEND_LABEL = "Overall greenness"

Y_LABEL = "Greenness Chromatic Coordinate (GCC)"
X_LABEL = "Month"

SHOW_TITLE = False
SHOW_LEGEND = True
SHOW_RAW_MEAN = False
SHOW_POINTS = False

SAVE_PNG = True
SAVE_PDF = True
SHOW_FIGURE = True


# ==========================================================
# FONT SETTINGS
# ==========================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})


# ==========================================================
# LOAD CSV
# ==========================================================

df = pd.read_csv(CSV_FILE)

required_cols = [
    DATE_COL,
    VALUE_COL,
]

for col in required_cols:

    if col not in df.columns:

        raise ValueError(
            f"Missing required column: {col}"
        )


df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    errors="coerce"
)

df[VALUE_COL] = pd.to_numeric(
    df[VALUE_COL],
    errors="coerce"
)

df = df.dropna(
    subset=[
        DATE_COL,
        VALUE_COL
    ]
)

if df.empty:

    raise ValueError(
        "No valid GCC observations found."
    )


print(
    f"Total GCC observations: {len(df)}"
)

print(
    f"Years represented: "
    f"{df[DATE_COL].dt.year.min()} - "
    f"{df[DATE_COL].dt.year.max()}"
)


# ==========================================================
# CREATE ANNUAL CYCLE
# ==========================================================
#
# Remove February 29 and map all dates onto a common
# non-leap year to avoid shifting dates after February.
# ==========================================================

df = df[
    ~(
        (df[DATE_COL].dt.month == 2)
        &
        (df[DATE_COL].dt.day == 29)
    )
].copy()


reference_dates = pd.to_datetime({

    "year": 2021,

    "month": df[DATE_COL].dt.month,

    "day": df[DATE_COL].dt.day,

})


df["annual_day"] = (
    reference_dates.dt.dayofyear
)


# ==========================================================
# CALCULATE MEAN GCC BY CALENDAR DAY
# ==========================================================

annual = (

    df.groupby("annual_day")[VALUE_COL]

    .agg(
        mean="mean",
        std="std",
        count="count"
    )

    .reset_index()

    .sort_values("annual_day")
)


annual["sem"] = (
    annual["std"]
    /
    np.sqrt(annual["count"])
)


# ==========================================================
# CREATE COMPLETE DAILY TIME SERIES
# ==========================================================
#
# This ensures that the Savitzky-Golay window represents
# calendar days rather than irregular observations.
# ==========================================================

complete_days = pd.DataFrame({

    "annual_day": np.arange(1, 366)

})


annual = complete_days.merge(

    annual,

    on="annual_day",

    how="left"
)


# ==========================================================
# INTERPOLATE MISSING DAILY GCC VALUES
# ==========================================================

annual["mean_interpolated"] = (

    annual["mean"]

    .interpolate(
        method="linear",
        limit_direction="both"
    )
)


# ==========================================================
# SAVITZKY-GOLAY SMOOTHING
# ==========================================================

x = annual["annual_day"].to_numpy()

y = annual["mean_interpolated"].to_numpy()


if APPLY_SAVGOL:

    window = SAVGOL_WINDOW

    # Ensure odd window length

    if window % 2 == 0:

        window += 1

    if window > len(y):

        window = (
            len(y)
            if len(y) % 2 == 1
            else len(y) - 1
        )

    if window <= SAVGOL_POLYORDER:

        raise ValueError(
            "Smoothing window must be larger "
            "than the polynomial order."
        )

    y_smooth = savgol_filter(

        y,

        window_length=window,

        polyorder=SAVGOL_POLYORDER
    )

else:

    y_smooth = y.copy()


annual["GCC_smooth"] = y_smooth


# ==========================================================
# ERROR TYPE
# ==========================================================

if ERROR_TYPE == "std":

    error = annual["std"].to_numpy()

elif ERROR_TYPE == "sem":

    error = annual["sem"].to_numpy()

elif ERROR_TYPE == "none":

    error = np.zeros(len(annual))

else:

    raise ValueError(
        "ERROR_TYPE must be 'std', 'sem', or 'none'."
    )


# ==========================================================
# CREATE FIGURE
# ==========================================================

fig, ax = plt.subplots(

    figsize=(
        FIG_WIDTH,
        FIG_HEIGHT
    )
)


# ==========================================================
# ERROR BAND
# ==========================================================

if SHOW_ERROR_BARS and ERROR_TYPE != "none":

    raw_mean = annual["mean"].to_numpy()

    ax.fill_between(

        x,

        raw_mean - error,

        raw_mean + error,

        color=LINE_COLOUR,

        alpha=ERROR_ALPHA,

        linewidth=0,

    )


# ==========================================================
# OPTIONAL RAW MEAN
# ==========================================================

if SHOW_RAW_MEAN:

    ax.plot(

        x,

        annual["mean"],

        color="grey",

        linewidth=1,

        alpha=0.5,

        label="Raw daily mean",

    )


# ==========================================================
# OPTIONAL POINTS
# ==========================================================

if SHOW_POINTS:

    ax.plot(

        x,

        annual["mean"],

        marker="o",

        markersize=3,

        linestyle="none",

        color=LINE_COLOUR,

        alpha=0.5,

    )


# ==========================================================
# MAIN GCC CURVE
# ==========================================================

ax.plot(

    x,

    y_smooth,

    color=LINE_COLOUR,

    linestyle="-",

    linewidth=LINE_WIDTH,

    label=LEGEND_LABEL,

)


# ==========================================================
# MONTH AXIS
# ==========================================================

month_starts = [

    1, 32, 60, 91, 121, 152,
    182, 213, 244, 274, 305, 335

]

month_labels = [

    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"

]


ax.set_xticks(
    month_starts
)

ax.set_xticklabels(
    month_labels
)

ax.set_xlim(
    1,
    365
)


# ==========================================================
# FORMATTING
# ==========================================================

if SHOW_TITLE:

    ax.set_title(
        TITLE,
        fontsize=TITLE_SIZE
    )


ax.set_xlabel(
    X_LABEL,
    fontsize=AXIS_LABEL_SIZE
)

ax.set_ylabel(
    Y_LABEL,
    fontsize=AXIS_LABEL_SIZE
)

ax.tick_params(
    axis="both",
    labelsize=TICK_SIZE
)

ax.grid(
    True,
    alpha=0.3
)

if SHOW_LEGEND:

    ax.legend(
        fontsize=LEGEND_SIZE,
        loc="upper right"
    )


fig.tight_layout()


# ==========================================================
# SAVE OUTPUT
# ==========================================================

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True
)


output_base = (

    OUTPUT_DIR

    /

    f"{CSV_FILE.stem}_mean_annual_curve"

)


if SAVE_PNG:

    fig.savefig(

        f"{output_base}.png",

        dpi=DPI,

        bbox_inches="tight"

    )


if SAVE_PDF:

    fig.savefig(

        f"{output_base}.pdf",

        bbox_inches="tight"

    )


# ==========================================================
# SHOW FIGURE
# ==========================================================

if SHOW_FIGURE:

    plt.show()

else:

    plt.close(fig)


print("\nDone.")

print(
    f"Figures saved to:\n{OUTPUT_DIR}"
)
