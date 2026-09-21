# ==========================================================
# GCC phenology timing from slope / rate of change
# Single dataset / single class
#
# Derives:
# - Green-up onset
# - Plateau onset
# - Peak GCC
# - Senescence onset
# ==========================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ==========================================================
# SETTINGS
# ==========================================================

CSV_FILE = r"path-to-greenness.csv"

OUTPUT_DIR = Path(
    r"path-to-output-directory"
)

CLASS_NAME = "class_13_broadleaf_deciduous"
CLASS_LABEL = "Broadleaf Deciduous"

DATE_COL = "date"
CLASS_COL = "class_folder"
VALUE_COL = "GCC"


# ----------------------------------------------------------
# Savitzky-Golay smoothing
# ----------------------------------------------------------

SAVGOL_WINDOW = 65
SAVGOL_POLYORDER = 3


# ==========================================================
# PHENOLOGY SETTINGS
# ==========================================================

# Minimum sustained positive slope for green-up onset.
# Units are GCC change per day.
GREENUP_SLOPE_THRESHOLD = 0.0005

# Plateau is identified when absolute slope becomes small.
PLATEAU_SLOPE_THRESHOLD = 0.0003

# Minimum sustained negative slope for senescence onset.
SENESCENCE_SLOPE_THRESHOLD = -0.0005

# Number of consecutive days required for a phase transition.
# This prevents a single noisy day triggering the date.
SUSTAINED_DAYS = 7

# Restrict searches to biologically sensible parts of the year.
# Adjust if needed for your ecosystem.
GREENUP_SEARCH_START = 40
GREENUP_SEARCH_END = 220

PLATEAU_SEARCH_START = 80
PLATEAU_SEARCH_END = 280

SENESCENCE_SEARCH_START = 180
SENESCENCE_SEARCH_END = 365


# ==========================================================
# FIGURE SETTINGS
# ==========================================================

FIG_WIDTH = 12
FIG_HEIGHT = 7
DPI = 300

LINE_COLOUR = "#08306B"
RAW_COLOUR = "grey"
DERIVATIVE_COLOUR = "black"

LINE_WIDTH = 2.5
RAW_ALPHA = 0.30

TITLE = "Annual GCC Phenology"
Y_LABEL = "Greenness Chromatic Coordinate (GCC)"

TITLE_SIZE = 18
AXIS_LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 12

SHOW_RAW_MEAN = True

SAVE_PNG = True
SAVE_PDF = True
SAVE_CSV = True
SHOW_FIGURE = True


# ==========================================================
# FUNCTIONS
# ==========================================================

def ensure_odd(window):
    if window % 2 == 0:
        window += 1
    return window


def smooth_curve(y):
    window = ensure_odd(SAVGOL_WINDOW)

    if window > len(y):
        window = len(y) if len(y) % 2 == 1 else len(y) - 1

    if window <= SAVGOL_POLYORDER:
        raise ValueError(
            "Savitzky-Golay window must be larger than polynomial order."
        )

    return savgol_filter(
        y,
        window_length=window,
        polyorder=SAVGOL_POLYORDER
    )


def sustained_true(mask, n_days):
    """
    Returns index of first position where condition is true
    for n_days consecutively.
    """
    mask = np.asarray(mask, dtype=bool)

    if len(mask) < n_days:
        return None

    run = np.convolve(
        mask.astype(int),
        np.ones(n_days, dtype=int),
        mode="valid"
    )

    hits = np.where(run == n_days)[0]

    if len(hits) == 0:
        return None

    return int(hits[0])


def doy_to_date(doy):
    if pd.isna(doy):
        return None

    ref = pd.Timestamp("2021-01-01")

    return (
        ref + pd.Timedelta(days=int(doy) - 1)
    ).strftime("%d %b")


def find_event(
    doy,
    slope,
    start_doy,
    end_doy,
    condition,
    sustained_days
):
    """
    Find first sustained occurrence of condition
    within a specified day-of-year interval.
    """

    search_mask = (
        (doy >= start_doy) &
        (doy <= end_doy)
    )

    search_doy = doy[search_mask]
    search_slope = slope[search_mask]

    if len(search_doy) == 0:
        return np.nan

    condition_mask = condition(search_slope)

    idx = sustained_true(
        condition_mask,
        sustained_days
    )

    if idx is None:
        return np.nan

    return search_doy[idx]


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(CSV_FILE)

for col in [DATE_COL, CLASS_COL, VALUE_COL]:
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
        CLASS_COL,
        VALUE_COL
    ]
)

df = df[
    df[CLASS_COL] == CLASS_NAME
].copy()

if df.empty:
    raise ValueError(
        f"No data found for {CLASS_NAME}"
    )


# ==========================================================
# CREATE AVERAGE ANNUAL CYCLE
# ==========================================================

df["day_of_year"] = df[DATE_COL].dt.dayofyear

annual = (
    df.groupby("day_of_year")[VALUE_COL]
    .mean()
    .reset_index()
    .sort_values("day_of_year")
)


# ==========================================================
# CREATE COMPLETE DAILY SERIES
# ==========================================================

complete_days = pd.DataFrame({
    "day_of_year": np.arange(1, 366)
})

annual = complete_days.merge(
    annual,
    on="day_of_year",
    how="left"
)

# Linear interpolation across missing daily values.
annual[VALUE_COL] = (
    annual[VALUE_COL]
    .interpolate(
        method="linear",
        limit_direction="both"
    )
)


# ==========================================================
# SMOOTH GCC
# ==========================================================

annual["GCC_smooth"] = smooth_curve(
    annual[VALUE_COL].values
)

doy = annual["day_of_year"].values
gcc_raw = annual[VALUE_COL].values
gcc_smooth = annual["GCC_smooth"].values


# ==========================================================
# CALCULATE DAILY RATE OF CHANGE
# ==========================================================

# Because x is now one value per day, np.gradient gives
# approximately GCC change per day.
slope = np.gradient(gcc_smooth)

annual["GCC_slope"] = slope


# ==========================================================
# GREEN-UP ONSET
# ==========================================================

greenup_doy = find_event(
    doy=doy,
    slope=slope,
    start_doy=GREENUP_SEARCH_START,
    end_doy=GREENUP_SEARCH_END,
    condition=lambda s: s >= GREENUP_SLOPE_THRESHOLD,
    sustained_days=SUSTAINED_DAYS
)


# ==========================================================
# PEAK GCC
# ==========================================================

peak_index = np.argmax(gcc_smooth)

peak_doy = doy[peak_index]
peak_gcc = gcc_smooth[peak_index]


# ==========================================================
# PLATEAU ONSET
# ==========================================================

# Search after green-up and before/around peak season.
if not np.isnan(greenup_doy):

    plateau_start_search = max(
        int(greenup_doy),
        PLATEAU_SEARCH_START
    )

else:

    plateau_start_search = PLATEAU_SEARCH_START


plateau_doy = find_event(
    doy=doy,
    slope=slope,
    start_doy=plateau_start_search,
    end_doy=PLATEAU_SEARCH_END,
    condition=lambda s: np.abs(s) <= PLATEAU_SLOPE_THRESHOLD,
    sustained_days=SUSTAINED_DAYS
)


# ==========================================================
# SENESCENCE ONSET
# ==========================================================

senescence_doy = find_event(
    doy=doy,
    slope=slope,
    start_doy=SENESCENCE_SEARCH_START,
    end_doy=SENESCENCE_SEARCH_END,
    condition=lambda s: s <= SENESCENCE_SLOPE_THRESHOLD,
    sustained_days=SUSTAINED_DAYS
)


# ==========================================================
# DERIVED DURATIONS
# ==========================================================

if not np.isnan(plateau_doy) and not np.isnan(senescence_doy):

    plateau_duration = (
        senescence_doy - plateau_doy
    )

else:

    plateau_duration = np.nan


if not np.isnan(greenup_doy) and not np.isnan(senescence_doy):

    active_season_duration = (
        senescence_doy - greenup_doy
    )

else:

    active_season_duration = np.nan


# ==========================================================
# RESULTS TABLE
# ==========================================================

results = pd.DataFrame({
    "class": [CLASS_LABEL],

    "greenup_DOY": [greenup_doy],
    "greenup_date": [
        doy_to_date(greenup_doy)
    ],

    "plateau_start_DOY": [plateau_doy],
    "plateau_start_date": [
        doy_to_date(plateau_doy)
    ],

    "peak_DOY": [peak_doy],
    "peak_date": [
        doy_to_date(peak_doy)
    ],
    "peak_GCC": [peak_gcc],

    "senescence_onset_DOY": [
        senescence_doy
    ],
    "senescence_onset_date": [
        doy_to_date(senescence_doy)
    ],

    "plateau_duration_days": [
        plateau_duration
    ],

    "active_season_duration_days": [
        active_season_duration
    ],

    "greenup_slope_threshold": [
        GREENUP_SLOPE_THRESHOLD
    ],

    "plateau_slope_threshold": [
        PLATEAU_SLOPE_THRESHOLD
    ],

    "senescence_slope_threshold": [
        SENESCENCE_SLOPE_THRESHOLD
    ],

    "sustained_days": [
        SUSTAINED_DAYS
    ],

    "savgol_window": [
        SAVGOL_WINDOW
    ],

    "savgol_polyorder": [
        SAVGOL_POLYORDER
    ],
})

print("\nPhenology results:")
print(results.to_string(index=False))


# ==========================================================
# PLOT
# ==========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT)
)


# Raw annual mean

if SHOW_RAW_MEAN:

    ax.plot(
        doy,
        gcc_raw,
        color=RAW_COLOUR,
        linewidth=1,
        alpha=RAW_ALPHA,
        label="Daily mean"
    )


# Smoothed GCC

ax.plot(
    doy,
    gcc_smooth,
    color=LINE_COLOUR,
    linewidth=LINE_WIDTH,
    label=CLASS_LABEL
)


# ==========================================================
# PHENOLOGY MARKERS
# ==========================================================

if not np.isnan(greenup_doy):

    ax.axvline(
        greenup_doy,
        linestyle="--",
        linewidth=1.5,
        label="Green-up onset"
    )


if not np.isnan(plateau_doy):

    ax.axvline(
        plateau_doy,
        linestyle=":",
        linewidth=1.5,
        label="Plateau onset"
    )


ax.axvline(
    peak_doy,
    linestyle="-.",
    linewidth=1.5,
    label="Peak GCC"
)


if not np.isnan(senescence_doy):

    ax.axvline(
        senescence_doy,
        linestyle="--",
        linewidth=1.5,
        label="Senescence onset"
    )


# ==========================================================
# MONTH AXIS
# ==========================================================

month_starts = [
    1, 32, 60, 91,
    121, 152, 182, 213,
    244, 274, 305, 335
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

ax.set_title(
    TITLE,
    fontsize=TITLE_SIZE
)

ax.set_xlabel(
    "Month",
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

ax.legend(
    fontsize=LEGEND_SIZE
)

ax.grid(
    True,
    alpha=0.3
)

fig.tight_layout()


# ==========================================================
# SAVE
# ==========================================================

outfile = (
    OUTPUT_DIR
    / f"{CLASS_NAME}_phenology_slope"
)

if SAVE_CSV:

    results.to_csv(
        f"{outfile}_dates.csv",
        index=False
    )

    annual.to_csv(
        f"{outfile}_daily_curve.csv",
        index=False
    )


if SAVE_PNG:

    fig.savefig(
        f"{outfile}.png",
        dpi=DPI,
        bbox_inches="tight"
    )


if SAVE_PDF:

    fig.savefig(
        f"{outfile}.pdf",
        bbox_inches="tight"
    )


if SHOW_FIGURE:

    plt.show()

else:

    plt.close(fig)


print("\nDone.")
print(f"Saved to: {OUTPUT_DIR}")
