# ==========================================================
# GCC PHENOLOGY ANALYSIS
# Single dataset / single class / average annual cycle
#
# Calculates:
# - Minimum GCC
# - Maximum GCC
# - Seasonal amplitude
# - SOS (Start of Season)
# - Peak GCC date
# - EOS (End of Season)
# - LOS (Length of Season)
# - Green-up duration
# - Senescence duration
# - Maximum positive rate of change
# - Maximum negative rate of change
#
# SOS:
# GCCmin + p * (GCCmax - GCCmin)
#
# EOS:
# GCCmax - p * (GCCmax - GCCmin)
#
# ==========================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ==========================================================
# SETTINGS
# ==========================================================

CSV_FILE = r"path-to-greenness-csv.csv"

OUTPUT_DIR = Path(
    r"path-to-output-directory"
)

CLASS_NAME = "class_13_broadleaf_deciduous"
CLASS_LABEL = "Broadleaf Deciduous"

DATE_COL = "date"
CLASS_COL = "class_folder"
VALUE_COL = "GCC"


# ==========================================================
# PHENOLOGY THRESHOLD
# ==========================================================

# 20% threshold = p = 0.20

P_THRESHOLD = 0.30


# ==========================================================
# SAVITZKY-GOLAY SMOOTHING
# ==========================================================

APPLY_SAVGOL = True

SAVGOL_WINDOW = 65
SAVGOL_POLYORDER = 3


# ==========================================================
# ERROR BAND
# ==========================================================

ERROR_TYPE = "std"
# Options:
# "std"
# "sem"
# "none"

ERROR_ALPHA = 0.15


# ==========================================================
# FIGURE SETTINGS
# ==========================================================

FIG_WIDTH = 12
FIG_HEIGHT = 6
DPI = 300

LINE_COLOUR = "#08306B"
ERROR_COLOUR = "#08306B"

RAW_COLOUR = "grey"

LINE_WIDTH = 2.5

TITLE = "Cryptogam GCC phenology"

Y_LABEL = "Greenness Chromatic Coordinate (GCC)"
X_LABEL = "Month"

TITLE_SIZE = 20
AXIS_LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 12

SHOW_RAW_DAILY_MEAN = False
SHOW_ERROR_BAND = True

SHOW_THRESHOLD_LINES = True

SAVE_PNG = True
SAVE_PDF = True

SAVE_RESULTS_CSV = True
SAVE_CURVE_CSV = True

SHOW_FIGURE = True


# ==========================================================
# FUNCTIONS
# ==========================================================

def ensure_odd(window):

    if window % 2 == 0:
        window += 1

    return window


def smooth_curve(y):

    if not APPLY_SAVGOL:
        return y.copy()

    window = ensure_odd(SAVGOL_WINDOW)

    if window > len(y):

        window = (
            len(y)
            if len(y) % 2 == 1
            else len(y) - 1
        )

    if window <= SAVGOL_POLYORDER:

        raise ValueError(
            "Savitzky-Golay window must be larger "
            "than the polynomial order."
        )

    return savgol_filter(
        y,
        window_length=window,
        polyorder=SAVGOL_POLYORDER
    )


def doy_to_date(doy):
    """
    Convert annual-cycle day number to readable month/day.

    Uses 2021 as a non-leap reference year.
    """

    if pd.isna(doy):
        return None

    date = (
        pd.Timestamp("2021-01-01")
        + pd.Timedelta(days=float(doy) - 1)
    )

    return date.strftime("%d %b")


def interpolate_crossing(x1, y1, x2, y2, threshold):
    """
    Linear interpolation to estimate the exact point
    where the curve crosses a threshold.
    """

    if y2 == y1:
        return float(x1)

    fraction = (
        (threshold - y1)
        / (y2 - y1)
    )

    return (
        x1
        + fraction * (x2 - x1)
    )


def find_rising_crossing(x, y, threshold, end_index):
    """
    Find threshold crossing on rising limb before peak.
    """

    for i in range(1, end_index + 1):

        if (
            y[i - 1] < threshold
            and y[i] >= threshold
        ):

            return interpolate_crossing(
                x[i - 1],
                y[i - 1],
                x[i],
                y[i],
                threshold
            )

    return np.nan


def find_falling_crossing(x, y, threshold, start_index):
    """
    Find threshold crossing on falling limb after peak.
    """

    for i in range(start_index + 1, len(y)):

        if (
            y[i - 1] > threshold
            and y[i] <= threshold
        ):

            return interpolate_crossing(
                x[i - 1],
                y[i - 1],
                x[i],
                y[i],
                threshold
            )

    return np.nan


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(CSV_FILE)


for col in [
    DATE_COL,
    CLASS_COL,
    VALUE_COL
]:

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
        f"No data found for: {CLASS_NAME}"
    )


print(
    f"Number of GCC observations: {len(df)}"
)

print(
    f"Years represented: "
    f"{df[DATE_COL].dt.year.min()}–"
    f"{df[DATE_COL].dt.year.max()}"
)


# ==========================================================
# CREATE AVERAGE ANNUAL CYCLE
# ==========================================================
#
# IMPORTANT:
# Rather than using raw day-of-year, month/day is mapped
# onto a common NON-LEAP year (2021).
#
# This prevents leap years shifting March–December
# observations by one day.
# ==========================================================

# Remove February 29

df = df[
    ~(
        (df[DATE_COL].dt.month == 2)
        &
        (df[DATE_COL].dt.day == 29)
    )
].copy()


def convert_to_annual_day(date):

    reference_date = pd.Timestamp(
        year=2021,
        month=date.month,
        day=date.day
    )

    return reference_date.dayofyear


df["annual_day"] = (
    df[DATE_COL]
    .apply(convert_to_annual_day)
)


# ==========================================================
# DAILY STATISTICS ACROSS ALL YEARS
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


annual["std"] = (
    annual["std"]
    .fillna(0)
)


annual["sem"] = (
    annual["std"]
    / np.sqrt(annual["count"])
)


# ==========================================================
# COMPLETE 365-DAY SERIES
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
# INTERPOLATE MISSING DAILY MEANS
# ==========================================================
#
# This creates an evenly spaced daily time series.
#
# Therefore a Savitzky-Golay window of 61 represents
# 61 DAILY positions rather than 61 irregular observations.
# ==========================================================

annual["mean_interpolated"] = (
    annual["mean"]
    .interpolate(
        method="linear",
        limit_direction="both"
    )
)


# ==========================================================
# SMOOTH GCC
# ==========================================================

annual["GCC_smooth"] = smooth_curve(
    annual["mean_interpolated"].values
)


x = annual["annual_day"].values

y = annual["GCC_smooth"].values


# ==========================================================
# FIND MINIMUM AND MAXIMUM
# ==========================================================

gcc_min = np.min(y)

gcc_max = np.max(y)

amplitude = (
    gcc_max - gcc_min
)


peak_index = np.argmax(y)

peak_doy = float(
    x[peak_index]
)

peak_date = doy_to_date(
    peak_doy
)


# ==========================================================
# SOS THRESHOLD
# ==========================================================
#
# SOS = GCCmin + p(GCCmax - GCCmin)
# ==========================================================

sos_threshold = (
    gcc_min
    + P_THRESHOLD * amplitude
)


sos_doy = find_rising_crossing(
    x=x,
    y=y,
    threshold=sos_threshold,
    end_index=peak_index
)


sos_date = doy_to_date(
    sos_doy
)


# ==========================================================
# EOS THRESHOLD
# ==========================================================
#
# EOS = GCCmax - p(GCCmax - GCCmin)
#
# For p = 0.20:
# EOS occurs after the curve has fallen through
# 20% of the amplitude from the seasonal maximum.
# ==========================================================

eos_threshold = (
    gcc_max
    - P_THRESHOLD * amplitude
)


eos_doy = find_falling_crossing(
    x=x,
    y=y,
    threshold=eos_threshold,
    start_index=peak_index
)


eos_date = doy_to_date(
    eos_doy
)


# ==========================================================
# LENGTH OF SEASON
# ==========================================================

if (
    not np.isnan(sos_doy)
    and
    not np.isnan(eos_doy)
):

    los = (
        eos_doy
        - sos_doy
    )

else:

    los = np.nan


# ==========================================================
# GREEN-UP PHASE
#
# Time between SOS and peak GCC
# ==========================================================

if not np.isnan(sos_doy):

    greenup_duration = (
        peak_doy
        - sos_doy
    )

else:

    greenup_duration = np.nan


# ==========================================================
# SENESCENCE PHASE
#
# Time between peak GCC and EOS
# ==========================================================

if not np.isnan(eos_doy):

    senescence_duration = (
        eos_doy
        - peak_doy
    )

else:

    senescence_duration = np.nan


# ==========================================================
# RATE OF CHANGE
# ==========================================================
#
# Numerical first derivative of smoothed GCC.
#
# Units:
# GCC change per day
#
# NOTE:
# These numerical slopes are useful descriptive metrics,
# but are not the logistic c and d parameters from a
# fitted double-logistic model.
# ==========================================================

annual["GCC_rate_of_change"] = (
    np.gradient(
        annual["GCC_smooth"].values,
        annual["annual_day"].values
    )
)


rate = (
    annual["GCC_rate_of_change"]
    .values
)


# Maximum positive change

max_greenup_rate_index = (
    np.argmax(rate)
)

max_greenup_rate = (
    rate[max_greenup_rate_index]
)

max_greenup_rate_doy = float(
    x[max_greenup_rate_index]
)


# Maximum negative change

max_senescence_rate_index = (
    np.argmin(rate)
)

max_senescence_rate = (
    rate[max_senescence_rate_index]
)

max_senescence_rate_doy = float(
    x[max_senescence_rate_index]
)


# ==========================================================
# RESULTS
# ==========================================================

results = pd.DataFrame({

    "class": [
        CLASS_LABEL
    ],

    "p_threshold": [
        P_THRESHOLD
    ],

    "GCC_min": [
        gcc_min
    ],

    "GCC_max": [
        gcc_max
    ],

    "amplitude": [
        amplitude
    ],

    "SOS_threshold_GCC": [
        sos_threshold
    ],

    "SOS_DOY": [
        sos_doy
    ],

    "SOS_date": [
        sos_date
    ],

    "peak_DOY": [
        peak_doy
    ],

    "peak_date": [
        peak_date
    ],

    "peak_GCC": [
        gcc_max
    ],

    "EOS_threshold_GCC": [
        eos_threshold
    ],

    "EOS_DOY": [
        eos_doy
    ],

    "EOS_date": [
        eos_date
    ],

    "LOS_days": [
        los
    ],

    "greenup_duration_days": [
        greenup_duration
    ],

    "senescence_duration_days": [
        senescence_duration
    ],

    "maximum_greenup_rate": [
        max_greenup_rate
    ],

    "maximum_greenup_rate_DOY": [
        max_greenup_rate_doy
    ],

    "maximum_greenup_rate_date": [
        doy_to_date(
            max_greenup_rate_doy
        )
    ],

    "maximum_senescence_rate": [
        max_senescence_rate
    ],

    "maximum_senescence_rate_DOY": [
        max_senescence_rate_doy
    ],

    "maximum_senescence_rate_date": [
        doy_to_date(
            max_senescence_rate_doy
        )
    ],

    "savgol_window": [
        SAVGOL_WINDOW
    ],

    "savgol_polyorder": [
        SAVGOL_POLYORDER
    ],

})


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n")
print("=" * 60)
print("PHENOLOGY RESULTS")
print("=" * 60)

print(
    results.to_string(
        index=False
    )
)


# ==========================================================
# PLOT
# ==========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


fig, ax = plt.subplots(
    figsize=(
        FIG_WIDTH,
        FIG_HEIGHT
    )
)


# ----------------------------------------------------------
# ERROR BAND
# ----------------------------------------------------------

if (
    SHOW_ERROR_BAND
    and
    ERROR_TYPE != "none"
):

    if ERROR_TYPE == "std":

        error = (
            annual["std"]
        )

    elif ERROR_TYPE == "sem":

        error = (
            annual["sem"]
        )

    else:

        error = None


    if error is not None:

        valid_error = (
            annual["mean"].notna()
            &
            error.notna()
        )

        ax.fill_between(

            annual.loc[
                valid_error,
                "annual_day"
            ],

            annual.loc[
                valid_error,
                "mean"
            ]
            -
            error[valid_error],

            annual.loc[
                valid_error,
                "mean"
            ]
            +
            error[valid_error],

            color=ERROR_COLOUR,
            alpha=ERROR_ALPHA,
            linewidth=0
        )


# ----------------------------------------------------------
# RAW DAILY MEAN
# ----------------------------------------------------------

if SHOW_RAW_DAILY_MEAN:

    ax.plot(
        annual["annual_day"],
        annual["mean"],
        color=RAW_COLOUR,
        linewidth=1,
        alpha=0.5,
        label="Mean GCC"
    )


# ----------------------------------------------------------
# SMOOTHED GCC
# ----------------------------------------------------------

ax.plot(
    x,
    y,
    color=LINE_COLOUR,
    linewidth=LINE_WIDTH,
    label=CLASS_LABEL
)


# ==========================================================
# THRESHOLD LINES
# ==========================================================

if SHOW_THRESHOLD_LINES:

    ax.axhline(
        sos_threshold,
        linestyle=":",
        linewidth=1,
        alpha=0.7
    )

    ax.axhline(
        eos_threshold,
        linestyle=":",
        linewidth=1,
        alpha=0.7
    )


# ==========================================================
# PHENOLOGY DATES
# ==========================================================

if not np.isnan(sos_doy):

    ax.axvline(
        sos_doy,
        linestyle="--",
        linewidth=1.5,
        label=f"SOS ({sos_date})"
    )


ax.axvline(
    peak_doy,
    linestyle="-.",
    linewidth=1.5,
    label=f"Peak ({peak_date})"
)


if not np.isnan(eos_doy):

    ax.axvline(
        eos_doy,
        linestyle="--",
        linewidth=1.5,
        label=f"EOS ({eos_date})"
    )


# ==========================================================
# MONTH AXIS
# ==========================================================

month_starts = [
    1,
    32,
    60,
    91,
    121,
    152,
    182,
    213,
    244,
    274,
    305,
    335
]


month_labels = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
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
# FORMAT
# ==========================================================

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

safe_class = (
    CLASS_NAME
    .replace(" ", "_")
    .replace("/", "_")
)


output_base = (
    OUTPUT_DIR
    / f"{safe_class}_phenology_p{int(P_THRESHOLD * 100)}"
)


if SAVE_RESULTS_CSV:

    results.to_csv(
        f"{output_base}_metrics.csv",
        index=False
    )


if SAVE_CURVE_CSV:

    annual.to_csv(
        f"{output_base}_annual_curve.csv",
        index=False
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


if SHOW_FIGURE:

    plt.show()

else:

    plt.close(fig)


print("\nDone.")
print(
    f"Results saved to: {OUTPUT_DIR}"
)
