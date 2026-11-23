
# ==========================================================
# Savitzky-Golay sensitivity analysis
# Broadleaf deciduous GCC annual cycle
#
# Tests multiple smoothing windows and:
#   1. Plots all smoothed curves against the unsmoothed data
#   2. Saves a separate sensitivity figure for each dataset
#   3. Saves all smoothed values to CSV
# ==========================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ==========================================================
# SETTINGS
# ==========================================================

DATASETS = [
    {
        "name": "SE-NM",
        "csv": r"path-to-dataset-1-greenness-metrics.csv",
    },
    {
        "name": "BE-LE",
        "csv": r"path-to-dataset-2-greenness-metrics.csv",
    },
]

OUTPUT_DIR = Path(
    r"path-to-output-directory/Savgol_sensitivity"
)

DATE_COL = "date"
CLASS_COL = "class_folder"
VALUE_COL = "GCC"

SELECTED_CLASS = "class_13_broadleaf_deciduous"

# Windows to compare
WINDOWS = [31, 45, 61, 75, 91]

POLYORDER = 3

FIG_WIDTH = 12
FIG_HEIGHT = 9
DPI = 300

FONT_SIZE = 16
LINE_WIDTH = 2

SAVE_PNG = True
SAVE_PDF = True
SHOW_FIGURES = True


# ==========================================================
# FONT SETTINGS
# ==========================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 2,
})


# ==========================================================
# FUNCTIONS
# ==========================================================

def ensure_odd_window(window):
    """Savitzky-Golay window length must be odd."""
    if window % 2 == 0:
        window += 1
    return window


def safe_savgol(y, window, polyorder):
    """
    Apply Savitzky-Golay smoothing while ensuring that
    window length is valid for the available observations.
    """
    y = np.asarray(y, dtype=float)

    if len(y) < 3:
        return y

    window = ensure_odd_window(window)

    if window > len(y):
        window = len(y) if len(y) % 2 == 1 else len(y) - 1

    if window <= polyorder or window < 3:
        return y

    return savgol_filter(
        y,
        window_length=window,
        polyorder=polyorder
    )


def load_and_aggregate(dataset):
    """
    Load GCC data and calculate mean GCC for each day of year.

    This reproduces the annual-cycle aggregation used in the
    original plotting script.
    """
    df = pd.read_csv(dataset["csv"])

    required_cols = [DATE_COL, CLASS_COL, VALUE_COL]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"{dataset['name']}: missing required column '{col}'"
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
        subset=[DATE_COL, CLASS_COL, VALUE_COL]
    )

    df = df[
        df[CLASS_COL] == SELECTED_CLASS
    ].copy()

    if df.empty:
        raise ValueError(
            f"No {SELECTED_CLASS} data found for "
            f"{dataset['name']}."
        )

    # Convert calendar date to day of year
    df["day_of_year"] = df[DATE_COL].dt.dayofyear

    # Mean GCC for each day of year
    daily = (
        df.groupby("day_of_year")[VALUE_COL]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("day_of_year")
    )

    daily["std"] = daily["std"].fillna(0)

    return daily


# ==========================================================
# MAIN
# ==========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

all_results = []


for dataset in DATASETS:

    dataset_name = dataset["name"]

    print(f"\nProcessing {dataset_name}...")

    daily = load_and_aggregate(dataset)

    x = daily["day_of_year"].to_numpy()
    y = daily["mean"].to_numpy()

    # ------------------------------------------------------
    # Calculate all smoothing windows
    # ------------------------------------------------------

    results = pd.DataFrame({
        "dataset": dataset_name,
        "day_of_year": x,
        "GCC_raw": y,
    })

    for window in WINDOWS:

        y_smooth = safe_savgol(
            y,
            window,
            POLYORDER
        )

        results[f"GCC_window_{window}"] = y_smooth

    all_results.append(results)


    # ======================================================
    # FIGURE
    # ======================================================

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT)
    )

    # Raw daily means
    ax.plot(
        x,
        y,
        marker="o",
        markersize=2.5,
        linestyle="none",
        alpha=0.35,
        label="Unsmoothed daily mean"
    )

    # Smoothed curves
    for window in WINDOWS:

        y_smooth = results[
            f"GCC_window_{window}"
        ].to_numpy()

        # Make selected 75-day curve slightly stronger
        if window == 75:
            linewidth = 3
            alpha = 1.0
            label = "75 days (selected)"
        else:
            linewidth = LINE_WIDTH
            alpha = 0.75
            label = f"{window} days"

        ax.plot(
            x,
            y_smooth,
            linewidth=linewidth,
            alpha=alpha,
            label=label
        )

    # ------------------------------------------------------
    # Month labels
    # ------------------------------------------------------

    month_starts = [
        1, 32, 60, 91, 121, 152,
        182, 213, 244, 274, 305, 335
    ]

    month_labels = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)

    ax.set_xlim(1, 366)

    ax.set_xlabel("Month")
    ax.set_ylabel(
        "Greenness Chromatic Coordinate (GCC)"
    )

    ax.set_title(
        f"{dataset_name} - Savitzky-Golay sensitivity analysis"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    output_base = (
        OUTPUT_DIR /
        f"{dataset_name}_savgol_sensitivity"
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

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# ==========================================================
# SAVE NUMERICAL RESULTS
# ==========================================================

combined_results = pd.concat(
    all_results,
    ignore_index=True
)

csv_output = (
    OUTPUT_DIR /
    "savgol_sensitivity_values.csv"
)

combined_results.to_csv(
    csv_output,
    index=False
)


print("\nDone.")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Sensitivity values saved to: {csv_output}")
