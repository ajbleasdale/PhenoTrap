
# ==========================================================
# Greenness curve plotting script
# All vegetation classes - supplementary figure
# ==========================================================

from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
# ==========================================================
# SETTINGS
# ==========================================================

DATASETS = [
    {
        "name": "SE-NM",
        "csv": r"path-to-dataset-1-greeenness-metrics.csv",
        "colour": "#0000a2",
    },
    {
        "name": "BE-LE",
        "csv": r"path-to-dataset-2-greeenness-metrics.csv",
        "colour": "#bc272d",
    },
]

OUTPUT_DIR = Path(
    r"path-to-output-folder"
)

# Exact vegetation class identifiers from the CSV files.
SELECTED_CLASSES = {
    "class_4_ericaceous_shrub": "Ericaceous shrubs",
    "class_5_briar_seedling": "Briars and seedlings",
    "class_6_graminoid": "Graminoids",
    "class_7_forb": "Forbs",
    "class_8_fern": "Ferns",
    "class_9_cryptogam": "Cryptogams",
    "class_10_vine": "Vines",
    "class_11_broadleaf_evergreen": "Broadleaf evergreen",
    "class_12_conifer": "Conifers",
    "class_13_broadleaf_deciduous": "Broadleaf deciduous",
}

DATE_COL = "date"
CLASS_COL = "class_folder"
VALUE_COL = "GCC"

PLOT_MODE = "annual_cycle"
ERROR_TYPE = "std"

APPLY_SAVGOL = True
SAVGOL_WINDOW = 31
SAVGOL_POLYORDER = 3

# Figure layout
N_COLUMNS = 2
PANEL_WIDTH = 8
PANEL_HEIGHT = 5

DPI = 300

FONT_SIZE = 14
TITLE_SIZE = 15
AXIS_LABEL_SIZE = 16
TICK_SIZE = 13
LEGEND_SIZE = 15

LINE_WIDTH = 2
ERROR_ALPHA = 0.15

Y_LABEL = "Green Chromatic Coordinate (GCC)"

# Shared Y-axis limits
# None = calculate automatically from all plotted data.
Y_LIMITS = None
Y_PADDING = 0.05

SHOW_POINTS = False
SHOW_ERROR_BARS = True
SHOW_SMOOTHED_LINE = True

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
# FUNCTIONS
# ==========================================================

def ensure_odd_window(window):
    if window % 2 == 0:
        window += 1
    return window


def safe_savgol(y, window, polyorder):
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


def load_dataset(dataset):
    df = pd.read_csv(dataset["csv"])

    required_cols = [DATE_COL, CLASS_COL, VALUE_COL]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"{dataset['name']}: Missing column: {col}"
            )

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL], errors="coerce"
    )
    df[VALUE_COL] = pd.to_numeric(
        df[VALUE_COL], errors="coerce"
    )

    df = df.dropna(
        subset=[DATE_COL, VALUE_COL, CLASS_COL]
    )

    df = df[
        df[CLASS_COL].isin(SELECTED_CLASSES)
    ].copy()

    df["dataset"] = dataset["name"]

    return df


def aggregate_data(df):
    df = df.copy()

    if PLOT_MODE == "continuous":
        group_cols = ["dataset", CLASS_COL, DATE_COL]
        x_col = DATE_COL

    elif PLOT_MODE == "annual_cycle":
        df["day_of_year"] = df[DATE_COL].dt.dayofyear

        group_cols = [
            "dataset",
            CLASS_COL,
            "day_of_year"
        ]
        x_col = "day_of_year"

    else:
        raise ValueError(
            "PLOT_MODE must be 'continuous' or 'annual_cycle'"
        )

    grouped = (
        df.groupby(group_cols)[VALUE_COL]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0)
    grouped["sem"] = (
        grouped["std"] / np.sqrt(grouped["count"])
    )

    if ERROR_TYPE == "std":
        grouped["error"] = grouped["std"]
    elif ERROR_TYPE == "sem":
        grouped["error"] = grouped["sem"]
    elif ERROR_TYPE == "none":
        grouped["error"] = 0
    else:
        raise ValueError(
            "ERROR_TYPE must be 'std', 'sem', or 'none'"
        )

    return grouped, x_col


# ==========================================================
# LOAD AND AGGREGATE
# ==========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_data = []

for dataset in DATASETS:
    all_data.append(load_dataset(dataset))

df_all = pd.concat(all_data, ignore_index=True)

if df_all.empty:
    raise ValueError(
        "No selected vegetation data found."
    )

summary, x_col = aggregate_data(df_all)

# ==========================================================
# PREPARE PLOTTING DATA AND COMMON Y-AXIS LIMITS
# ==========================================================

plot_data = {}
y_min_values = []
y_max_values = []

for class_id in SELECTED_CLASSES:

    for dataset in DATASETS:
        dname = dataset["name"]

        subset = summary[
            (summary["dataset"] == dname) &
            (summary[CLASS_COL] == class_id)
        ].sort_values(x_col)

        if subset.empty:
            print(f"No data: {dname} - {class_id}")
            continue

        x = subset[x_col].to_numpy()
        y = subset["mean"].to_numpy(dtype=float)
        err = subset["error"].to_numpy(dtype=float)

        if APPLY_SAVGOL:
            y_smooth = safe_savgol(
                y,
                SAVGOL_WINDOW,
                SAVGOL_POLYORDER
            )
        else:
            y_smooth = y

        y_plot = (
            y_smooth if SHOW_SMOOTHED_LINE else y
        )

        plot_data[(class_id, dname)] = {
            "x": x,
            "y": y,
            "y_plot": y_plot,
            "error": err,
        }

        # Include plotted lines in the common limits.
        y_min_values.append(np.min(y_plot))
        y_max_values.append(np.max(y_plot))

        # Include error shading if displayed.
        if SHOW_ERROR_BARS and ERROR_TYPE != "none":
            y_min_values.append(np.min(y - err))
            y_max_values.append(np.max(y + err))

        # Include raw points if displayed.
        if SHOW_POINTS:
            y_min_values.append(np.min(y))
            y_max_values.append(np.max(y))


if not plot_data:
    raise ValueError("No data available for plotting.")

if Y_LIMITS is None:
    global_min = min(y_min_values)
    global_max = max(y_max_values)

    y_range = global_max - global_min

    if y_range == 0:
        y_range = 0.01

    common_ylim = (
        global_min - y_range * Y_PADDING,
        global_max + y_range * Y_PADDING
    )
else:
    common_ylim = Y_LIMITS



# ==========================================================
# CREATE MULTI-PANEL FIGURE WITH PEAK GCC VALUES
# ==========================================================

from matplotlib.lines import Line2D

n_classes = len(SELECTED_CLASSES)
n_rows = math.ceil(n_classes / N_COLUMNS)

fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=N_COLUMNS,
    figsize=(
        PANEL_WIDTH * N_COLUMNS + 2,
        PANEL_HEIGHT * n_rows
    ),
    sharex=(PLOT_MODE == "annual_cycle"),
    sharey=True,
    squeeze=False
)

axes_flat = axes.flatten()

# Reserve space on the right of each subplot for peak GCC.
# The peak values are placed outside the plotting area.
PEAK_X_POSITION = 1.04
PEAK_TITLE_Y = 0.92
PEAK_FIRST_Y = 0.76
PEAK_SECOND_Y = 0.64

for class_index, (class_id, class_label) in enumerate(
    SELECTED_CLASSES.items()
):
    ax = axes_flat[class_index]

    peak_values = {}

    for dataset in DATASETS:
        dname = dataset["name"]
        colour = dataset["colour"]

        key = (class_id, dname)

        if key not in plot_data:
            continue

        data = plot_data[key]

        x = data["x"]
        y = data["y"]
        y_plot = data["y_plot"]
        err = data["error"]

        # ==================================================
        # ERROR SHADING
        # ==================================================

        if SHOW_ERROR_BARS and ERROR_TYPE != "none":
            ax.fill_between(
                x,
                y - err,
                y + err,
                color=colour,
                alpha=ERROR_ALPHA,
                linewidth=0
            )

        # ==================================================
        # RAW MEAN POINTS
        # ==================================================

        if SHOW_POINTS:
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3,
                linestyle="none",
                color=colour,
                alpha=0.7
            )

        # ==================================================
        # GCC CURVE
        # ==================================================

        ax.plot(
            x,
            y_plot,
            color=colour,
            linewidth=LINE_WIDTH,
            linestyle="-",
            label=dname
        )

        # ==================================================
        # PEAK GCC VALUE AND DATE
        # ==================================================

        peak_index = int(np.argmax(y_plot))

        peak_gcc = float(y_plot[peak_index])
        peak_x = x[peak_index]

        if PLOT_MODE == "annual_cycle":
            peak_date = (
                    datetime(2024, 1, 1)
                    + timedelta(days=int(peak_x) - 1)
            ).strftime("%d %b")
        else:
            peak_date = pd.to_datetime(peak_x).strftime("%d %b %Y")

        peak_values[dname] = {
            "value": peak_gcc,
            "date": peak_date,
            "colour": colour
        }

        # Small dot at the peak of the smoothed curve
        ax.plot(
            peak_x,
            peak_gcc,
            marker="o",
            markersize=5,
            linestyle="none",
            color=colour,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=6
        )

    # ======================================================
    # PANEL TITLE
    # ======================================================

    panel_letter = chr(97 + class_index)

    ax.set_title(
        f"({panel_letter}) {class_label}",
        loc="left",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    # ======================================================
    # PEAK GCC COLUMN
    # ======================================================

    ax.text(
        PEAK_X_POSITION,
        PEAK_TITLE_Y,
        "Peak GCC",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        clip_on=False
    )

    for dataset_index, dataset in enumerate(DATASETS):

        dname = dataset["name"]
        colour = dataset["colour"]

        y_position = (
            PEAK_FIRST_Y
            if dataset_index == 0
            else PEAK_SECOND_Y
        )

        if dname in peak_values:
            peak_text = (
                f"{dname}: {peak_values[dname]['value']:.3f}\n"
                f"({peak_values[dname]['date']})"
            )
        else:
            peak_text = f"{dname}: N/A"

        ax.text(
            PEAK_X_POSITION,
            y_position,
            peak_text,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=11,
            color=colour,
            clip_on=False
        )

    # ======================================================
    # AXIS FORMATTING
    # ======================================================

    ax.set_ylim(common_ylim)
    ax.grid(True, alpha=0.3)

    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE
    )

    if PLOT_MODE == "annual_cycle":
        ax.set_xlim(1, 366)


# Hide unused panels.
for ax in axes_flat[n_classes:]:
    ax.set_visible(False)


# ==========================================================
# X-AXIS FORMATTING
# ==========================================================

if PLOT_MODE == "annual_cycle":

    month_starts = [
        1, 32, 60, 91, 121, 152,
        182, 213, 244, 274, 305, 335
    ]

    month_labels = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    for ax in axes_flat[:n_classes]:
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)

    fig.supxlabel(
        "Month",
        fontsize=AXIS_LABEL_SIZE,
        y=0.035
    )

else:
    fig.supxlabel(
        "Date",
        fontsize=AXIS_LABEL_SIZE,
        y=0.035
    )


fig.supylabel(
    Y_LABEL,
    fontsize=AXIS_LABEL_SIZE,
    x=0.015
)


# ==========================================================
# SHARED LEGEND
# ==========================================================

legend_handles = [
    Line2D(
        [0], [0],
        color=dataset["colour"],
        linewidth=LINE_WIDTH,
        label=dataset["name"]
    )
    for dataset in DATASETS
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(DATASETS),
    fontsize=LEGEND_SIZE,
    frameon=False,
    bbox_to_anchor=(0.5, 0.005)
)


# ==========================================================
# SPACING
# ==========================================================

# Extra horizontal spacing accommodates the peak GCC
# annotations positioned outside each subplot.
fig.subplots_adjust(
    left=0.09,
    right=0.85,
    top=0.96,
    bottom=0.07,
    hspace=0.20,
    wspace=0.35
)


# ==========================================================
# SAVE FIGURE
# ==========================================================

output_base = (
    OUTPUT_DIR /
    f"all_vegetation_greenness_peaks_{PLOT_MODE}"
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

print("Done.")
print(f"Common Y-axis limits: {common_ylim}")
print(f"Figures saved to: {OUTPUT_DIR}")
