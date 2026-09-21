# ==========================================================
# Multi-site Single-class GCC curve
# ==========================================================


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==========================================================
# SETTINGS
# ==========================================================

DATASETS = [
    {
        "name": "SE-NM",
        "csv": r"path-to-site1-greenness.csv",
        "colour": "#0000a2",
    },
    {
        "name": "BE-LE",
        "csv": r"path-to-site2-greenness.csv",
        "colour": "#bc272d",
    },
]

OUTPUT_DIR = Path(
    r"path-to-output-directory"
)

SELECTED_CLASSES = [

]

MANUAL_BREAKPOINTS = {
    "Boreal": {
        "i": 154,
        "ii": 186,
        "iii": 222,
        "iv": 249,
        "v": 276,
    },
    "Temperate": {
        "i": 79,
        "ii": 126,
        "iii": 131,
        "iv": 293,
        "v": 326,
    },
}

DATE_COL = "date"
CLASS_COL = "class_folder"
VALUE_COL = "GCC"

PLOT_MODE = "annual_cycle"

ERROR_TYPE = "std"

APPLY_SAVGOL = True
SAVGOL_WINDOW = 75
SAVGOL_POLYORDER = 3

FIG_WIDTH = 12
FIG_HEIGHT = 9
DPI = 300

FONT_SIZE = 18
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 18
TICK_SIZE = 18
LEGEND_SIZE = 20

LINE_WIDTH = 2
ERROR_ALPHA = 0.15

TITLE = "Deciduous Broadleaf - Mean annual greenness curve"
Y_LABEL = "Greenness Chromatic Coordinate (GCC)"

SHOW_POINTS = False
SHOW_ERROR_BARS = True
SHOW_SMOOTHED_LINE = True
SHOW_BREAKPOINTS = False
SHOW_BREAKPOINT_LINES = False

SAVE_PNG = True
SAVE_PDF = True
SHOW_FIGURE = True


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

    return savgol_filter(y, window_length=window, polyorder=polyorder)


def load_dataset(dataset):
    df = pd.read_csv(dataset["csv"])

    required_cols = [DATE_COL, CLASS_COL, VALUE_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")

    df = df.dropna(subset=[DATE_COL, VALUE_COL, CLASS_COL])
    df = df[df[CLASS_COL].isin(SELECTED_CLASSES)]

    df["dataset"] = dataset["name"]

    return df


def aggregate_data(df):
    if PLOT_MODE == "continuous":
        group_cols = ["dataset", CLASS_COL, DATE_COL]
        x_col = DATE_COL

    elif PLOT_MODE == "annual_cycle":
        df = df.copy()
        df["day_of_year"] = df[DATE_COL].dt.dayofyear
        group_cols = ["dataset", CLASS_COL, "day_of_year"]
        x_col = "day_of_year"

    else:
        raise ValueError("PLOT_MODE must be 'continuous' or 'annual_cycle'")

    grouped = (
        df.groupby(group_cols)[VALUE_COL]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0)
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])

    if ERROR_TYPE == "std":
        grouped["error"] = grouped["std"]
    elif ERROR_TYPE == "sem":
        grouped["error"] = grouped["sem"]
    elif ERROR_TYPE == "none":
        grouped["error"] = 0
    else:
        raise ValueError("ERROR_TYPE must be 'std', 'sem', or 'none'")

    return grouped, x_col


# ==========================================================
# MAIN
# ==========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_data = []

for dataset in DATASETS:
    df = load_dataset(dataset)
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

if df_all.empty:
    raise ValueError("No deciduous data found. Check class name and CSV paths.")

summary, x_col = aggregate_data(df_all)

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

for dataset in DATASETS:
    dname = dataset["name"]
    colour = dataset["colour"]

    plot_df = summary[
        (summary["dataset"] == dname) &
        (summary[CLASS_COL] == "class_13_broadleaf_deciduous")
    ].copy()

    plot_df = plot_df.sort_values(x_col)

    if plot_df.empty:
        continue

    x = plot_df[x_col].to_numpy()
    y = plot_df["mean"].to_numpy()
    err = plot_df["error"].fillna(0).to_numpy()

    if APPLY_SAVGOL:
        y_smooth = safe_savgol(y, SAVGOL_WINDOW, SAVGOL_POLYORDER)
    else:
        y_smooth = y

    if SHOW_ERROR_BARS and ERROR_TYPE != "none":
        ax.fill_between(
            x,
            y - err,
            y + err,
            color=colour,
            alpha=ERROR_ALPHA,
            linewidth=0,
        )

    if SHOW_POINTS:
        ax.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linestyle="none",
            color=colour,
            alpha=0.7,
        )

    if SHOW_SMOOTHED_LINE:
        ax.plot(
            x,
            y_smooth,
            color=colour,
            linestyle="-",
            linewidth=LINE_WIDTH,
            label=f"{dname}",
        )
    else:
        ax.plot(
            x,
            y,
            color=colour,
            linestyle="-",
            linewidth=LINE_WIDTH,
            label=f"{dname}",
        )

    # ======================================================
    # ADD BREAKPOINT DOTS + NUMBER LABELS
    # ======================================================

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
    })


    if SHOW_BREAKPOINTS and dname in MANUAL_BREAKPOINTS:

        for i, (breakpoint_name, doy) in enumerate(
            MANUAL_BREAKPOINTS[dname].items(),
            start=1
        ):

            y_break = np.interp(doy, x, y_smooth)

            if SHOW_BREAKPOINT_LINES:
                ax.axvline(
                    x=doy,
                    color=colour,
                    linestyle=":",
                    linewidth=1.4,
                    alpha=0.55,
                    zorder=4,
                )

            ax.plot(
                doy,
                y_break,
                marker="o",
                markersize=6,
                color=colour,
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=6,
            )

            ax.text(
                doy,
                y_break + 0.003,
                breakpoint_name,      # "i", "ii", "iii", ...
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color=colour,         # Match the curve colour
                zorder=7,
            )


#ax.set_title(TITLE, fontsize=TITLE_SIZE)
ax.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_SIZE)

if PLOT_MODE == "continuous":
    ax.set_xlabel("Date", fontsize=AXIS_LABEL_SIZE)
else:
    ax.set_xlabel("Month", fontsize=AXIS_LABEL_SIZE)

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

ax.tick_params(axis="both", labelsize=TICK_SIZE)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=LEGEND_SIZE, loc='upper right')

fig.tight_layout()

output_base = OUTPUT_DIR / f"deciduous_greenness_curve_{PLOT_MODE}"

if SAVE_PNG:
    fig.savefig(f"{output_base}.png", dpi=DPI, bbox_inches="tight")

if SAVE_PDF:
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")

if SHOW_FIGURE:
    plt.show()
else:
    plt.close(fig)

print("Done.")
print(f"Figures saved to: {OUTPUT_DIR}")
