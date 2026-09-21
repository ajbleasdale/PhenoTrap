# ==========================================================
# Single-site Single-class GCC curve
# ==========================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib.dates as mdates

# ==========================================================
# SETTINGS
# ==========================================================

CSV_FILE = r"path-to-csv-file.csv"

OUTPUT_DIR = Path(r"path-to-output-directory")


CLASS_NAME = "class_13_broadleaf_deciduous"
CLASS_LABEL = "broadleaf_deciduous"

DATE_COL = "date"
CLASS_COL = "class_folder"
VALUE_COL = "GCC"

# Minimum number of images needed per day-of-year
MIN_IMAGES_PER_DAY = 1

# Error band
ERROR_TYPE = "std"
# Options: "std", "sem", "none"

# Smoothing
APPLY_SAVGOL = True
SAVGOL_WINDOW = 21
SAVGOL_POLYORDER = 2

# Figure appearance
FIG_WIDTH = 13
FIG_HEIGHT = 8
DPI = 300

LINE_COLOUR = "darkorange"
ERROR_COLOUR = "darkorange"

LINE_WIDTH = 2.5
ERROR_ALPHA = 0.15

TITLE = "Fern - Mean annual greenness curve"
Y_LABEL = "Greenness Chromatic Coordinate (GCC)"
X_LABEL = "Month"

# ==========================================================
# MANUAL PHENOLOGICAL BREAKPOINTS
# ==========================================================

ADD_BREAKPOINTS = False

# Use day-of-year values
MANUAL_BREAKPOINTS = {
    "Start of season": 96,
    "Peak": 131,
    "End of season": 324,
}

BREAKPOINT_LINE_COLOUR = "darkorange"
BREAKPOINT_LINESTYLE = "--"
BREAKPOINT_ALPHA = 0.5
BREAKPOINT_LINEWIDTH = 1.6
BREAKPOINT_TEXT_SIZE = 11

TITLE_SIZE = 18
AXIS_LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 13

SAVE_PNG = True
SAVE_PDF = True
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
        return y

    if len(y) < 5:
        return y

    window = ensure_odd(SAVGOL_WINDOW)

    if window > len(y):
        window = len(y) if len(y) % 2 == 1 else len(y) - 1

    if window <= SAVGOL_POLYORDER or window < 3:
        return y

    return savgol_filter(
        y,
        window_length=window,
        polyorder=SAVGOL_POLYORDER
    )


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(CSV_FILE)

for col in [DATE_COL, CLASS_COL, VALUE_COL]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")

df = df.dropna(subset=[DATE_COL, CLASS_COL, VALUE_COL])

df = df[df[CLASS_COL] == CLASS_NAME].copy()

if df.empty:
    raise ValueError(f"No data found for class: {CLASS_NAME}")


# ==========================================================
# ANNUAL CYCLE AGGREGATION
# ==========================================================

df["day_of_year"] = df[DATE_COL].dt.dayofyear

grouped = (
    df.groupby("day_of_year")[VALUE_COL]
    .agg(["mean", "std", "count"])
    .reset_index()
    .sort_values("day_of_year")
)

grouped["std"] = grouped["std"].fillna(0)
grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])

grouped = grouped[grouped["count"] >= MIN_IMAGES_PER_DAY].copy()

if grouped.empty:
    raise ValueError(
        "No data left after MIN_IMAGES_PER_DAY filtering. "
        "Try lowering MIN_IMAGES_PER_DAY."
    )

if ERROR_TYPE == "std":
    error = grouped["std"].values
elif ERROR_TYPE == "sem":
    error = grouped["sem"].values
elif ERROR_TYPE == "none":
    error = np.zeros(len(grouped))
else:
    raise ValueError("ERROR_TYPE must be 'std', 'sem', or 'none'")

x = grouped["day_of_year"].values
y = grouped["mean"].values
y_smooth = smooth_curve(y)


# ==========================================================
# PLOT
# ==========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

if ERROR_TYPE != "none":
    ax.fill_between(
        x,
        y - error,
        y + error,
        color=ERROR_COLOUR,
        alpha=ERROR_ALPHA,
        linewidth=0,
    )

ax.plot(
    x,
    y_smooth,
    color=LINE_COLOUR,
    linewidth=LINE_WIDTH,
    label=CLASS_LABEL,
)

# ==========================================================
# ADD MANUAL BREAKPOINTS
# ==========================================================

if ADD_BREAKPOINTS:

    for label, doy in MANUAL_BREAKPOINTS.items():

        # Vertical breakpoint line
        ax.axvline(
            x=doy,
            color=BREAKPOINT_LINE_COLOUR,
            linestyle=BREAKPOINT_LINESTYLE,
            linewidth=BREAKPOINT_LINEWIDTH,
            alpha=0.8,
        )

        # Point at the intersection with the smoothed curve
        y_at_doy = np.interp(doy, x, y_smooth)

        ax.scatter(
            doy,
            y_at_doy,
            color=BREAKPOINT_LINE_COLOUR,
            edgecolors="white",  # optional, makes the point stand out
            s=60,                 # marker size
            linewidth=1.0,
            zorder=10,
        )

        #ax.text(
        #    doy,
        #    y_top,
        #    label,
        #    rotation=90,
        #    verticalalignment="top",
        #    horizontalalignment="right",
        #    fontsize=BREAKPOINT_TEXT_SIZE,
        #    color=BREAKPOINT_LINE_COLOUR,
        #)

ax.set_title(TITLE, fontsize=TITLE_SIZE)
ax.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_SIZE)
ax.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_SIZE)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": TICK_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})

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
ax.legend(fontsize=LEGEND_SIZE, loc = 'upper right')
ax.grid(True, alpha=0.3)

fig.tight_layout()


# ==========================================================
# SAVE
# ==========================================================

safe_class = CLASS_NAME.replace(" ", "_").replace("/", "_")

outfile = OUTPUT_DIR / f"annual_gcc_curve_{safe_class}"

if SAVE_PNG:
    fig.savefig(f"{outfile}.png", dpi=DPI, bbox_inches="tight")

if SAVE_PDF:
    fig.savefig(f"{outfile}.pdf", bbox_inches="tight")

if SHOW_FIGURE:
    plt.show()
else:
    plt.close(fig)

print("Done.")
print(f"Saved to: {OUTPUT_DIR}")
