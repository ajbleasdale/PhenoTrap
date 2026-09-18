# ============================================================
# Snow cover from pixel-summary results
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import savgol_filter

# ======================
# USER SETTINGS
# ======================

CSV_PATH = r"path-to-pixel-summary.csv"

OUTPUT_DIR = r"path-to-output-directory"

DATE_COL = "date"
SNOW_COL = "snow_ice_pixels"

# Pixels considered to represent ground cover
GROUND_COVER_COLS = [
    "snow_ice_pixels",
    "bare_pixels",
    "ericaceous_shrub_pixels",
    "briar_seedling_pixels",
    "graminoid_pixels",
    "forb_pixels",
    "fern_pixels",
    "cryptogam_pixels",
]

# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

SHOW_ERROR_BARS = True
ERROR_TYPE = "sem"       # "sem" or "std"

APPLY_SG_SMOOTHING = True
SG_WINDOW = 5
SG_POLYORDER = 2

FIGSIZE = (12, 9)
DPI = 300
FONT_SIZE = 18

Y_LABEL = "Snow cover (% of ground cover)"
X_LABEL = "Date"
Y_LIMITS = (0, 115)

LINE_COLOR = "#0000a2"

# ======================
# BREAKPOINTS
# ======================

BREAKPOINTS = {
    "onset": "2017-11-19",
    "melt_start": "2017-03-26",
    "disappearance": "2017-05-14",
}

SHOW_BREAKPOINT_LINES = True

# ======================
# LOAD DATA
# ======================

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
})

print(f"Loading: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# ------------------------------------------------------------
# Check columns
# ------------------------------------------------------------

required_cols = [DATE_COL, SNOW_COL] + GROUND_COVER_COLS

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        "The following columns are missing from the CSV:\n"
        + "\n".join(missing_cols)
    )

# ======================
# DATE
# ======================

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    errors="coerce",
    utc=True
).dt.tz_convert(None)

df = df.dropna(subset=[DATE_COL]).copy()

# Calendar date only
df["date"] = df[DATE_COL].dt.normalize()

# ======================
# CALCULATE SNOW COVER
# ======================

# Total pixels belonging to ground-cover classes
df["ground_cover_pixels"] = (
    df[GROUND_COVER_COLS]
    .sum(axis=1)
)

# Snow as % of ground cover
df["snow_cover_percent"] = np.where(
    df["ground_cover_pixels"] > 0,
    (
        df[SNOW_COL]
        / df["ground_cover_pixels"]
    ) * 100,
    np.nan
)

df["snow_cover_percent"] = (
    pd.to_numeric(
        df["snow_cover_percent"],
        errors="coerce"
    )
    .clip(lower=0, upper=100)
)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=["snow_cover_percent"]
)

# ======================
# DAILY SUMMARY
# ======================

daily = (
    df
    .groupby("date")["snow_cover_percent"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .sort_values("date")
)

daily["sem"] = (
    daily["std"]
    / np.sqrt(daily["count"])
)

# Save daily results
summary_csv = (
    output_dir
    / "snow_cover_daily_summary.csv"
)

daily.to_csv(
    summary_csv,
    index=False
)

# ======================
# PLOT
# ======================

fig, ax = plt.subplots(
    figsize=FIGSIZE
)

x = daily["date"]

y = daily["mean"].to_numpy(
    dtype=float
)

# ------------------------------------------------------------
# Error band
# ------------------------------------------------------------

if SHOW_ERROR_BARS:

    yerr = (
        daily[ERROR_TYPE]
        .fillna(0)
        .to_numpy(dtype=float)
    )

    lower = np.clip(
        y - yerr,
        0,
        100
    )

    upper = np.clip(
        y + yerr,
        0,
        100
    )

    ax.fill_between(
        x,
        lower,
        upper,
        color=LINE_COLOR,
        alpha=0.15,
        linewidth=0,
    )

# ------------------------------------------------------------
# Savitzky-Golay smoothing
# ------------------------------------------------------------

if APPLY_SG_SMOOTHING and len(y) >= SG_WINDOW:

    window = SG_WINDOW

    # Window must be odd
    if window % 2 == 0:
        window += 1

    if window > len(y):
        window = len(y)

    if window % 2 == 0:
        window -= 1

    if window > SG_POLYORDER:

        y_plot = savgol_filter(
            y,
            window_length=window,
            polyorder=SG_POLYORDER
        )

        y_plot = np.clip(
            y_plot,
            0,
            100
        )

    else:
        y_plot = y

else:
    y_plot = y

# ------------------------------------------------------------
# Main line
# ------------------------------------------------------------

ax.plot(
    x,
    y_plot,
    color=LINE_COLOR,
    linewidth=2.5,
    label="Snow cover",
)

# ======================
# BREAKPOINT LINES
# ======================

if SHOW_BREAKPOINT_LINES:

    for name, date_string in BREAKPOINTS.items():

        if date_string is None:
            continue

        breakpoint_date = pd.to_datetime(
            date_string
        )

        ax.axvline(
            breakpoint_date,
            color=LINE_COLOR,
            linestyle=":",
            linewidth=1.8,
            alpha=0.5,
        )

# ======================
# AXES
# ======================

ax.set_xlabel(X_LABEL)
ax.set_ylabel(Y_LABEL)

ax.set_ylim(Y_LIMITS)

# Use actual data range
start_date = daily["date"].min()
end_date = daily["date"].max()

ax.set_xlim(
    start_date,
    end_date
)

# ======================
# YEAR TICKS
# ======================

year_ticks = [start_date]
year_labels = [str(start_date.year)]

for year in range(
    start_date.year + 1,
    end_date.year + 1
):

    year_date = pd.Timestamp(
        f"{year}-01-01"
    )

    if year_date <= end_date:

        year_ticks.append(
            year_date
        )

        year_labels.append(
            str(year)
        )

ax.set_xticks(
    year_ticks,
    minor=False
)

ax.set_xticklabels(
    year_labels,
    minor=False
)

# ======================
# MONTH TICKS
# ======================

ax.xaxis.set_minor_locator(
    mdates.MonthLocator()
)

ax.xaxis.set_minor_formatter(
    mdates.DateFormatter("%b")
)

# ======================
# TICK STYLE
# ======================

ax.tick_params(
    axis="x",
    which="major",
    labelsize=18,
    pad=20
)

ax.tick_params(
    axis="x",
    which="minor",
    labelsize=14,
    pad=4
)

plt.setp(
    ax.get_xticklabels(minor=False),
    rotation=0,
    ha="center"
)

plt.setp(
    ax.get_xticklabels(minor=True),
    rotation=0,
    ha="center"
)

# ======================
# GRID / LEGEND
# ======================

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    loc="upper center",
    frameon=True
)

fig.tight_layout()

# ======================
# SAVE
# ======================

out_png = (
    output_dir
    / "snow_cover_pixel_summary.png"
)

fig.savefig(
    out_png,
    dpi=DPI,
    bbox_inches="tight"
)

plt.show()

print(f"Saved figure: {out_png}")
print(f"Saved daily summary: {summary_csv}")
