# ============================================================
# Continuous daily stacked bar chart
# One stacked bar per day, no gaps between bars
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ======================
# USER SETTINGS
# ======================

CSV_FILE = r"path-to-pixel-summary-csv.csv"
OUTPUT_DIR = r"path-to-output-directory"

DATE_COL = "date"

# Columns to stack
CLASS_COLS = [
    "sky_pixels",
    "snow_ice_pixels",
    "bare_pixels",
    "ericaceous_shrub_pixels",
    "briar_seedling_pixels",
    "graminoid_pixels",
    "forb_pixels",
    "fern_pixels",
    "cryptogam_pixels",
    "vine_pixels",
    "broadleaf_evergreen_pixels",
    "conifer_pixels",
    "broadleaf_deciduous_pixels",
]

# Optional nicer labels for legend
CLASS_LABELS = {
    "sky_pixels": "Sky",
    "snow_ice_pixels": "Snow and Ice",
    "bare_pixels": "Bare",
    "ericaceous_shrub_pixels": "Ericaceous Shrub",
    "briar_seedling_pixels": "Briar and Seedling",
    "graminoid_pixels": "Graminoid",
    "forb_pixels": "Forb",
    "fern_pixels": "Fern",
    "cryptogam_pixels": "Cryptogam",
    "vine_pixels": "Vine",
    "broadleaf_evergreen_pixels": "Broadleaf Evergreen",
    "conifer_pixels": "Conifer",
    "broadleaf_deciduous_pixels": "Broadleaf Deciduous",
}


# Optional colours
# Set to None to use matplotlib defaults

CLASS_COLOURS = {
    "sky_pixels": "#9BD5EF",# Blue

    "snow_ice_pixels": "#c8c8c8",             # Grey

    "bare_pixels": "#9d2c00",                 # Dark Red / earthy brown

    "cryptogam_pixels": "#7E4794",            # Purple

    "graminoid_pixels": "#A6A83A",             # Olive

    "forb_pixels": "#f0c571",                  # Gold

    "fern_pixels": "#26734D",                  # Dark green

    "ericaceous_shrub_pixels": "#e25759",            # Red

    "vine_pixels": "#D65A9E",                  # Pink / magenta

    "briar_seedling_pixels": "#59a89c",          # Teal

    "broadleaf_deciduous_pixels": "#79C267",   # Light green

    "broadleaf_evergreen_pixels": "#257A70",   # Dark teal

    "conifer_pixels": "#174A3F",               # Very dark green
}

# ======================
# PLOT SETTINGS
# ======================

FIG_WIDTH = 12
FIG_HEIGHT = 9
DPI = 300

TITLE = "BE-LE"
X_LABEL = "Date"
Y_LABEL = "Percentage Cover (%)"

FONT_SIZE = 18
TITLE_SIZE = 20
AXIS_LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 16

BAR_WIDTH_DAYS = 1.0
EDGE_COLOUR = "none"
EDGE_WIDTH = 0


Y_AS_PERCENT = True   # True = stacked bars sum to 100% each day

DATE_TICK_INTERVAL_MONTHS = 3
DATE_FORMAT = "%b %y"

SAVE_NAME = "daily_stacked_bar_chart_percent_BE-LE.png"

# ======================
# LOAD DATA
# ======================

df = pd.read_csv(CSV_FILE)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])

if CLASS_COLS is None:
    EXCLUDE_COLS = [
        "total_pixels",
        "valid_pixels_excluding_ignore",
        "background_pixels",
    ]

    CLASS_COLS = [
        col for col in df.columns
        if "pixel" in col.lower()
        and "perc" not in col.lower()
        and col.lower() not in [c.lower() for c in EXCLUDE_COLS]
    ]

print("Using these pixel columns:")
for col in CLASS_COLS:
    print(" -", col)

# ======================
# SUMMARISE BY DAY
# ======================

df["day"] = df[DATE_COL].dt.floor("D")

daily = (
    df.groupby("day")[CLASS_COLS]
    .sum()
    .sort_index()
)

# Fill missing dates so the time series is continuous
full_dates = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
daily = daily.reindex(full_dates, fill_value=0)
daily.index.name = "day"

# Convert to percentage if wanted
# Convert each day to percentage so every daily bar sums to 100%
if Y_AS_PERCENT:
    daily_total = daily.sum(axis=1)

    daily = (
        daily.div(daily_total.replace(0, pd.NA), axis=0)
        .fillna(0)
        * 100
    )

    # Force y-axis to 0–100
    Y_MIN = 0
    Y_MAX = 100


# ======================
# PLOT
# ======================

LEGEND_MIN_PERCENT = 0.5

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

bottom = pd.Series(0, index=daily.index)

for col in CLASS_COLS:
    label = CLASS_LABELS.get(col, col)

    colour = None
    if CLASS_COLOURS is not None:
        colour = CLASS_COLOURS.get(col, None)

    if Y_AS_PERCENT:
        show_in_legend = daily[col].max() >= LEGEND_MIN_PERCENT
    else:
        show_in_legend = daily[col].sum() > 0

    ax.bar(
        daily.index,
        daily[col],
        bottom=bottom,
        width=BAR_WIDTH_DAYS,
        label=label if show_in_legend else "_nolegend_",
        color=colour,
        edgecolor=EDGE_COLOUR,
        linewidth=EDGE_WIDTH,
        align="center",
    )

    bottom += daily[col]

# ======================
# STYLE
# ======================

ax.set_title(TITLE, fontsize=TITLE_SIZE)
ax.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel("Percentage (%)" if Y_AS_PERCENT else Y_LABEL, fontsize=AXIS_LABEL_SIZE)
#ax.set_xlim( left=pd.Timestamp("2017-01-01"))

if Y_AS_PERCENT:
    ax.set_ylim(0, 100)

ax.tick_params(axis="both", labelsize=TICK_SIZE)

# ============================================================
# X-AXIS: ACTUAL DATA RANGE + YEAR LABELS
# ============================================================

# Actual data range
start_date = daily.index.min()
end_date = daily.index.max()

ax.set_xlim(start_date, end_date)


# ============================================================
# YEAR TICKS
# ============================================================

# First year label is placed at the actual start of the data
year_ticks = [start_date]
year_labels = [str(start_date.year)]

# Following years are placed at January 1
for year in range(start_date.year + 1, end_date.year + 1):
    year_date = pd.Timestamp(f"{year}-01-01")

    if year_date <= end_date:
        year_ticks.append(year_date)
        year_labels.append(str(year))


ax.set_xticks(
    year_ticks,
    minor=False
)

ax.set_xticklabels(
    year_labels,
    minor=False
)


# ============================================================
# MONTH TICKS
# ============================================================

# Months every 3 months: Apr, Jul, Oct
ax.xaxis.set_minor_locator(
    mdates.MonthLocator(
        bymonth=[4, 7, 10]
    )
)

ax.xaxis.set_minor_formatter(
    mdates.DateFormatter("%b")
)


# ============================================================
# TICK STYLE
# ============================================================

# Years: larger and lower
ax.tick_params(
    axis="x",
    which="major",
    labelsize=16,
    pad=14
)

# Months: smaller and closer to axis
ax.tick_params(
    axis="x",
    which="minor",
    labelsize=11,
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

plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

ax.legend(
    fontsize=LEGEND_SIZE,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
)

ax.margins(x=0)
ax.grid(False)

plt.tight_layout()

# ======================
# SAVE
# ======================

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

out_path = output_dir / SAVE_NAME
plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
plt.close()

print(f"Saved stacked daily bar chart to: {out_path}")
