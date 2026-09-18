from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ======================
# USER SETTINGS
# ======================

CSV_FILE = r"path-to-pixel-summary.csv"
OUTPUT_DIR = r"path-to-output-directory"

# ------------------------------------------------------------
# DEPLOYMENT TO PLOT
# ------------------------------------------------------------

DEPLOYMENT_COL = "deployment_id"
DEPLOYMENT_ID = "SE-NM_113" #change to desired deployment

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
CLASS_COLOURS = {
    "sky_pixels": "#9BD5EF",
    "snow_ice_pixels": "#c8c8c8",
    "bare_pixels": "#9d2c00",
    "cryptogam_pixels": "#7E4794",
    "graminoid_pixels": "#A6A83A",
    "forb_pixels": "#f0c571",
    "fern_pixels": "#26734D",
    "ericaceous_shrub_pixels": "#e25759",
    "vine_pixels": "#D65A9E",
    "briar_seedling_pixels": "#59a89c",
    "broadleaf_deciduous_pixels": "#79C267",
    "broadleaf_evergreen_pixels": "#257A70",
    "conifer_pixels": "#174A3F",
}

# ======================
# PLOT SETTINGS
# ======================

FIG_WIDTH = 12
FIG_HEIGHT = 9
DPI = 300

TITLE = DEPLOYMENT_ID
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

Y_AS_PERCENT = True

LEGEND_MIN_PERCENT = 0.5

SAVE_NAME = f"daily_stacked_bar_chart_percent_{DEPLOYMENT_ID}.png"

# ======================
# LOAD DATA
# ======================

df = pd.read_csv(CSV_FILE)

# ======================
# EXTRACT DEPLOYMENT ID FROM FILENAME
# ======================

# Example filename:
# SE-NM_1_2024-07-08_120000.JPG
#
# Extracted deployment_id:
# SE-NM_1

df["deployment_id"] = df["filename"].str.extract(
    r"^(.+?)_\d{4}-\d{2}-\d{2}_\d{6}",
    expand=False
)

# Check that deployment IDs were successfully extracted
if df["deployment_id"].isna().all():
    raise ValueError(
        "Could not extract deployment IDs from the filename column."
    )

print("Deployments found:")
print(sorted(df["deployment_id"].dropna().unique()))


# ======================
# FILTER TO ONE DEPLOYMENT
# ======================

df = df[
    df["deployment_id"] == DEPLOYMENT_ID
].copy()

if df.empty:
    raise ValueError(
        f"No rows were found for deployment '{DEPLOYMENT_ID}'."
    )

print()
print(f"Selected deployment: {DEPLOYMENT_ID}")
print(f"Number of image records: {len(df):,}")
# ======================
# PROCESS DATE
# ======================

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    errors="coerce"
)

df = df.dropna(
    subset=[DATE_COL]
)

if df.empty:
    raise ValueError(
        f"No valid dates found for deployment '{DEPLOYMENT_ID}'."
    )


# ======================
# CHECK CLASS COLUMNS
# ======================

missing_cols = [
    col for col in CLASS_COLS
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        "The following class columns were not found:\n"
        + "\n".join(missing_cols)
    )

print("\nUsing these pixel columns:")

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

print()
print(f"Start date: {daily.index.min().date()}")
print(f"End date:   {daily.index.max().date()}")
print(f"Days represented: {len(daily):,}")


# ======================
# FILL MISSING DATES
# ======================

full_dates = pd.date_range(
    daily.index.min(),
    daily.index.max(),
    freq="D"
)

daily = daily.reindex(
    full_dates,
    fill_value=0
)

daily.index.name = "day"


# ======================
# CONVERT TO PERCENTAGE
# ======================

if Y_AS_PERCENT:

    daily_total = daily.sum(axis=1)

    daily = (
        daily.div(
            daily_total.replace(0, pd.NA),
            axis=0
        )
        .fillna(0)
        * 100
    )


# ======================
# PLOT
# ======================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=DPI
)

bottom = pd.Series(
    0.0,
    index=daily.index
)


for col in CLASS_COLS:

    label = CLASS_LABELS.get(
        col,
        col
    )

    colour = None

    if CLASS_COLOURS is not None:
        colour = CLASS_COLOURS.get(
            col,
            None
        )

    # Only include meaningful classes in legend
    if Y_AS_PERCENT:
        show_in_legend = (
            daily[col].max()
            >= LEGEND_MIN_PERCENT
        )
    else:
        show_in_legend = (
            daily[col].sum() > 0
        )

    ax.bar(
        daily.index,
        daily[col],
        bottom=bottom,
        width=BAR_WIDTH_DAYS,
        label=(
            label
            if show_in_legend
            else "_nolegend_"
        ),
        color=colour,
        edgecolor=EDGE_COLOUR,
        linewidth=EDGE_WIDTH,
        align="center",
    )

    bottom += daily[col]


# ======================
# STYLE
# ======================

ax.set_title(
    TITLE,
    fontsize=TITLE_SIZE
)

ax.set_xlabel(
    X_LABEL,
    fontsize=AXIS_LABEL_SIZE
)

ax.set_ylabel(
    "Percentage (%)"
    if Y_AS_PERCENT
    else Y_LABEL,
    fontsize=AXIS_LABEL_SIZE
)

if Y_AS_PERCENT:
    ax.set_ylim(
        0,
        100
    )

ax.tick_params(
    axis="both",
    labelsize=TICK_SIZE
)


# ============================================================
# X-AXIS RANGE
# ============================================================

start_date = daily.index.min()
end_date = daily.index.max()

# Slight half-day padding prevents first/last bars being cut off
ax.set_xlim(
    start_date - pd.Timedelta(hours=12),
    end_date + pd.Timedelta(hours=12)
)


# ============================================================
# DATE AXIS
# ============================================================

# Show a tick every day
ax.xaxis.set_major_locator(
    mdates.DayLocator(interval=14)
)

# Format as day + abbreviated month
# Example: 01 Jul, 02 Jul, 03 Jul
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%d %b")
)

ax.tick_params(
    axis="x",
    which="major",
    labelsize=TICK_SIZE,
    pad=6
)

plt.setp(
    ax.get_xticklabels(),
    rotation=0,
    ha="center"
)

# ======================
# LEGEND
# ======================

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

output_dir = Path(
    OUTPUT_DIR
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

out_path = (
    output_dir
    / SAVE_NAME
)

plt.savefig(
    out_path,
    dpi=DPI,
    bbox_inches="tight"
)

plt.close()
