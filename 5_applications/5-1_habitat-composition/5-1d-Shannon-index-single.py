
# ============================================================
# SHANNON DIVERSITY INDEX
# ONE SPECIFIC DEPLOYMENT
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# USER SETTINGS
# ============================================================

CSV_FILE = r"path-to-dataset-image-pixel-summary.csv"

OUTPUT_DIR = Path(
    r"\\storage-ume.slu.se\home$\arbl0002\My Documents\1-Active\Results\Shannon Index"
)

# Deployment to analyse
# Example filename:
# BE-LE_12_2018-04-21_120000.JPG
DEPLOYMENT = "BE-LE_138" # change to deployment ID

# Column containing image filename
FILENAME_COL = "filename"

DATE_COL = "date"

# Line colour
LINE_COLOUR = "#bc272d"
LINE_COLOUR = "#0000a2"

# ============================================================
# VEGETATION CLASSES TO INCLUDE
# ============================================================

VEGETATION_COLS = [
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


# ============================================================
# PLOT SETTINGS
# ============================================================

FIG_WIDTH = 9
FIG_HEIGHT = 9
DPI = 300

TITLE = DEPLOYMENT

X_LABEL = "Date"
Y_LABEL = "Shannon Diversity Index (H′)"

FONT_SIZE = 18
TITLE_SIZE = 20
AXIS_LABEL_SIZE = 20
TICK_SIZE = 16

LINE_WIDTH = 2

# None = raw daily Shannon index
# 7 = 7-day rolling mean, etc.
ROLLING_WINDOW = None

SAVE_NAME = f"{DEPLOYMENT}_daily_shannon_index.png"


# ============================================================
# FONT
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
})


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

print(f"Total rows in CSV: {len(df):,}")


# ============================================================
# FILTER TO SELECTED DEPLOYMENT
# ============================================================
#
# This assumes filenames begin with the deployment ID:
#
# BE-LE_12_2018-04-21_120000.JPG
# ^^^^^^^^
#
# Using DEPLOYMENT + "_" prevents BE-LE_1 accidentally
# matching BE-LE_10, BE-LE_11, etc.
# ============================================================

df = df[
    df[FILENAME_COL]
    .astype(str)
    .str.startswith(DEPLOYMENT + "_")
].copy()


print(f"Selected deployment: {DEPLOYMENT}")
print(f"Images found: {len(df):,}")


if df.empty:
    raise ValueError(
        f"No images were found for deployment '{DEPLOYMENT}'.\n"
        f"Check DEPLOYMENT and FILENAME_COL."
    )


# ============================================================
# DATE
# ============================================================

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    errors="coerce"
)

df = df.dropna(
    subset=[DATE_COL]
)


# ============================================================
# CHECK VEGETATION COLUMNS
# ============================================================

missing_cols = [
    col for col in VEGETATION_COLS
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        "Missing vegetation columns:\n"
        + "\n".join(missing_cols)
    )


print("\nVegetation classes used:")

for col in VEGETATION_COLS:
    print(" -", col)


# ============================================================
# SUMMARISE BY DAY
# ============================================================

df["day"] = df[DATE_COL].dt.floor("D")

daily = (
    df.groupby("day")[VEGETATION_COLS]
    .sum()
    .sort_index()
)


# ============================================================
# TOTAL VEGETATION PIXELS
# ============================================================

daily["vegetation_pixels"] = (
    daily[VEGETATION_COLS]
    .sum(axis=1)
)


# ============================================================
# VEGETATION CLASS RICHNESS
# ============================================================

daily["vegetation_class_richness"] = (
    daily[VEGETATION_COLS] > 0
).sum(axis=1)


# ============================================================
# SHANNON DIVERSITY INDEX
# ============================================================
#
# H' = -sum(p_i * ln(p_i))
#
# p_i = proportion of vegetation pixels belonging
#       to vegetation class i
# ============================================================

def calculate_shannon(row):

    values = row[VEGETATION_COLS].to_numpy(
        dtype=float
    )

    total = values.sum()

    if total == 0:
        return np.nan

    proportions = values / total

    # Remove zeros because ln(0) is undefined
    proportions = proportions[
        proportions > 0
    ]

    return -np.sum(
        proportions * np.log(proportions)
    )


daily["shannon_index"] = daily.apply(
    calculate_shannon,
    axis=1
)


# ============================================================
# PIELOU'S EVENNESS
# ============================================================

richness = daily[
    "vegetation_class_richness"
]

daily["shannon_evenness"] = np.where(
    richness > 1,
    daily["shannon_index"] / np.log(richness),
    np.nan
)


# ============================================================
# OPTIONAL SMOOTHING
# ============================================================

if ROLLING_WINDOW is not None:

    daily["shannon_plot"] = (
        daily["shannon_index"]
        .rolling(
            window=ROLLING_WINDOW,
            center=True,
            min_periods=1
        )
        .mean()
    )

else:

    daily["shannon_plot"] = (
        daily["shannon_index"]
    )


# ============================================================
# SAVE RESULTS CSV
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

csv_output = (
    OUTPUT_DIR /
    f"{DEPLOYMENT}_daily_shannon_index.csv"
)

daily.to_csv(csv_output)

print(
    f"\nSaved results to:\n{csv_output}"
)


# ============================================================
# SUMMARY
# ============================================================

MAX_SHANNON = np.log(
    len(VEGETATION_COLS)
)

print("\nShannon index summary:")

print(
    daily["shannon_index"].describe()
)

print(
    f"\nMaximum theoretical Shannon index: "
    f"{MAX_SHANNON:.3f}"
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=DPI
)


ax.plot(
    daily.index,
    daily["shannon_plot"],
    color=LINE_COLOUR,
    linewidth=LINE_WIDTH
)


# ============================================================
# LABELS
# ============================================================

ax.set_title(
    TITLE,
    fontname="Arial"
)

ax.set_xlabel(
    X_LABEL,
    fontname="Arial"
)

ax.set_ylabel(
    Y_LABEL,
    fontname="Arial"
)


# ============================================================
# Y AXIS
# ============================================================
#
# Fixed to theoretical range so different deployments
# can be compared using exactly the same scale.
# ============================================================

ax.set_ylim(
    0,
    1.5
)


# ============================================================
# X AXIS
# ============================================================

start_date = daily.index.min()
end_date = daily.index.max()

ax.set_xlim(
    start_date,
    end_date
)


# ============================================================
# YEAR TICKS
# ============================================================

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


# ============================================================
# X-AXIS: DATE EVERY 2 WEEKS
# ============================================================

ax.xaxis.set_major_locator(
    mdates.WeekdayLocator(
        interval=1
    )
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%d %b %Y")
)

ax.tick_params(
    axis="x",
    which="major",
    labelsize=TICK_SIZE
)

plt.setp(
    ax.get_xticklabels(),
    rotation=0,
    ha="center",
    fontname="Arial"
)

for label in ax.get_yticklabels():
    label.set_fontname("Arial")
# ============================================================
# TICK STYLE
# ============================================================

ax.tick_params(
    axis="x",
    which="major",
    labelsize=16,
    pad=14
)

ax.tick_params(
    axis="x",
    which="minor",
    labelsize=11,
    pad=4
)

ax.tick_params(
    axis="y",
    labelsize=TICK_SIZE
)


# Force all tick labels to Arial

for label in ax.get_xticklabels(
    minor=False
):
    label.set_fontname("Arial")

for label in ax.get_xticklabels(
    minor=True
):
    label.set_fontname("Arial")

for label in ax.get_yticklabels():
    label.set_fontname("Arial")


# ============================================================
# FINAL STYLE
# ============================================================

ax.grid(False)

ax.margins(x=0)

plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

out_path = (
    OUTPUT_DIR /
    SAVE_NAME
)

plt.savefig(
    out_path,
    dpi=DPI,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nSaved figure to:\n{out_path}"
)
