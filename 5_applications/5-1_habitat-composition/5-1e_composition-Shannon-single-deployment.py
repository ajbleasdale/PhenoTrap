
# ============================================================
# INDIVIDUAL DEPLOYMENT COMPARISON
# SE-NM vs BE-LE
#
# (a) SE-NM deployment - vegetation composition
# (b) BE-LE deployment - vegetation composition
# (c) SE-NM deployment - Shannon diversity
# (d) BE-LE deployment - Shannon diversity
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch


# ============================================================
# USER SETTINGS
# ============================================================

DATASETS = [
    {
        "site": "SE-NM",
        "csv": (
            r"path-to-SE-NM-pixel-summary.csv"
        ),

        # CHANGE THIS
        "deployment": "SE-NM_113",

        "shannon_colour": "#0000a2",
    },

    {
        "site": "BE-LE",
        "csv": (
            r"path-to-BE-LE-pixel-summary.csv"
        ),

        # CHANGE THIS
        "deployment": "BE-LE_138",

        "shannon_colour": "#bc272d",
    },
]


OUTPUT_DIR = Path(
    r"path-to-output-directory"
)

SAVE_NAME = "SE-NM_BE-LE_deployment_comparison.png"

DATE_COL = "date"


# ============================================================
# COMPOSITION CLASSES
# ============================================================

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


# ============================================================
# VEGETATION CLASSES FOR SHANNON
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
# CLASS LABELS
# ============================================================

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


# ============================================================
# CLASS COLOURS
# ============================================================

CLASS_COLOURS = {
    "sky_pixels": "#A3D6F3",                    # light blue
    "snow_ice_pixels": "#C8C8C8",               # light grey
    "bare_pixels": "#9D2C00",                   # dark rust/brown
    "cryptogam_pixels": "#7E4794",              # purple
    "graminoid_pixels": "#D4A72C",              # ochre/gold
    "forb_pixels": "#F0C571",                   # pale yellow
    "fern_pixels": "#2E7D32",                   # strong green
    "ericaceous_shrub_pixels": "#D95F5F",       # coral/red
    "vine_pixels": "#C44E9B",                   # magenta
    "briar_seedling_pixels": "#285F84",         # medium blue
    "broadleaf_deciduous_pixels": "#7CB342",     # yellow-green
    "broadleaf_evergreen_pixels": "#00D7D2",     # teal/cyan
    "conifer_pixels": "#174A3F",                 # very dark green
}




# ============================================================
# PLOT SETTINGS
# ============================================================

FIG_WIDTH = 12
FIG_HEIGHT = 9
DPI = 300

FONT_SIZE = 14
TITLE_SIZE = 17
AXIS_LABEL_SIZE = 16
TICK_SIZE = 13
LEGEND_SIZE = 12

BAR_WIDTH_DAYS = 1.0
LINE_WIDTH = 1.8

LEGEND_MIN_PERCENT = 0.5

# None = raw daily Shannon
# 7 = centred 7-day rolling mean, etc.
ROLLING_WINDOW = None

# Deployment plots generally need more frequent date ticks
DATE_TICK_INTERVAL = 14


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
    "legend.fontsize": LEGEND_SIZE,
})


# ============================================================
# SHANNON FUNCTION
# ============================================================

def calculate_shannon(row):

    values = row[
        VEGETATION_COLS
    ].to_numpy(dtype=float)

    total = values.sum()

    if total == 0:
        return np.nan

    proportions = values / total

    proportions = proportions[
        proportions > 0
    ]

    return -np.sum(
        proportions * np.log(proportions)
    )


# ============================================================
# PROCESS EACH SITE / DEPLOYMENT
# ============================================================

all_results = {}

legend_classes = set()


for dataset in DATASETS:

    site = dataset["site"]
    deployment_id = dataset["deployment"]
    csv_file = dataset["csv"]

    print("\n" + "=" * 70)
    print(f"Processing {site}: {deployment_id}")
    print("=" * 70)

    # ========================================================
    # LOAD CSV
    # ========================================================

    df = pd.read_csv(csv_file)


    # ========================================================
    # EXTRACT DEPLOYMENT FROM FILENAME
    # ========================================================

    # Example:
    #
    # SE-NM_113_2024-07-08_120000.JPG
    #
    # becomes:
    #
    # SE-NM_113

    df["deployment_id"] = df["filename"].str.extract(
        r"^(.+?)_\d{4}-\d{2}-\d{2}_\d{6}",
        expand=False
    )


    if df["deployment_id"].isna().all():

        raise ValueError(
            f"{site}: could not extract deployment IDs "
            f"from filename."
        )


    # ========================================================
    # PRINT AVAILABLE DEPLOYMENTS
    # ========================================================

    print(f"\nDeployments found in {site}:")

    for dep in sorted(
        df["deployment_id"]
        .dropna()
        .unique()
    ):
        print(" -", dep)


    # ========================================================
    # FILTER SELECTED DEPLOYMENT
    # ========================================================

    df = df[
        df["deployment_id"] == deployment_id
    ].copy()


    if df.empty:

        raise ValueError(
            f"\nNo rows found for deployment "
            f"'{deployment_id}' in {site}."
        )


    print()
    print(f"Selected deployment: {deployment_id}")
    print(f"Image records: {len(df):,}")


    # ========================================================
    # PROCESS DATE
    # ========================================================

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce"
    )

    df = df.dropna(
        subset=[DATE_COL]
    )


    if df.empty:

        raise ValueError(
            f"No valid dates found for {deployment_id}."
        )


    # ========================================================
    # CHECK COLUMNS
    # ========================================================

    required_cols = list(
        dict.fromkeys(
            CLASS_COLS + VEGETATION_COLS
        )
    )

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]


    if missing_cols:

        raise ValueError(
            f"{site}: missing required columns:\n"
            + "\n".join(missing_cols)
        )


    # ========================================================
    # CONVERT TO DAY
    # ========================================================

    df["day"] = (
        df[DATE_COL]
        .dt.floor("D")
    )


    # ========================================================
    # DAILY PIXEL TOTALS
    # ========================================================

    daily_pixels = (
        df
        .groupby("day")[CLASS_COLS]
        .sum()
        .sort_index()
    )


    print(
        f"Start date: "
        f"{daily_pixels.index.min().date()}"
    )

    print(
        f"End date:   "
        f"{daily_pixels.index.max().date()}"
    )


    # ========================================================
    # CONTINUOUS DATE RANGE
    # ========================================================

    full_dates = pd.date_range(
        daily_pixels.index.min(),
        daily_pixels.index.max(),
        freq="D"
    )


    daily_pixels = daily_pixels.reindex(
        full_dates,
        fill_value=0
    )

    daily_pixels.index.name = "day"


    # ========================================================
    # DAILY TOTAL
    # ========================================================

    daily_total = (
        daily_pixels[CLASS_COLS]
        .sum(axis=1)
    )


    # ========================================================
    # COMPOSITION PERCENTAGE
    # ========================================================

    daily_percent = (
        daily_pixels[CLASS_COLS]
        .div(
            daily_total.replace(
                0,
                np.nan
            ),
            axis=0
        )
        .fillna(0)
        * 100
    )


    # ========================================================
    # LEGEND CLASSES
    # ========================================================

    for col in CLASS_COLS:

        if (
            daily_percent[col].max()
            >= LEGEND_MIN_PERCENT
        ):

            legend_classes.add(col)


    # ========================================================
    # SHANNON INDEX
    # ========================================================

    daily_pixels["shannon_index"] = (
        daily_pixels.apply(
            calculate_shannon,
            axis=1
        )
    )


    # Days with no image data should remain gaps,
    # not Shannon = 0

    daily_pixels.loc[
        daily_total == 0,
        "shannon_index"
    ] = np.nan


    # ========================================================
    # OPTIONAL SMOOTHING
    # ========================================================

    if ROLLING_WINDOW is not None:

        daily_pixels["shannon_plot"] = (
            daily_pixels["shannon_index"]
            .rolling(
                window=ROLLING_WINDOW,
                center=True,
                min_periods=1
            )
            .mean()
        )

    else:

        daily_pixels["shannon_plot"] = (
            daily_pixels["shannon_index"]
        )


    # ========================================================
    # STORE
    # ========================================================

    all_results[site] = {
        "deployment": deployment_id,
        "pixels": daily_pixels,
        "percent": daily_percent,
        "colour": dataset["shannon_colour"],
    }


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=DPI,

    # Composition and Shannon within each site align
    sharex="col",

    # Keep Shannon y scales independent
    sharey=False,

    gridspec_kw={
        "height_ratios": [1.15, 1],
        "hspace": 0.08,
        "wspace": 0.18,
    }
)


ax_a = axes[0, 0]
ax_b = axes[0, 1]

ax_c = axes[1, 0]
ax_d = axes[1, 1]


# ============================================================
# PANEL DEFINITIONS
# ============================================================

PANELS = [
    {
        "site": "SE-NM",
        "composition_ax": ax_a,
        "shannon_ax": ax_c,
        "composition_letter": "a",
        "shannon_letter": "c",
    },

    {
        "site": "BE-LE",
        "composition_ax": ax_b,
        "shannon_ax": ax_d,
        "composition_letter": "b",
        "shannon_letter": "d",
    },
]


# ============================================================
# PLOT
# ============================================================

for panel in PANELS:

    site = panel["site"]

    ax_comp = panel["composition_ax"]
    ax_shannon = panel["shannon_ax"]

    results = all_results[site]

    deployment_id = results["deployment"]

    daily_pixels = results["pixels"]
    daily_percent = results["percent"]


    # ========================================================
    # TOP ROW - VEGETATION COMPOSITION
    # ========================================================

    bottom = pd.Series(
        0.0,
        index=daily_percent.index
    )


    for col in CLASS_COLS:

        ax_comp.bar(
            daily_percent.index,
            daily_percent[col],
            bottom=bottom,
            width=BAR_WIDTH_DAYS,
            color=CLASS_COLOURS[col],
            edgecolor="none",
            linewidth=0,
            align="center",
        )

        bottom += daily_percent[col]


    ax_comp.set_ylim(
        0,
        100
    )

    ax_comp.set_title(
        f"{deployment_id}",
        fontsize=TITLE_SIZE
    )

    ax_comp.grid(False)
    ax_comp.margins(x=0)


    # Panel letter

    #ax_comp.text(
    #    0.015,
    #    0.97,
    #    #f"({panel['composition_letter']})",
    #    #transform=ax_comp.transAxes,
        #fontsize=16,
        #fontweight="bold",
        #va="top",
        #ha="right",
    #)


    # ========================================================
    # BOTTOM ROW - SHANNON
    # ========================================================

    ax_shannon.plot(
        daily_pixels.index,
        daily_pixels["shannon_plot"],
        color=results["colour"],
        linewidth=LINE_WIDTH,
    )



    ax_shannon.grid(False)
    ax_shannon.margins(x=0)


    # Panel letter

   # ax_shannon.text(
   #     0.015,
   #     0.97,
   #     f"({panel['shannon_letter']})",
   #     transform=ax_shannon.transAxes,
   #     fontsize=16,
   #     fontweight="bold",
   #     va="top",
   #     ha="left",
   # )


    # ========================================================
    # DATE RANGE
    # ========================================================

    start_date = daily_pixels.index.min()
    end_date = daily_pixels.index.max()


    ax_comp.set_xlim(
        start_date - pd.Timedelta(hours=12),
        end_date + pd.Timedelta(hours=12)
    )


    # ========================================================
    # DATE TICKS
    # ========================================================

    ax_shannon.xaxis.set_major_locator(
        mdates.DayLocator(
            interval=DATE_TICK_INTERVAL
        )
    )

    ax_shannon.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%d %b"
        )
    )

    ax_shannon.tick_params(
        axis="x",
        which="major",
        labelsize=TICK_SIZE,
        pad=6
    )

    plt.setp(
        ax_shannon.get_xticklabels(),
        rotation=0,
        ha="center"
    )

# ============================================================
# SHARED SHANNON Y-AXIS
# Use panel (d) / BE-LE to determine the range
# ============================================================

be_le_shannon = (
    all_results["BE-LE"]["pixels"]["shannon_plot"]
    .dropna()
)

# Maximum value occurring in panel (d)
shannon_ymax = be_le_shannon.max()

# Add 5% headroom above the highest BE-LE value
shannon_ymax = shannon_ymax * 1.05

# Apply BE-LE range to BOTH Shannon panels
ax_c.set_ylim(
    0,
    shannon_ymax
)

ax_d.set_ylim(
    0,
    shannon_ymax
)

# ============================================================
# AXIS LABELS
# ============================================================

ax_a.set_ylabel(
    "Percentage (%)",
    fontsize=AXIS_LABEL_SIZE
)

ax_b.set_ylabel(
    "Percentage (%)",
    fontsize=AXIS_LABEL_SIZE
)

ax_c.set_ylabel(
    "Shannon Diversity Index (H′)",
    fontsize=AXIS_LABEL_SIZE
)

ax_d.set_ylabel(
    "Shannon Diversity Index (H′)",
    fontsize=AXIS_LABEL_SIZE
)


ax_c.set_xlabel(
    "Date",
    fontsize=AXIS_LABEL_SIZE
)

ax_d.set_xlabel(
    "Date",
    fontsize=AXIS_LABEL_SIZE
)


# ============================================================
# REMOVE TOP DATE LABELS
# ============================================================

ax_a.tick_params(
    axis="x",
    which="both",
    labelbottom=False
)

ax_b.tick_params(
    axis="x",
    which="both",
    labelbottom=False
)


# ============================================================
# FONT / TICKS
# ============================================================

for ax in axes.flat:

    ax.tick_params(
        axis="y",
        labelsize=TICK_SIZE
    )

    for label in ax.get_xticklabels():
        label.set_fontname("Arial")

    for label in ax.get_yticklabels():
        label.set_fontname("Arial")


# ============================================================
# SHARED VEGETATION LEGEND
# ============================================================

legend_handles = []


for col in CLASS_COLS:

    if col in legend_classes:

        legend_handles.append(
            Patch(
                facecolor=CLASS_COLOURS[col],
                edgecolor="none",
                label=CLASS_LABELS[col]
            )
        )


fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=4,
    frameon=False,
    fontsize=LEGEND_SIZE,
)


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.08,
    right=0.97,
    top=0.94,
    bottom=0.16,
    hspace=0.08,
    wspace=0.18,
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

out_path = (
    OUTPUT_DIR
    / SAVE_NAME
)


plt.savefig(
    out_path,
    dpi=DPI,
    bbox_inches="tight"
)

plt.close()


print("\n" + "=" * 70)
print("Finished")
print("=" * 70)

print(
    f"\nSaved deployment comparison to:\n"
    f"{out_path}"
)
