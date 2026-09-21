# ==========================================================
# Continuous vegetation GCC curve
# All vegetation classes combined
# SE-NM vs BE-LE
# 2017–2024
# ==========================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.signal import savgol_filter


# ==========================================================
# SETTINGS
# ==========================================================

DATASETS = [
    {
        "name": "SE-NM",
        "csv": r"path-to-site1-greenness.csv",
        "colour": "#0b81a2",
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


DATE_COL = "date"
CLASS_COL = "class_folder"
VALUE_COL = "GCC"


# ==========================================================
# VEGETATION CLASSES TO INCLUDE
# ==========================================================
#
# Only vegetation classes are included.
# Sky, snow/ice, bare ground, background etc. are excluded.
#
# Edit these names if the class names in your CSV differ.
# ==========================================================

VEGETATION_CLASSES = [
    "class_4_ericaceous_shrub",
    "class_5_briar_seedling",
    "class_6_graminoid",
    "class_7_forb",
    "class_8_fern",
    "class_9_cryptogam",
    "class_10_vine",
    "class_11_broadleaf_evergreen",
    "class_12_conifer",
    "class_13_broadleaf_deciduous",
]


# ==========================================================
# DATE RANGE
# ==========================================================

START_DATE = pd.Timestamp("2017-01-01")
END_DATE = pd.Timestamp("2024-12-31")


# ==========================================================
# AGGREGATION
# ==========================================================
#
# Mean is recommended for GCC.
#
# Each point represents mean GCC of all included vegetation
# observations recorded on that date.
# ==========================================================

ERROR_TYPE = "sem"
# Options:
# "std"
# "sem"
# "none"


MIN_OBSERVATIONS_PER_DAY = 1


# ==========================================================
# SMOOTHING
# ==========================================================

APPLY_SAVGOL = True

SAVGOL_WINDOW = 21
SAVGOL_POLYORDER = 2


# ==========================================================
# FIGURE SETTINGS
# ==========================================================

FIG_WIDTH = 12
FIG_HEIGHT = 9
DPI = 300

FONT_SIZE = 16
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 16
LEGEND_SIZE = 16

LINE_WIDTH = 2.2
ERROR_ALPHA = 0.15

TITLE = "Vegetation greenness"
Y_LABEL = "Greenness Chromatic Coordinate (GCC)"
X_LABEL = "Date"


SHOW_ERROR_BARS = True
SHOW_POINTS = False
SHOW_SMOOTHED_LINE = True

SAVE_PNG = True
SAVE_PDF = True
SHOW_FIGURE = True


# ==========================================================
# FONT
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
        window = (
            len(y)
            if len(y) % 2 == 1
            else len(y) - 1
        )

    if window <= polyorder or window < 3:
        return y

    return savgol_filter(
        y,
        window_length=window,
        polyorder=polyorder
    )


def load_dataset(dataset):

    print(
        f"Loading {dataset['name']}: "
        f"{dataset['csv']}"
    )

    df = pd.read_csv(
        dataset["csv"]
    )


    # ------------------------------------------------------
    # Check columns
    # ------------------------------------------------------

    required_cols = [
        DATE_COL,
        CLASS_COL,
        VALUE_COL
    ]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset['name']} missing columns: "
            f"{missing}"
        )


    # ------------------------------------------------------
    # Parse values
    # ------------------------------------------------------

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
    ).copy()


    # ------------------------------------------------------
    # Restrict dates
    # ------------------------------------------------------

    df = df[
        df[DATE_COL].between(
            START_DATE,
            END_DATE
        )
    ].copy()


    # ------------------------------------------------------
    # KEEP VEGETATION ONLY
    # ------------------------------------------------------

    df = df[
        df[CLASS_COL].isin(
            VEGETATION_CLASSES
        )
    ].copy()


    if df.empty:

        raise ValueError(
            f"No vegetation data found for "
            f"{dataset['name']}.\n"
            f"Check VEGETATION_CLASSES against "
            f"the values in '{CLASS_COL}'."
        )


    df["dataset"] = dataset["name"]


    return df


# ==========================================================
# LOAD ALL DATA
# ==========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

all_data = []

for dataset in DATASETS:

    df = load_dataset(dataset)

    all_data.append(df)


df_all = pd.concat(
    all_data,
    ignore_index=True
)


# ==========================================================
# SHOW WHICH CLASSES ARE ACTUALLY INCLUDED
# ==========================================================

print("\nVegetation classes included:")

for class_name in sorted(
    df_all[CLASS_COL].unique()
):

    print(
        " -",
        class_name
    )


# ==========================================================
# NORMALISE TO CALENDAR DAY
# ==========================================================

df_all["day"] = (
    df_all[DATE_COL]
    .dt.floor("D")
)


# ==========================================================
# COMBINE ALL VEGETATION GCC VALUES PER DAY
# ==========================================================
#
# IMPORTANT:
#
# We use the MEAN rather than sum.
#
# The mean represents overall vegetation greenness.
# ==========================================================

daily = (
    df_all
    .groupby(
        ["dataset", "day"]
    )[VALUE_COL]
    .agg(
        mean="mean",
        std="std",
        count="count"
    )
    .reset_index()
)


daily["std"] = (
    daily["std"]
    .fillna(0)
)

daily["sem"] = (
    daily["std"]
    / np.sqrt(daily["count"])
)


# Minimum observation threshold
daily = daily[
    daily["count"]
    >= MIN_OBSERVATIONS_PER_DAY
].copy()


# ==========================================================
# ERROR VALUES
# ==========================================================

if ERROR_TYPE == "std":

    daily["error"] = daily["std"]

elif ERROR_TYPE == "sem":

    daily["error"] = daily["sem"]

elif ERROR_TYPE == "none":

    daily["error"] = 0

else:

    raise ValueError(
        "ERROR_TYPE must be "
        "'std', 'sem', or 'none'"
    )


daily = daily.sort_values(
    ["dataset", "day"]
)


# ==========================================================
# SAVE DAILY GCC SUMMARY
# ==========================================================

summary_csv = (
    OUTPUT_DIR
    / "continuous_vegetation_GCC_daily_summary.csv"
)

daily.to_csv(
    summary_csv,
    index=False
)


# ==========================================================
# PLOT
# ==========================================================

fig, ax = plt.subplots(
    figsize=(
        FIG_WIDTH,
        FIG_HEIGHT
    )
)


for dataset in DATASETS:

    dataset_name = dataset["name"]
    colour = dataset["colour"]


    plot_df = daily[
        daily["dataset"]
        == dataset_name
    ].copy()


    plot_df = plot_df.sort_values(
        "day"
    )


    if plot_df.empty:

        print(
            f"WARNING: No data for "
            f"{dataset_name}"
        )

        continue


    x = plot_df["day"]

    y = (
        plot_df["mean"]
        .to_numpy(dtype=float)
    )

    error = (
        plot_df["error"]
        .fillna(0)
        .to_numpy(dtype=float)
    )


    # ======================================================
    # SMOOTH GCC CURVE
    # ======================================================

    if APPLY_SAVGOL:

        y_smooth = safe_savgol(
            y,
            SAVGOL_WINDOW,
            SAVGOL_POLYORDER
        )

    else:

        y_smooth = y


    # ======================================================
    # ERROR BAND
    # ======================================================

    if (
        SHOW_ERROR_BARS
        and ERROR_TYPE != "none"
    ):

        ax.fill_between(
            x,
            y - error,
            y + error,
            color=colour,
            alpha=ERROR_ALPHA,
            linewidth=0,
        )


    # ======================================================
    # RAW DAILY POINTS
    # ======================================================

    if SHOW_POINTS:

        ax.plot(
            x,
            y,
            marker="o",
            markersize=2.5,
            linestyle="none",
            color=colour,
            alpha=0.35,
        )


    # ======================================================
    # LINE
    # ======================================================

    if SHOW_SMOOTHED_LINE:

        y_plot = y_smooth

    else:

        y_plot = y


    ax.plot(
        x,
        y_plot,
        color=colour,
        linewidth=LINE_WIDTH,
        label=dataset_name,
    )


# ==========================================================
# AXIS LABELS
# ==========================================================

# ax.set_title(TITLE)

ax.set_xlabel(
    X_LABEL
)

ax.set_ylabel(
    Y_LABEL
)


# ==========================================================
# X-AXIS RANGE
# ==========================================================
#
# Only show where data actually exist.
# Do not artificially extend back to January.
# ==========================================================

start_date = daily["day"].min()
end_date = daily["day"].max()

ax.set_xlim(
    start_date,
    end_date
)


# ==========================================================
# YEAR TICKS
# ==========================================================
#
# First year is positioned at the actual beginning
# of the dataset.
#
# Following years are positioned at January 1.
# ==========================================================

year_ticks = [
    start_date
]

year_labels = [
    str(start_date.year)
]


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


# ==========================================================
# MONTH TICKS
# ==========================================================
#
# Smaller quarterly labels:
#
# year → Apr → Jul → Oct → next year
# ==========================================================

ax.xaxis.set_minor_locator(
    mdates.MonthLocator(
        bymonth=[
            4,
            7,
            10
        ]
    )
)

ax.xaxis.set_minor_formatter(
    mdates.DateFormatter(
        "%b"
    )
)


# ==========================================================
# X-AXIS TICK STYLE
# ==========================================================

# Years larger and lower
ax.tick_params(
    axis="x",
    which="major",
    labelsize=16,
    pad=14
)

# Months smaller and closer
ax.tick_params(
    axis="x",
    which="minor",
    labelsize=11,
    pad=4
)

# Y-axis
ax.tick_params(
    axis="y",
    labelsize=TICK_SIZE
)


plt.setp(
    ax.get_xticklabels(
        minor=False
    ),
    rotation=0,
    ha="center"
)

plt.setp(
    ax.get_xticklabels(
        minor=True
    ),
    rotation=0,
    ha="center"
)


# ==========================================================
# GRID
# ==========================================================

ax.grid(
    axis="y",
    alpha=0.3
)


# ==========================================================
# LEGEND
# ==========================================================

ax.legend(
    loc="upper right",
    frameon=True
)


# ==========================================================
# LAYOUT
# ==========================================================

fig.tight_layout()


# ==========================================================
# SAVE
# ==========================================================

output_base = (
    OUTPUT_DIR
    / "continuous_total_vegetation_GCC"
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
    f"Figure saved to: "
    f"{OUTPUT_DIR}"
)
print(
    f"Daily summary saved to: "
    f"{summary_csv}"
)
