# ============================================================
# Snow cover continuous comparison
# Boreal-Model vs Boreal-Manual
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

OUTPUT_DIR = r"path-to-output-directory"

BOREAL_MANUAL_CSV = r"path-to-comparison-data.csv"

DATASETS = [
    {
        "name": "Model",
        "csv": r"path-to-pixel-summary.csv",
        "type": "pixels",
        "date_col": "date",
        "snow_pixels_col": "snow_ice_pixels",
        "ground_cover_pixel_cols": [
            "snow_ice_pixels",
            "bare_pixels",
            "ericaceous_shrub_pixels",
            "briar_seedling_pixels",
            "graminoid_pixels",
            "forb_pixels",
            "fern_pixels",
            "cryptogam_pixels",
        ],
        "color": "#0000a2",
        "linestyle": "-",
    },
    {
        "name": "Manual",
        "csv": BOREAL_MANUAL_CSV,
        "type": "snow_cover",
        "date_col": "date_recorded",
        "snow_cover_col": "Snow.cover",
        "color": "#50ad9f",
        "linestyle": "--",
    },
]

SHOW_ERROR_BARS = True
ERROR_TYPE = "sem"  # "sem" or "std"

APPLY_SG_SMOOTHING = True
SG_WINDOW = 5
SG_POLYORDER = 2

FIGSIZE = (12, 9)
DPI = 300
FONT_SIZE = 18

TITLE = "Boreal snow cover"
Y_LABEL = "Snow cover (% of ground cover)"
X_LABEL = "Date"
Y_LIMITS = (0, 115)

# If Snow.cover in manual CSV is 0–1, set True.
# If already 0–100, leave False.
MANUAL_SNOW_IS_FRACTION = False

# ============================================================
# BREAKPOINT SETTINGS
# Manually edit these dates
# ============================================================

BREAKPOINTS = {
    "Model": {
        "onset": "2017-11-19",
        "melt_start": "2017-03-26",
        "disappearance": "2017-05-14",
    },
    "Manual": {
        "onset": "2017-11-19",
        "melt_start": "2017-03-27",
        "disappearance": "2017-05-23",
    },
}

BREAKPOINT_STYLES = {
    "onset": {
        "label": "Onset",
        "linestyle": ":",
    },
    "melt_start": {
        "label": "Melt start",
        "linestyle": ":",
    },
    "disappearance": {
        "label": "Disappearance",
        "linestyle": ":",
    },
}

SHOW_BREAKPOINT_LINES = True
SHOW_BREAKPOINT_TEXT = True

# ======================
# SCRIPT
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

all_data = []

for config in DATASETS:

    dataset_name = config["name"]
    csv_path = config["csv"]

    print(f"Loading {dataset_name}: {csv_path}")

    df = pd.read_csv(csv_path)

    date_col = config["date_col"]

    if date_col not in df.columns:
        raise ValueError(
            f"{dataset_name}: date column '{date_col}' not found.\n"
            f"Available columns are:\n{list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
        utc=True
    ).dt.tz_convert(None)

    df = df.dropna(subset=[date_col]).copy()

    if config["type"] == "pixels":

        snow_col = config["snow_pixels_col"]
        ground_cols = config["ground_cover_pixel_cols"]

        missing_cols = [
            c for c in [snow_col] + ground_cols
            if c not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"{dataset_name}: These columns are missing from the CSV:\n"
                + "\n".join(missing_cols)
            )

        df["ground_cover_pixels"] = df[ground_cols].sum(axis=1)

        df["snow_cover_percent"] = np.where(
            df["ground_cover_pixels"] > 0,
            (df[snow_col] / df["ground_cover_pixels"]) * 100,
            np.nan
        )

    elif config["type"] == "snow_cover":

        snow_cover_col = config["snow_cover_col"]

        if snow_cover_col not in df.columns:
            raise ValueError(
                f"{dataset_name}: snow-cover column '{snow_cover_col}' not found.\n"
                f"Available columns are:\n{list(df.columns)}"
            )

        df["snow_cover_percent"] = pd.to_numeric(
            df[snow_cover_col],
            errors="coerce"
        )

        if MANUAL_SNOW_IS_FRACTION:
            df["snow_cover_percent"] *= 100

    else:
        raise ValueError(f"Unknown dataset type: {config['type']}")

    df["snow_cover_percent"] = df["snow_cover_percent"].clip(lower=0, upper=100)

    df["dataset"] = dataset_name
    df["date"] = df[date_col]

    all_data.append(df[[
        "dataset",
        "date",
        "snow_cover_percent"
    ]])

data = pd.concat(all_data, ignore_index=True)

# Make sure both datasets use calendar date only
data["date"] = pd.to_datetime(data["date"]).dt.normalize()

data["snow_cover_percent"] = pd.to_numeric(
    data["snow_cover_percent"],
    errors="coerce"
)


data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=["snow_cover_percent"])

# ============================================================
# DAILY SUMMARY THROUGH REAL TIME
# ============================================================

daily = (
    data
    .groupby(["dataset", "date"])["snow_cover_percent"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

daily["sem"] = daily["std"] / daily["count"] ** 0.5
daily = daily.sort_values(["dataset", "date"])

summary_csv = output_dir / "snow_cover_continuous_daily_summary.csv"
combined_csv = output_dir / "snow_cover_continuous_all_values.csv"

daily.to_csv(summary_csv, index=False)
data.to_csv(combined_csv, index=False)

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=FIGSIZE)

for config in DATASETS:

    dataset_name = config["name"]

    sub = (
        daily[daily["dataset"] == dataset_name]
        .sort_values("date")
        .copy()
    )

    sub["mean"] = pd.to_numeric(sub["mean"], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=["mean"])

    if sub.empty:
        print(f"WARNING: No data to plot for {dataset_name}")
        continue

    x = sub["date"]
    y = sub["mean"].to_numpy(dtype=float)

    color = config.get("color", None)
    linestyle = config.get("linestyle", "-")

    if SHOW_ERROR_BARS:
        yerr = sub[ERROR_TYPE].fillna(0).to_numpy(dtype=float)

        lower = np.clip(y - yerr, 0, 100)
        upper = np.clip(y + yerr, 0, 100)

        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    if APPLY_SG_SMOOTHING and len(y) >= SG_WINDOW:
        window = SG_WINDOW

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
            y_plot = np.clip(y_plot, 0, 100)
        else:
            y_plot = y
    else:
        y_plot = y

    ax.plot(
        x,
        y_plot,
        color=color,
        linestyle=linestyle,
        linewidth=2.5,
        label=dataset_name,
    )
# ============================================================
# MODEL vs MANUAL ERROR METRICS
# ============================================================

comparison = (
    daily
    .pivot(
        index="date",
        columns="dataset",
        values="mean"
    )
    .dropna(subset=["Model", "Manual"])
    .copy()
)

print("\nMatched dates:")
print(len(comparison))

if len(comparison) == 0:
    print("WARNING: No matching Model and Manual dates found.")

else:
    # Difference in percentage points
    comparison["error"] = (
        comparison["Model"] - comparison["Manual"]
    )

    # Absolute error
    comparison["absolute_error"] = (
        comparison["error"].abs()
    )

    # Squared error
    comparison["squared_error"] = (
        comparison["error"] ** 2
    )

    # MAE
    mae = comparison["absolute_error"].mean()

    # RMSE
    rmse = np.sqrt(
        comparison["squared_error"].mean()
    )

    print(f"MAE:  {mae:.2f} percentage points")
    print(f"RMSE: {rmse:.2f} percentage points")

    # Save comparison
    comparison.to_csv(
        output_dir / "snow_cover_model_manual_error.csv"
    )
# ============================================================
# ADD BREAKPOINT LINES
# ============================================================

if SHOW_BREAKPOINT_LINES:

    used_labels = set()

    for config in DATASETS:

        dataset_name = config["name"]
        color = config.get("color", None)

        if dataset_name not in BREAKPOINTS:
            continue

        for breakpoint_key, date_string in BREAKPOINTS[dataset_name].items():

            if date_string is None:
                continue

            breakpoint_date = pd.to_datetime(date_string)

            style = BREAKPOINT_STYLES[breakpoint_key]
            breakpoint_label = style["label"]

            legend_label = f"{breakpoint_label}"

            if legend_label in used_labels:
                legend_label = None
            else:
                used_labels.add(legend_label)

            ax.axvline(
                breakpoint_date,
                color=color,
                linestyle=style["linestyle"],
                linewidth=1.8,
                alpha=0.5,
               # label=legend_label,
            )

            #if SHOW_BREAKPOINT_TEXT:
            #    ax.text(
            #        breakpoint_date,
            #        Y_LIMITS[1] - 4,
            #        breakpoint_label,
            #        rotation=90,
            #        va="top",
            #        ha="right",
            #        color=color,
            #        fontsize=FONT_SIZE - 3,
            #    )

# ============================================================
# FORMAT AXES
# ============================================================

# ax.set_title(TITLE)
ax.set_xlabel(X_LABEL)
ax.set_ylabel(Y_LABEL)
ax.set_ylim(Y_LIMITS)
#ax.set_xlim(
#    left=pd.Timestamp("2017-01-01")
#)

# ============================================================
# X-AXIS: ACTUAL DATA RANGE + YEAR LABELS
# ============================================================

# Use the actual plotted data range
start_date = data["date"].min()
end_date = data["date"].max()

ax.set_xlim(start_date, end_date)


# ============================================================
# YEAR TICKS
# ============================================================

# First year label sits at the actual start date
year_ticks = [start_date]
year_labels = [str(start_date.year)]

# Later years sit at January 1
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

# Smaller labels every 3 months
ax.xaxis.set_minor_locator(
    mdates.MonthLocator(
       # bymonth=[4, 7, 10]
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
    labelsize=18,
    pad=20
)

# Months: smaller and closer to the axis
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

# ------------------------------------------------------------
# GRID / LEGEND
# ------------------------------------------------------------
ax.grid(True, alpha=0.3)

ax.legend(
    loc="upper center",
    ncol=3,
    frameon=True
)

fig.tight_layout()

out_png = output_dir / "snow_cover_continuous_comparison_with_breakpoints.png"

fig.savefig(
    out_png,
    dpi=DPI,
    bbox_inches="tight"
)

plt.show()

print(f"Saved figure: {out_png}")
print(f"Saved summary CSV: {summary_csv}")
print(f"Saved combined values CSV: {combined_csv}")
