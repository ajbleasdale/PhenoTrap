import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# PATHS
# ============================================================
csv_path = r"path-to-csv-file"
output_folder = r"path-to-output-folder"



# ============================================================
# SETTINGS
# ============================================================
date_column = "date"
dataset_column = "dataset"

start_date = pd.Timestamp("2017-01-01")
end_date = pd.Timestamp("2024-10-01")

dataset_order = ["BE-LE", "SE-NM"]

color_map = {
    "BE-LE": "#e9c716", #red
    "SE-NM": "#50ad9f", #yellow
}

figure_size = (12, 9)
dpi = 300
show_plot = True


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 15,
})


# ============================================================
# LOAD DATA
# ============================================================
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(csv_path)

required_columns = [date_column, dataset_column]
missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# PREPARE DATA
# ============================================================
df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce"
)

invalid_dates = df[date_column].isna().sum()

if invalid_dates > 0:
    print(f"Dropping {invalid_dates} rows with invalid dates.")

df = df.dropna(subset=[date_column]).copy()

# Restrict dates to 2017–2024
df = df[
    df[date_column].between(start_date, end_date)
].copy()

df[dataset_column] = (
    df[dataset_column]
    .fillna("Missing")
    .astype(str)
)


# ============================================================
# CONVERT DATES TO MONTHS
# ============================================================
df["month"] = (
    df[date_column]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# ============================================================
# COUNT IMAGES PER MONTH
# ============================================================
counts = (
    df.groupby(["month", dataset_column])
      .size()
      .unstack(fill_value=0)
)


# Add missing months as zero
all_months = pd.date_range(
    start="2017-01-01",
    end="2024-10-01",
    freq="MS"
)

counts = counts.reindex(
    all_months,
    fill_value=0
)

counts.index.name = "month"


# ============================================================
# DATASET ORDER
# ============================================================
existing = [
    c for c in dataset_order
    if c in counts.columns
]

remaining = [
    c for c in counts.columns
    if c not in existing
]

counts = counts[existing + remaining]


print("\nMonthly counts:")
print(counts)


# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=figure_size)

bottom = pd.Series(
    0,
    index=counts.index,
    dtype=float
)

for dataset in counts.columns:

    values = counts[dataset]

    ax.bar(
        counts.index,
        values,
        bottom=bottom,

        # Thinner monthly bars
        width=18,

        label=dataset,
        color=color_map.get(dataset),

        # Black outlines
        edgecolor="black",
        linewidth=0.7,

        align="center"
    )

    bottom += values


# ============================================================
# AXES
# ============================================================
ax.set_xlabel("Date")
ax.set_ylabel("Image Count")
#ax.set_title("Camera Trap Imagery Monthly Coverage")


# ============================================================
# X-AXIS FORMATTING
# ============================================================

# Major ticks: years
ax.xaxis.set_major_locator(
    mdates.YearLocator()
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y")
)

# Minor ticks: March, June, September
ax.xaxis.set_minor_locator(
    mdates.MonthLocator(
        bymonth=[4, 7, 10]
    )
)

ax.xaxis.set_minor_formatter(
    mdates.DateFormatter("%b")
)

# Year label size
ax.tick_params(
    axis="x",
    which="major",
    labelsize=16,
    pad=15
)

# Smaller month labels
ax.tick_params(
    axis="x",
    which="minor",
    labelsize=11,
    pad=5
)

# Limits
ax.set_xlim(
    pd.Timestamp("2016-12-15"),
    pd.Timestamp("2024-10-01")
)

# ============================================================
# LEGEND
# ============================================================
ax.legend()

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================
output_path = os.path.join(
    output_folder,
    "monthly_stacked_bar_by_dataset.png"
)

plt.savefig(
    output_path,
    dpi=dpi,
    bbox_inches="tight"
)

print(f"Saved figure to: {output_path}")


if show_plot:
    plt.show()
else:
    plt.close()
