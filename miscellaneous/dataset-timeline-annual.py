import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
csv_path = r"path-to-csv-file"
output_folder = r"path-to-output-folder"

date_column = "date"

# Select one by removing #
#stack_column = "dataset" #stack by dataset
stack_column = "partition_original"  #stack by partition


category_order = None

color_map = {
    # dataset
    "BE-LE": "#bc272d",  #red
    "SE-NM": "#e9c716",  #yellow

    # partition
    "train": "#bc272d",  #red
    "validation": "#e9c716", #yellow
    "test": "#50ad9f", #teal
}



figure_size = (12, 9)
dpi = 300
show_plot = True
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,          # base size
    "axes.titlesize": 18,     # title
    "axes.labelsize": 20,     # axis labels
    "xtick.labelsize": 16,    # x tick labels
    "ytick.labelsize": 16,    # y tick labels
    "legend.fontsize": 16,    # legend text
    "legend.title_fontsize": 15
})

# ============================================================
# SETUP
# ============================================================
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(csv_path)

required_columns = [date_column, stack_column]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Parse dates
df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

# Drop invalid dates
n_invalid = df[date_column].isna().sum()
if n_invalid > 0:
    print(f"Dropping {n_invalid} rows with invalid dates in column '{date_column}'")
df = df.dropna(subset=[date_column]).copy()

# Fill missing stack labels
df[stack_column] = df[stack_column].fillna("Missing").astype(str)


# ============================================================
# CREATE MONTH LABELS
# ============================================================
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

df["month_label"] = df[date_column].dt.strftime("%b")
df["month_label"] = pd.Categorical(df["month_label"], categories=month_order, ordered=True)

# ============================================================
# GROUP COUNTS
# ============================================================
counts = (
    df.groupby(["month_label", stack_column], observed=False)
      .size()
      .unstack(fill_value=0)
      .reindex(month_order, fill_value=0)
)

wanted_order = ["train", "validation", "test"]
counts = counts[[c for c in wanted_order if c in counts.columns]]


print("\nCounts table:")
print(counts)

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=figure_size)

bottom = [0] * len(counts.index)

for category in counts.columns:
    values = counts[category].values
    color = color_map.get(category, None)

    ax.bar(
        counts.index.astype(str),
        values,
        bottom=bottom,
        label=category,
        color=color,
        edgecolor="black",
        linewidth=1.0
    )

    bottom = [b + v for b, v in zip(bottom, values)]

ax.set_xlabel("Month")
ax.set_ylabel("Image Count")
#ax.set_title(f"Camera Trap Imagery Monthly Coverage")


plt.xticks(rotation=0, ha="center")
ax.legend()
plt.tight_layout()

# ============================================================
# SAVE
# ============================================================
output_name = f"monthly_stacked_bar_by_{stack_column}.png"
output_path = os.path.join(output_folder, output_name)

plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

if show_plot:
    plt.show()
else:
    plt.close()
