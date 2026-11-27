import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================
plt.rcParams["font.family"] = "Arial"

# Input and output files
csv_path = r"path-to-iou-results.csv"
output_path = r"path-to-output.png"

# CSV column names
class_column = "Class"
metric_columns = ["Overall", "BE-LE", "SE-NM"]

# Labels shown in the legend
metric_labels = {
    "Overall": "Overall",
    "BE-LE": "BE–LE",
    "SE-NM": "SE–NM",
}

# Figure dimensions in inches
figure_width = 12
figure_height = 9

# Bar appearance
bar_height = 0.22
bar_colours = {
    "Overall": "#e9c716",
    "BE-LE": "#bc272d",
    "SE-NM": "#50ad9f",
}



# Font settings
font_family = "Arial"
axis_label_size = 20
tick_label_size = 16
legend_size = 16
title_size = 18
value_label_size = 8

# Axis settings
x_axis_label = "Intersection over Union (%)"
plot_title = None  # Example: "Per-class IoU"
x_axis_min = 0
x_axis_max = 100
x_tick_interval = 10

# Display options
sort_classes = False
sort_by = "Overall"
sort_ascending = True

show_grid = True
show_value_labels = False
show_legend = True

# Values in the CSV:
# True  = values are stored from 0 to 1, e.g. 0.834
# False = values are already percentages, e.g. 83.4
convert_to_percent = True

# Image export
output_dpi = 300
transparent_background = False


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

plt.rcParams["font.family"] = font_family

df = pd.read_csv(csv_path)

required_columns = [class_column] + metric_columns
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing from the CSV file: "
        + ", ".join(missing_columns)
    )

# Keep only the columns used in the plot
plot_df = df[required_columns].copy()

# Convert metric columns to numeric values
for column in metric_columns:
    plot_df[column] = pd.to_numeric(
        plot_df[column],
        errors="coerce"
    )

# Convert proportions to percentages when required
if convert_to_percent:
    plot_df[metric_columns] = plot_df[metric_columns] * 100

# Optional sorting
if sort_classes:
    plot_df = plot_df.sort_values(
        by=sort_by,
        ascending=sort_ascending
    )

plot_df = plot_df.reset_index(drop=True)


# ============================================================
# CREATE CLUSTERED HORIZONTAL BAR CHART
# ============================================================

number_of_classes = len(plot_df)
number_of_metrics = len(metric_columns)

y_positions = np.arange(number_of_classes)

fig, ax = plt.subplots(
    figsize=(figure_width, figure_height)
)

# Centre the group of bars around each class position
offsets = (
    np.arange(number_of_metrics)
    - (number_of_metrics - 1) / 2
) * bar_height

# ============================================================
# PREPARE ORDER BEFORE PLOTTING
# ============================================================

wanted_order = ["Overall", "BE-LE", "SE-NM"]

available_groups = [
    metric for metric in wanted_order
    if metric in plot_df.columns
]

plot_df = plot_df[[class_column] + available_groups]

# Positive y is visually higher on a non-inverted axis
offsets = {
    "Overall": bar_height,       # top
    "BE-LE": 0,                  # middle
    "SE-NM": -bar_height,        # bottom
}

y_positions = np.arange(len(plot_df))


# ============================================================
# DRAW BARS
# ============================================================

for metric in available_groups:

    values = plot_df[metric].to_numpy(dtype=float)
    bar_positions = y_positions + offsets[metric]

    bars = ax.barh(
        bar_positions,
        values,
        height=bar_height,
        label=metric_labels.get(metric, metric),
        color=bar_colours.get(metric),
        edgecolor="black",
        linewidth=0.4
    )

    if show_value_labels:
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue

            ax.text(
                bar.get_width() + 0.7,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=value_label_size
            )


# ============================================================
# FORMAT THE FIGURE
# ============================================================


available_groups = [c for c in wanted_order if c in plot_df.columns]
plot_df = plot_df[[class_column] + available_groups]

ax.set_yticks(y_positions)
ax.set_yticklabels(
    plot_df[class_column],
    fontsize=tick_label_size
)

# First CSV class appears at the top
#ax.invert_yaxis()

ax.set_xlabel(
    x_axis_label,
    fontsize=axis_label_size
)

ax.set_ylabel(
    "Class",
    fontsize=axis_label_size
)

if plot_title:
    ax.set_title(
        plot_title,
        fontsize=title_size
    )

ax.set_xlim(x_axis_min, x_axis_max)
ax.set_xticks(
    np.arange(
        x_axis_min,
        x_axis_max + x_tick_interval,
        x_tick_interval
    )
)

ax.tick_params(
    axis="x",
    labelsize=tick_label_size
)

ax.tick_params(
    axis="y",
    labelsize=tick_label_size
)

if show_grid:
    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.5,
        alpha=0.4
    )
    ax.set_axisbelow(True)

if show_legend:
    ax.legend(
        #title="Evaluation group",
        fontsize=legend_size,
        title_fontsize=legend_size,
        #frameon=False
    )

# Places the first CSV class at the top of the figure
#ax.invert_yaxis()

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)



plt.tight_layout()

plt.savefig(
    output_path,
    dpi=output_dpi,
    bbox_inches="tight",
    transparent=transparent_background
)

plt.show()
