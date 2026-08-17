import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ======================
# EDIT PATHS
# ======================
input_file = Path(r"path-to.input-file.xlsx")
output_dir = r"path-to-output-directory"
#output_dir.mkdir(exist_ok=True)

# ======================
# SETTINGS YOU CAN CHANGE
# ======================
sheet_name = "summary"   # change if needed

partition_col = "split"
class_col = "class_name"
#pixel_col = "images_with_class"
pixel_col = "class_pixels_in_split"


# bar colours for train / val / test
partition_colours = {
    "train": "#bc272d", #red
    "validation": "#e9c716", #yellow
 #  "test": "#0000a2", #blue
    "test": "#50ad9f",  # teal

}

# text sizes
title_size = 18
axis_label_size = 20
tick_label_size = 16
legend_size = 16

# figure size
fig_width = 12
fig_height = 9

# ======================
# READ FILE
# ======================
if input_file.suffix.lower() == ".csv":
    df = pd.read_csv(input_file)
else:
    df = pd.read_excel(input_file, sheet_name=sheet_name)

# clean names
df.columns = df.columns.str.strip()
df[partition_col] = df[partition_col].astype(str).str.strip().str.lower()
df["class_id"] = pd.to_numeric(df["class_id"], errors="coerce")

# convert pixels to million pixels
#df["million_pixels"] = df[pixel_col]
df["million_pixels"] = df[pixel_col] / 1_000_000

# ======================
# PIVOT TO STACKED FORMAT
# ======================
plot_df = df.pivot_table(
    index=["class_id", "class_name"],
    columns=partition_col,
    values="million_pixels",
    aggfunc="sum",
    fill_value=0
)

# sort using numeric class_id
plot_df = plot_df.sort_index(level=0)

# use class names as displayed labels
plot_df.index = [name for _, name in plot_df.index]

# keep partition order
wanted_order = ["train", "validation", "test"]
plot_df = plot_df[[c for c in wanted_order if c in plot_df.columns]]

# ======================
# PLOT STACKED BAR CHART
# ======================

plt.rcParams["font.family"] = "Arial"

#ax = plot_df.plot(
#    kind="bar",
#    stacked=True,
#    figsize=(fig_width, fig_height),
#    color=[partition_colours.get(c, "#999999") for c in plot_df.columns],
#    edgecolor="black",
#    linewidth=0.2
#)

ax = plot_df.plot(
    kind="barh",
    stacked=True,
    figsize=(fig_width, fig_height),
    color=[partition_colours.get(c, "#999999") for c in plot_df.columns],
    edgecolor="black",
    linewidth=0.2
)

# Put grid behind the bars
ax.set_axisbelow(True)

# Add vertical gridlines
ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)

#ax.set_title("Class pixel counts by partition - SIMPLE", fontsize=title_size)
ax.set_ylabel("Class", fontsize=axis_label_size)
ax.set_xlabel("Pixels (millions)", fontsize=axis_label_size)
ax.set_xlim(0,1400)

#ax.set_ylabel("Class Count", fontsize=axis_label_size)

#ax.tick_params(axis="x", labelsize=tick_label_size, rotation=90,)
ax.tick_params(axis="x", labelsize=tick_label_size)
ax.tick_params(axis="y", labelsize=tick_label_size)

ax.legend(
    #title="Partition",
    fontsize=legend_size,
    title_fontsize=legend_size
)

plt.tight_layout()

# ======================
# SAVE
# ======================
out_png = r"path-to-output-image.png"
#out_pdf = output_dir / "class_pixels_stacked_by_partition.pdf"

plt.savefig(out_png, dpi=300)
#plt.savefig(out_pdf)
plt.show()

print(f"Saved PNG: {out_png}")
#print(f"Saved PDF: {out_pdf}")
