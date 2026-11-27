
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# Paths
# ======================
xlsx_file = r"path-to-epoch-results.xlsx"
output_dir = Path(xlsx_file).parent / "training_curve-Complex"
output_dir.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Arial"
# ======================
# Font sizes
# ======================
title_size = 18
axis_label_size = 20
tick_size = 16
legend_size = 16

# ======================
# Read Excel file
# ======================
df = pd.read_excel(xlsx_file)

# Clean column names
df.columns = df.columns.str.strip()

# Make sure values are numeric
df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
df["val_mIoU_supported"] = pd.to_numeric(df["val_mIoU_supported"], errors="coerce")

df = df.dropna(subset=["epoch", "train_loss", "val_mIoU_supported"])
df = df.sort_values("epoch")

# ======================
# Line colours
# ======================
train_loss_color = "#50ad9f"
val_miou_color = "#bc272d"
#best_epoch_color = "darkgreen"

# ======================
# Plot training curve
# ======================
fig, ax1 = plt.subplots(figsize=(12, 9))

# Training loss
ax1.plot(
    df["epoch"],
    df["train_loss"],
#    marker="o",
    color=train_loss_color,
    label="Training loss"
)

# Validation mIoU
ax2 = ax1.twinx()
ax2.plot(
    df["epoch"],
    df["val_mIoU_supported"],
#    marker="s",
#    linestyle="--",
    color=val_miou_color,
    label="Validation mIoU"
)
ax2.set_ylabel("Validation mIoU Supported")
ax2.set_ylim(0.35, 0.75)

# Put grid behind the bars
ax1.set_axisbelow(True)

# Add vertical gridlines
ax1.grid(
    axis="x",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)

# ======================
# Best epoch line
# ======================
#if "is_best_epoch" in df.columns:
#    df["is_best_epoch"] = pd.to_numeric(df["is_best_epoch"], errors="coerce").fillna(0)
#
#    best_rows = df[df["is_best_epoch"] == 1]
#
#    for _, row in best_rows.iterrows():
#        ax1.axvline(
#            x=row["epoch"],
#            linestyle=":",
#            linewidth=2,
#            color=best_epoch_color,
#            label=f"Best Epoch"
#        #label = f"Best Epoch: {int(row['epoch'])}"
#        )

# ======================
# Title and legend
# ======================

# Axis labels
ax1.set_xlabel("Epoch", fontsize=axis_label_size)
ax1.set_ylabel("Training loss", fontsize=axis_label_size)

ax2.set_ylabel(
    "Validation mIoU (supported)",
    fontsize=axis_label_size
)

# Tick sizes
ax1.tick_params(axis="both", labelsize=tick_size)
ax2.tick_params(axis="y", labelsize=tick_size)

## Title
#ax1.set_title(
#    "CNN Training Curve: Loss and Validation mIoU - SIMPLE",
#    fontsize=title_size
#)
#
# Get legend items
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

# Put legend in custom order:
# Training Loss, Validation mIoU, Best Epoch
legend_lines = [lines_1[0], lines_2[0]]
legend_labels = [labels_1[0], labels_2[0]]

#legend_lines = [lines_1[0], lines_2[0], lines_1[1]]
#legend_labels = [labels_1[0], labels_2[0], labels_1[1]

ax1.legend(
    legend_lines,
    legend_labels,
    loc="best",
    fontsize=legend_size
)

# ======================
# Save figure
# ======================
output_png = output_dir / "cnn_training_curve_loss_val_miou.png"
#output_pdf = output_dir / "cnn_training_curve_loss_val_miou.pdf"
plt.tight_layout()
plt.savefig(output_png, dpi=300)
#plt.savefig(output_pdf)

plt.show()

print(f"Saved PNG to: {output_png}")
#print(f"Saved PDF to: {output_pdf}")
