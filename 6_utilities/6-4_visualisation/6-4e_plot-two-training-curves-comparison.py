
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# Paths
# ======================
simple_xlsx = r"path-to-epoch-results-1.xlsx"
complex_xlsx = r"path-to-epoch-results-2.xlsx"

output_dir = Path(complex_xlsx).parent / "simple_vs_complex_training_curves"
output_dir.mkdir(exist_ok=True)

# ======================
# Settings
# ======================
miou_col = "val_mIoU_supported"   # change to "val_mIoU_all" if needed
loss_col = "train_loss"
epoch_col = "epoch"

simple_name = "Simple"
complex_name = "Complex"

simple_color = "darkblue"
complex_color = "darkred"

title_size = 20
axis_label_size = 16
tick_size = 14
legend_size = 13

# ======================
# Read files
# ======================
def read_training_file(xlsx_file):
    df = pd.read_excel(xlsx_file)
    df.columns = df.columns.str.strip()

    df[epoch_col] = pd.to_numeric(df[epoch_col], errors="coerce")
    df[loss_col] = pd.to_numeric(df[loss_col], errors="coerce")
    df[miou_col] = pd.to_numeric(df[miou_col], errors="coerce")

    df = df.dropna(subset=[epoch_col, loss_col, miou_col])
    df = df.sort_values(epoch_col)

    return df

simple_df = read_training_file(simple_xlsx)
complex_df = read_training_file(complex_xlsx)

# ======================
# Find best epoch
# ======================
def get_best_epoch(df):
    if "is_best_epoch" in df.columns:
        best_rows = df[df["is_best_epoch"] == 1]

        if len(best_rows) > 0:
            return best_rows.iloc[0]

    # Fallback: highest validation mIoU
    idx = df[miou_col].idxmax()
    return df.loc[idx]

simple_best = get_best_epoch(simple_df)
complex_best = get_best_epoch(complex_df)

# ======================
# 1. Validation mIoU comparison
# ======================
plt.figure(figsize=(12, 8))

plt.plot(
    simple_df[epoch_col],
    simple_df[miou_col],
    color=simple_color,
    label=f"mIoU - {simple_name}"
)

plt.plot(
    complex_df[epoch_col],
    complex_df[miou_col],
    color=complex_color,
    label=f"mIoU - {complex_name}"
)

plt.plot(
    [simple_best[epoch_col], simple_best[epoch_col]],
    [0.3, simple_best[miou_col]],
    color=simple_color,
    linestyle=":",
    linewidth=2,
    label=f"Best Epoch - {simple_name}"
)

plt.plot(
    [complex_best[epoch_col], complex_best[epoch_col]],
    [0.3, complex_best[miou_col]],
    color=complex_color,
    linestyle=":",
    linewidth=2,
    label=f"Best Epoch - {complex_name}"
)





#plt.axvline(
#    simple_best[epoch_col],
#    color=simple_color,
#    linestyle=":",
#    linewidth=2,
#    label=f"Best Epoch - {simple_name}"
#    #label = f"{simple_name} Best Epoch ({int(simple_best[epoch_col])})"
#)
#
#plt.axvline(
#    complex_best[epoch_col],
#    color=complex_color,
#    linestyle=":",
#    linewidth=2,
#    label=f"Best Epoch - {complex_name}"
#    #label=f"{complex_name} Best Epoch ({int(complex_best[epoch_col])})"
#)

plt.scatter(
    simple_best[epoch_col],
    simple_best[miou_col],
    color=simple_color,
    s=25,
    zorder=5
)

plt.scatter(
    complex_best[epoch_col],
    complex_best[miou_col],
    color=complex_color,
    s=25,
    zorder=5
)

plt.xlabel("Epoch", fontsize=axis_label_size)
plt.ylabel("Validation mIoU Supported", fontsize=axis_label_size)
plt.ylim(0.3, 1.0)
plt.title("Validation mIoU Comparison: Simple vs Complex", fontsize=title_size)
plt.tick_params(axis="both", labelsize=tick_size)
plt.legend(fontsize=legend_size)
plt.tight_layout()

output_png = output_dir / "validation_miou_simple_vs_complex.png"
plt.savefig(output_png, dpi=300)
plt.show()

print(f"Saved: {output_png}")

# ======================
# 2. Training loss comparison
# ======================
plt.figure(figsize=(12, 8))

plt.plot(
    simple_df[epoch_col],
    simple_df[loss_col],
    color=simple_color,
    label=f"{simple_name} training loss"
)

plt.plot(
    complex_df[epoch_col],
    complex_df[loss_col],
    color=complex_color,
    label=f"{complex_name} training loss"
)

plt.axvline(
    simple_best[epoch_col],
    color=simple_color,
    linestyle=":",
    linewidth=2,
    label=f"{simple_name} Best Epoch"
)

plt.axvline(
    complex_best[epoch_col],
    color=complex_color,
    linestyle=":",
    linewidth=2,
    label=f"{complex_name} Best Epoch"
)

plt.xlabel("Epoch", fontsize=axis_label_size)
plt.ylabel("Training Loss", fontsize=axis_label_size)
plt.title("Training Loss Comparison: Simple vs Complex", fontsize=title_size)
plt.tick_params(axis="both", labelsize=tick_size)
plt.legend(fontsize=legend_size)
plt.tight_layout()

output_png = output_dir / "training_loss_simple_vs_complex.png"
plt.savefig(output_png, dpi=300)
plt.show()

print(f"Saved: {output_png}")

# ======================
# 3. Side-by-side training curves
# ======================
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=False)

models = [
    (simple_df, simple_name, simple_color, axes[0]),
    (complex_df, complex_name, complex_color, axes[1])
]

for df, model_name, colour, ax1 in models:
    ax2 = ax1.twinx()

    loss_line = ax1.plot(
        df[epoch_col],
        df[loss_col],
        color=colour,
        linestyle="-",
        label="Training Loss"
    )

    miou_line = ax2.plot(
        df[epoch_col],
        df[miou_col],
        color="black",
        linestyle="--",
        label="Validation mIoU"

    )

    ax1.set_title(model_name, fontsize=title_size)
    ax1.set_xlabel("Epoch", fontsize=axis_label_size)
    ax1.set_ylabel("Training Loss", fontsize=axis_label_size)
    ax2.set_ylabel("Validation mIoU Supported", fontsize=axis_label_size)
    ax2.set_ylim(0, 1)

    ax1.tick_params(axis="both", labelsize=tick_size)
    ax2.tick_params(axis="y", labelsize=tick_size)

    lines = loss_line + miou_line
    labels = [line.get_label() for line in lines]

    ax1.legend(
        lines,
        labels,
        loc="best",
        fontsize=legend_size
    )

fig.suptitle(
    "Training Curves: Simple vs Complex Model",
    fontsize=title_size + 2
)

plt.tight_layout()

output_png = output_dir / "side_by_side_training_curves_simple_vs_complex.png"
plt.savefig(output_png, dpi=300)
plt.show()

print(f"Saved: {output_png}")
