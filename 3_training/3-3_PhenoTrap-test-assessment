
from PIL import Image, ImageFile
import os
import re
import json
from pathlib import Path
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp import autocast
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==========================================================
# USER SETTINGS
# ==========================================================
csv_file = r"path-to-image-index.csv"
images_root = r"path-to-image-directory"
masks_root = r"path-to-mask-directory"


# Set this to the NEW trained model's best checkpoint before running.
checkpoint_path = r"path-to-checkpoint.pth"
# Optional explicit CSV column; None auto-detects common column names.
SITE_COLUMN = None
# Optional override if the training config is stored elsewhere.
TRAINING_CONFIG = None

output_dir = r"path-to-results-output"

os.makedirs(output_dir, exist_ok=True)


# ----------------------------------------------------------
# PARTITIONS TO EVALUATE
#
# ["test"]         = test only
# ["val"]          = validation only
# ["val", "test"]  = both in one Excel workbook
# ----------------------------------------------------------

eval_partitions = ["test"]


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# ==========================================================
# CLASS SETTINGS
# ==========================================================

RAW_IGNORE_VALUE = 255
ignore_index = RAW_IGNORE_VALUE

num_classes = 14

class_names = ['background', 'sky', 'snow_ice', 'bare', 'ericaceous_shrub', 'briar_seedling', 'graminoid', 'forb', 'fern', 'cryptogam', 'vine', 'broadleaf_evergreen', 'conifer', 'broadleaf_deciduous']

# Use the exact supported class set saved during training, not test frequencies.
config_path = Path(TRAINING_CONFIG) if TRAINING_CONFIG else Path(checkpoint_path).with_name(
    Path(checkpoint_path).name.removesuffix("_best.pth") + "_config.json"
)
if not Path(checkpoint_path).is_file():
    raise FileNotFoundError(f"Set checkpoint_path to your new best model: {checkpoint_path}")
with config_path.open() as f:
    training_config = json.load(f)
if training_config["data"]["num_classes"] != num_classes:
    raise ValueError("Training config does not match the 14-class taxonomy")
supported_classes = training_config["data"]["supported_classes"]
if not supported_classes or any(not isinstance(c, int) or not 1 <= c < num_classes for c in supported_classes) or len(set(supported_classes)) != len(supported_classes):
    raise ValueError(f"Invalid supported classes: {supported_classes}")
print(f"Supported classes from {config_path}: {supported_classes}")

assert len(class_names) == num_classes


# ==========================================================
# MASK ENCODING
# ==========================================================

def encode_mask(x):

    arr = np.array(x, dtype=np.int64)

    # Invalid class IDs become ignore pixels
    arr[
        (arr != RAW_IGNORE_VALUE)
        & (arr >= num_classes)
    ] = RAW_IGNORE_VALUE

    return arr


# ==========================================================
# DATASET
# ==========================================================

class ForestSegmentationEval(Dataset):

    def __init__(self, df, images_root, masks_root):

        self.df = df.reset_index(drop=True)

        self.images_root = images_root
        self.masks_root = masks_root

        self.image_transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # --------------------------------------------------
        # Image filename
        # --------------------------------------------------

        image_filename = str(row["image_name"]).strip()

        if not image_filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            image_filename += ".jpg"

        image_path = os.path.join(
            self.images_root,
            image_filename,
        )

        # --------------------------------------------------
        # Mask filename
        # --------------------------------------------------

        mask_filename = str(row["mask_name"]).strip()

        if not mask_filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            mask_filename += ".png"

        mask_path = os.path.join(
            self.masks_root,
            mask_filename,
        )

        # --------------------------------------------------
        # Load image and mask
        # --------------------------------------------------

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        with Image.open(mask_path) as msk:
            mask = msk.convert("L")

        if image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch:\n"
                f"Image: {image_filename} {image.size}\n"
                f"Mask: {mask_filename} {mask.size}"
            )

        # --------------------------------------------------
        # Full-resolution evaluation
        # No resizing, cropping or random augmentation
        # --------------------------------------------------

        image = self.image_transform(image=np.array(image))["image"]

        mask = torch.from_numpy(
            encode_mask(mask)
        ).long()

        return image, mask, image_filename


# ==========================================================
# LOAD MODEL
# ==========================================================

# The checkpoint contains the trained model weights.
# No pretrained weights need to be downloaded again.
#
# aux_loss=True matches the original training architecture.

model = deeplabv3_resnet101(
    weights=None,
    weights_backbone=None,
    aux_loss=True,
)

model.classifier[4] = nn.Conv2d(
    256,
    num_classes,
    kernel_size=1,
)

state = torch.load(
    checkpoint_path,
    map_location="cpu",
)

# Supports raw state_dict and checkpoint dictionaries
if isinstance(state, dict) and "model_state_dict" in state:

    model.load_state_dict(
        state["model_state_dict"]
    )

else:

    model.load_state_dict(state)

model = model.to(device)
model.eval()

print(f"Loaded checkpoint: {checkpoint_path}")


# ==========================================================
# LOAD DATASET INDEX
# ==========================================================

df = pd.read_csv(csv_file)

df["partition"] = (
    df["partition"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# Site codes are read from CSV metadata, never from a fixed site list.
SITE_PATTERN = re.compile(r"(?<![A-Z0-9])[A-Z]{2}-[A-Z0-9]{2,}(?![A-Z0-9])")


def detect_sites(frame):
    columns = {str(c).strip().lower(): c for c in frame.columns}
    if SITE_COLUMN is not None:
        if SITE_COLUMN not in frame.columns:
            raise ValueError(f"Missing site column: {SITE_COLUMN}")
        explicit = SITE_COLUMN
    else:
        explicit = next((columns[c] for c in ("dataset_code", "site_code", "dataset", "site", "location_code", "location") if c in columns), None)
    result = []
    for idx, row in frame.iterrows():
        if explicit is not None:
            value = str(row[explicit]).strip().upper()
            if not SITE_PATTERN.fullmatch(value):
                raise ValueError(f"Invalid site code in {explicit}, row {idx}: {value!r}")
            result.append(value)
        else:
            # Fall back to codes embedded in CSV deployment IDs or filenames.
            codes = set()
            for col in ("deployment_id", "image_name", "mask_name"):
                if col in frame.columns:
                    codes.update(SITE_PATTERN.findall(str(row[col]).upper()))
            if len(codes) != 1:
                raise ValueError(f"Expected one site code at row {idx}; found {sorted(codes)}. Set SITE_COLUMN to your dataset code column.")
            result.append(next(iter(codes)))
    return result


def get_partition_df(df, partition_name):

    partition_name = partition_name.lower()

    if partition_name == "test":

        return df[
            df["partition"] == "test"
        ].reset_index(drop=True)

    if partition_name in ["val", "valid", "validation"]:

        return df[
            df["partition"].isin(
                ["val", "valid", "validation"]
            )
        ].reset_index(drop=True)

    if partition_name == "train":

        return df[
            df["partition"] == "train"
        ].reset_index(drop=True)

    if partition_name == "all":

        return df.reset_index(drop=True)

    raise ValueError(
        f"Unknown partition: {partition_name}"
    )


# ==========================================================
# CONFUSION MATRIX HELPERS
# ==========================================================

def empty_confusion():

    return torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )


def update_confusion(conf, preds, masks):

    # Exclude ignore pixels
    valid = masks != ignore_index

    gt = masks[valid].reshape(-1)
    pr = preds[valid].reshape(-1)

    if gt.numel() == 0:
        return conf

    k = gt * num_classes + pr

    binc = torch.bincount(
        k,
        minlength=num_classes * num_classes,
    )

    conf += binc.reshape(
        num_classes,
        num_classes,
    ).cpu()

    return conf


# ==========================================================
# METRIC CALCULATIONS
# ==========================================================

def safe_divide(numerator, denominator):

    numerator = np.asarray(
        numerator,
        dtype=np.float64,
    )

    denominator = np.asarray(
        denominator,
        dtype=np.float64,
    )

    return np.divide(
        numerator,
        denominator,
        out=np.full(
            numerator.shape,
            np.nan,
            dtype=np.float64,
        ),
        where=denominator > 0,
    )


def safe_mean(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    valid = ~np.isnan(values)

    if not np.any(valid):
        return np.nan

    return float(np.mean(values[valid]))


def fwiou_over_subset(iou_per_class, confusion, class_ids):
    ids = np.asarray(list(class_ids), dtype=np.int64)
    support = np.asarray(confusion).sum(axis=1)[ids].astype(np.float64)
    iou = np.asarray(iou_per_class, dtype=np.float64)[ids]
    valid = (support > 0) & np.isfinite(iou)
    total = support[valid].sum()
    if total == 0:
        return float("nan")
    return float(np.sum(support[valid] * iou[valid]) / total)


def compute_metrics(confusion):

    confusion = confusion.astype(np.int64)

    # ------------------------------------------------------
    # True positives, false positives, false negatives
    # ------------------------------------------------------

    tp = np.diag(confusion)

    gt_pixels = confusion.sum(axis=1)

    predicted_pixels = confusion.sum(axis=0)

    fp = predicted_pixels - tp

    fn = gt_pixels - tp

    total = confusion.sum()

    tn = total - tp - fp - fn

    # ------------------------------------------------------
    # Per-class IoU
    # ------------------------------------------------------

    iou = safe_divide(
        tp,
        tp + fp + fn,
    )

    # ------------------------------------------------------
    # Per-class precision
    # ------------------------------------------------------

    precision = safe_divide(
        tp,
        tp + fp,
    )

    # ------------------------------------------------------
    # Per-class recall
    # ------------------------------------------------------

    recall = safe_divide(
        tp,
        tp + fn,
    )

    # ------------------------------------------------------
    # Per-class F1
    # ------------------------------------------------------

    f1 = safe_divide(
        2 * tp,
        2 * tp + fp + fn,
    )

    # ------------------------------------------------------
    # Per-class one-vs-rest accuracy
    # ------------------------------------------------------

    if total > 0:

        accuracy = (
            tp + tn
        ).astype(np.float64) / total

    else:

        accuracy = np.full(
            num_classes,
            np.nan,
        )

    # ------------------------------------------------------
    # Mean metrics
    # ------------------------------------------------------

    mIoU = safe_mean(iou)

    mean_precision = safe_mean(precision)

    mean_recall = safe_mean(recall)

    mean_f1 = safe_mean(f1)

    mean_accuracy = safe_mean(accuracy)

    # ------------------------------------------------------
    # Overall pixel accuracy
    # ------------------------------------------------------

    pixel_accuracy = (
        tp.sum() / total
        if total > 0
        else np.nan
    )

    # ------------------------------------------------------
    # Frequency-weighted IoU
    # ------------------------------------------------------

    if total > 0:

        freq = gt_pixels / total

        FWIoU = np.nansum(freq * iou)

    else:

        FWIoU = np.nan

    # ------------------------------------------------------
    # Per-class results
    # ------------------------------------------------------

    rows = []

    for cid in range(num_classes):

        rows.append({

            "class_id": cid,

            "class_name": class_names[cid],
            "is_supported_class": int(cid in supported_classes),

            "gt_pixels": int(gt_pixels[cid]),

            "predicted_pixels": int(
                predicted_pixels[cid]
            ),

            "tp": int(tp[cid]),

            "fp": int(fp[cid]),

            "fn": int(fn[cid]),

            "tn": int(tn[cid]),

            "iou": iou[cid],

            "precision": precision[cid],

            "recall": recall[cid],

            "f1": f1[cid],

            "accuracy_one_vs_rest": accuracy[cid],

        })

    return {

        "mIoU": mIoU,
        "mIoU_supported": safe_mean(iou[supported_classes]),
        "FWIoU_supported": fwiou_over_subset(iou, confusion, supported_classes),

        "FWIoU": FWIoU,

        "pixel_accuracy": pixel_accuracy,

        "mean_accuracy": mean_accuracy,

        "mean_precision": mean_precision,

        "mean_recall": mean_recall,

        "mean_f1": mean_f1,

        "per_class_df": pd.DataFrame(rows),

    }


# ==========================================================
# CONFUSION MATRIX — PIXEL COUNTS
# ==========================================================

def confusion_to_count_dataframe(confusion):

    row_labels = [
        f"gt_{name}"
        for name in class_names
    ]

    col_labels = [
        f"pred_{name}"
        for name in class_names
    ]

    df_conf = pd.DataFrame(
        confusion,
        index=row_labels,
        columns=col_labels,
    )

    df_conf.insert(
        0,
        "ground_truth_class",
        row_labels,
    )

    df_conf["row_total_gt_pixels"] = (
        confusion.sum(axis=1)
    )

    return df_conf.reset_index(drop=True)


# ==========================================================
# CONFUSION MATRIX — ROW PERCENTAGES
#
# Each ground-truth class row sums to 100%.
# Diagonal = per-class recall (%).
# ==========================================================

def confusion_to_row_percent_dataframe(confusion):

    row_labels = [
        f"gt_{name}"
        for name in class_names
    ]

    col_labels = [
        f"pred_{name}"
        for name in class_names
    ]

    row_sums = confusion.sum(
        axis=1,
        keepdims=True,
    )

    pct = np.divide(

        confusion * 100.0,

        row_sums,

        out=np.zeros_like(
            confusion,
            dtype=np.float64,
        ),

        where=row_sums > 0,

    )

    df_conf = pd.DataFrame(
        pct,
        index=row_labels,
        columns=col_labels,
    )

    df_conf.insert(
        0,
        "ground_truth_class",
        row_labels,
    )

    df_conf["row_total_gt_pixels"] = (
        row_sums.flatten()
    )

    return df_conf.reset_index(drop=True)


# ==========================================================
# CONFUSION MATRIX — COLUMN PERCENTAGES
#
# Each predicted class column sums to 100%.
# Diagonal = per-class precision (%).
# ==========================================================

def confusion_to_column_percent_dataframe(confusion):

    row_labels = [
        f"gt_{name}"
        for name in class_names
    ]

    col_labels = [
        f"pred_{name}"
        for name in class_names
    ]

    col_sums = confusion.sum(
        axis=0,
        keepdims=True,
    )

    pct = np.divide(

        confusion * 100.0,

        col_sums,

        out=np.zeros_like(
            confusion,
            dtype=np.float64,
        ),

        where=col_sums > 0,

    )

    df_conf = pd.DataFrame(
        pct,
        index=row_labels,
        columns=col_labels,
    )

    df_conf.insert(
        0,
        "ground_truth_class",
        row_labels,
    )

    return df_conf.reset_index(drop=True)


# ==========================================================
# EVALUATE ONE PARTITION
# ==========================================================

def evaluate_partition(partition_name):

    eval_df = get_partition_df(
        df,
        partition_name,
    )

    if len(eval_df) == 0:

        print(
            f"Skipping {partition_name}: no rows found."
        )

        return None

    print(
        f"\nEvaluating {len(eval_df)} images "
        f"from partition: {partition_name}"
    )

    # ------------------------------------------------------
    # Dataset and loader
    # ------------------------------------------------------

    eval_set = ForestSegmentationEval(
        eval_df,
        images_root,
        masks_root,
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------
    # Separate confusion matrices
    # ------------------------------------------------------

    sites = detect_sites(eval_df)
    groups = {name: empty_confusion() for name in ["overall", *sorted(set(sites))]}
    group_image_counts = {name: 0 for name in groups}
    print("Detected sites:", pd.Series(sites).value_counts().to_dict())

    # ------------------------------------------------------
    # NEW: Store individual image IoU results
    # ------------------------------------------------------

    per_image_rows = []

    # ------------------------------------------------------
    # Inference
    # ------------------------------------------------------

    model.eval()

    with torch.no_grad():

        for idx, (images, masks, filenames) in enumerate(tqdm(
            eval_loader,
            desc=f"Evaluating {partition_name}",
        )):

            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            )

            with autocast(
                "cuda",
                enabled=(device.type == "cuda"),
            ):

                logits = model(images)["out"]

            preds = torch.argmax(
                logits,
                dim=1,
            )

            # ----------------------------------------------
            # Image filename and site
            # ----------------------------------------------

            original_filename = os.path.basename(
                filenames[0]
            )

            site = sites[idx]

            # ----------------------------------------------
            # Calculate image confusion matrix
            # ----------------------------------------------

            image_confusion = update_confusion(
                empty_confusion(),
                preds,
                masks,
            )

            # ----------------------------------------------
            # NEW: Calculate IoU for this individual image
            # ----------------------------------------------

            image_conf_np = image_confusion.numpy()

            tp = np.diag(image_conf_np)

            gt_pixels = image_conf_np.sum(axis=1)

            predicted_pixels = image_conf_np.sum(axis=0)

            fp = predicted_pixels - tp

            fn = gt_pixels - tp

            image_iou = safe_divide(
                tp,
                tp + fp + fn,
            )

            # ----------------------------------------------
            # NEW: Create one row per image
            # ----------------------------------------------

            image_row = {

                "partition": partition_name,

                "image_name": original_filename,

                "site": site,
                "deployment_id": eval_df.iloc[idx].get("deployment_id", ""),
                "mask_name": eval_df.iloc[idx]["mask_name"],

                "valid_gt_pixels": int(
                    image_conf_np.sum()
                ),

            }

            # Add IoU for every class

            for cid in range(num_classes):

                image_row[
                    class_names[cid]
                ] = image_iou[cid]

            # ----------------------------------------------
            # Add image-level mIoU
            # ----------------------------------------------

            image_row["mIoU"] = safe_mean(
                image_iou
            )

            # ----------------------------------------------
            # Add image-level FWIoU
            # ----------------------------------------------

            total_pixels = gt_pixels.sum()

            if total_pixels > 0:

                freq = gt_pixels / total_pixels

                image_row["FWIoU"] = float(
                    np.nansum(freq * image_iou)
                )

            else:

                image_row["FWIoU"] = np.nan

            # ----------------------------------------------
            # Add number of classes with defined IoU
            # ----------------------------------------------

            image_row["n_valid_classes"] = int(
                np.sum(~np.isnan(image_iou))
            )

            # ----------------------------------------------
            # Overall
            # ----------------------------------------------

            image_row["pixel_accuracy"] = float(tp.sum() / total_pixels) if total_pixels else np.nan
            image_row["mIoU_supported"] = safe_mean(image_iou[supported_classes])
            image_row["FWIoU_supported"] = fwiou_over_subset(image_iou, image_conf_np, supported_classes)
            for cid, name in enumerate(class_names):
                image_row[f"gt_pixels_{name}"] = int(gt_pixels[cid])
            per_image_rows.append(image_row)

            groups["overall"] += image_confusion

            group_image_counts["overall"] += 1

            groups[site] += image_confusion
            group_image_counts[site] += 1

    # ------------------------------------------------------
    # NEW: Convert per-image results to DataFrame
    # ------------------------------------------------------

    if len(per_image_rows) != len(eval_df):
        raise RuntimeError("Per-image result count does not match index")
    if sum(group_image_counts[k] for k in groups if k != "overall") != len(eval_df):
        raise RuntimeError("Site counts do not match index")
    if not torch.equal(sum((groups[k] for k in groups if k != "overall"), empty_confusion()), groups["overall"]):
        raise RuntimeError("Site confusion matrices do not sum to overall")
    if sum(r["valid_gt_pixels"] for r in per_image_rows) != int(groups["overall"].sum()):
        raise RuntimeError("Per-image and overall pixel totals disagree")
    per_image_df = pd.DataFrame(
        per_image_rows
    )

    # ------------------------------------------------------
    # Print group counts
    # ------------------------------------------------------

    print(f"\nImage counts for {partition_name}:")

    for group_name, count in group_image_counts.items():

        print(f"  {group_name}: {count}")

    # ------------------------------------------------------
    # Return aggregated and individual image results
    # ------------------------------------------------------

    return groups, group_image_counts, per_image_df

# ==========================================================
# SAVE ONE XLSX WORKBOOK
# ==========================================================

timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
    "%Y%m%d_%H%M%S"
)

partition_label = "_".join(eval_partitions)

xlsx_path = os.path.join(
    output_dir,
    f"segmentation_metrics_autosites_"
    f"{partition_label}_{timestamp}.xlsx",
)

summary_rows = []

with pd.ExcelWriter(
    xlsx_path,
    engine="openpyxl",
) as writer:

    # ------------------------------------------------------
    # Loop through selected partitions
    # ------------------------------------------------------

    for partition_name in eval_partitions:

        result = evaluate_partition(
            partition_name
        )

        if result is None:
            continue

        # --------------------------------------------------
        # NEW: Unpack individual image results as well
        # --------------------------------------------------

        (
            groups,
            group_image_counts,
            per_image_df,
        ) = result

        # --------------------------------------------------
        # NEW: Save individual image IoU sheet
        # --------------------------------------------------

        per_image_df.to_excel(

            writer,

            sheet_name=f"{partition_name}_PerImage_IoU",

            index=False,

        )

        print(
            f"Saved per-image IoU for "
            f"{len(per_image_df)} {partition_name} images."
        )

        # --------------------------------------------------
        # Loop through overall and detected sites
        # --------------------------------------------------

        for group_name, conf_tensor in groups.items():

            image_count = group_image_counts[
                group_name
            ]

            if image_count == 0:

                print(
                    f"Skipping {partition_name} / "
                    f"{group_name}: no matching images."
                )

                continue

            confusion = conf_tensor.numpy()

            metrics = compute_metrics(
                confusion
            )

            # ----------------------------------------------
            # Summary metrics
            # ----------------------------------------------

            summary_rows.append({

                "partition": partition_name,

                "group": group_name,

                "image_count": image_count,

                "valid_gt_pixels": int(
                    confusion.sum()
                ),

                "pixel_accuracy": metrics[
                    "pixel_accuracy"
                ],

                "mean_accuracy": metrics[
                    "mean_accuracy"
                ],

                "mIoU": metrics["mIoU"],

                "FWIoU": metrics["FWIoU"],
                "mIoU_supported": metrics["mIoU_supported"],
                "FWIoU_supported": metrics["FWIoU_supported"],

                "mean_precision": metrics[
                    "mean_precision"
                ],

                "mean_recall": metrics[
                    "mean_recall"
                ],

                "mean_f1": metrics["mean_f1"],

            })

            # ----------------------------------------------
            # Sheet prefix
            # ----------------------------------------------

            sheet_prefix = (
                f"{partition_name}_"
                f"{'all' if group_name == 'overall' else group_name}"
            )

            # ----------------------------------------------
            # Per-class metrics
            # ----------------------------------------------

            metrics["per_class_df"].to_excel(

                writer,

                sheet_name=f"{sheet_prefix}_Metrics",

                index=False,

            )

            # ----------------------------------------------
            # Confusion matrix — counts
            # ----------------------------------------------

            confusion_to_count_dataframe(
                confusion
            ).to_excel(

                writer,

                sheet_name=f"{sheet_prefix}_Cnt",

                index=False,

            )

            # ----------------------------------------------
            # Confusion matrix — row percentages
            # ----------------------------------------------

            confusion_to_row_percent_dataframe(
                confusion
            ).to_excel(

                writer,

                sheet_name=f"{sheet_prefix}_RowPct",

                index=False,

            )

            # ----------------------------------------------
            # Confusion matrix — column percentages
            # ----------------------------------------------

            confusion_to_column_percent_dataframe(
                confusion
            ).to_excel(

                writer,

                sheet_name=f"{sheet_prefix}_ColPct",

                index=False,

            )

            print(
                f"Saved {partition_name} / "
                f"{group_name} results."
            )

    # ------------------------------------------------------
    # Summary sheet
    # ------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_excel(

        writer,

        sheet_name="Summary",

        index=False,

    )


# ==========================================================
# SAVE SUMMARY CSV
# ==========================================================

summary_csv = os.path.join(

    output_dir,

    f"summary_metrics_autosites_"
    f"{partition_label}_{timestamp}.csv",

)

summary_df.to_csv(

    summary_csv,

    index=False,

)


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\nDone.")

print(
    f"Saved Excel results to:\n{xlsx_path}"
)

print(
    f"Saved summary CSV to:\n{summary_csv}"
)

print("\nSummary:")

print(
    summary_df.to_string(index=False)
)
