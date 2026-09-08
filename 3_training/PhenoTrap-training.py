from PIL import Image, ImageFile
import os
import json
import datetime
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from openpyxl import load_workbook
from openpyxl.styles import Alignment
from collections import defaultdict


# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ======================
# Paths
# ======================
csv_file = r"path-to-index.csv"
images_root = r"path-to-images-directory"
masks_root = r"path-to-masks-directory"
trial_root = os.path.dirname(masks_root) # models and log directory created within the same folder
models_dir = os.path.join(trial_root, "3-d_models") #rename for models directory
os.makedirs(models_dir, exist_ok=True)

logs_dir = os.path.join(trial_root, "3-e_training-outputs") 
os.makedirs(logs_dir, exist_ok=True)

log_xlsx_path = os.path.join(logs_dir, "experiment-log.xlsx")  # must exist (from template)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ======================
# Split & mask encoding
# ======================
RAW_IGNORE_VALUE = 255 # igmore all pixel vaölues of 255
num_classes = 14 # all classes including background
ignore_index = RAW_IGNORE_VALUE


def split_from_index(csv_path: str):
    df = pd.read_csv(csv_path)
    df["partition"] = df["partition"].astype(str).str.strip().str.lower()

    train_df = df[df["partition"] == "train"].reset_index(drop=True)
    val_df = df[df["partition"].isin(["val", "valid", "validation"])].reset_index(drop=True)
    test_df = df[df["partition"] == "test"].reset_index(drop=True)

    assert len(train_df) > 0, "Train split is empty"
    assert len(val_df) > 0, "Validation split is empty"
    assert len(test_df) > 0, "Test split is empty"

    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


def encode_mask(x) -> np.ndarray:
    arr = np.array(x, dtype=np.int64)
    # any invalid class id -> ignore
    arr[(arr != RAW_IGNORE_VALUE) & (arr >= num_classes)] = RAW_IGNORE_VALUE
    return arr


train_df, val_df, test_df = split_from_index(csv_file)
print(train_df["partition"].value_counts())
print(val_df["partition"].value_counts())
print(test_df["partition"].value_counts())

def check_deployment_leakage(train_df, val_df, test_df):
    if not all("deployment_id" in x.columns for x in [train_df, val_df, test_df]):
        print("WARNING: deployment_id column not found; skipping leakage check.")
        return

    train_ids = set(train_df["deployment_id"])
    val_ids = set(val_df["deployment_id"])
    test_ids = set(test_df["deployment_id"])

    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids

    if overlap_train_val:
        raise ValueError(f"Deployment leakage detected between train and val: {len(overlap_train_val)} shared IDs")
    if overlap_train_test:
        raise ValueError(f"Deployment leakage detected between train and test: {len(overlap_train_test)} shared IDs")
    if overlap_val_test:
        raise ValueError(f"Deployment leakage detected between val and test: {len(overlap_val_test)} shared IDs")

    print("No deployment overlap detected across train/val/test.")

check_deployment_leakage(train_df, val_df, test_df)


# ======================
# Augmentations
# ======================
# NOTE: you log TARGET_H/W but training uses crops (CROP_H/W)
TARGET_H = 1076
TARGET_W = 1536

CROP_H, CROP_W = 1000, 1200  # crops training images to size to preserve pixel resolution

train_aug = A.Compose(
    [
        A.PadIfNeeded(
            min_height=CROP_H,
            min_width=CROP_W,
            border_mode=0,
            value=0,
            mask_value=ignore_index,
        ),
        A.RandomCrop(height=CROP_H, width=CROP_W),
        A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.2),
        # lighter than before to reduce instability
        A.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            scale=(0.95, 1.05),
            rotate=(-7, 7),
            mode=0,
            p=0.4,
        ),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

val_aug = A.Compose(
    [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


# ======================
# Dataset
# ======================
class ForestSegmentation(Dataset):
    def __init__(self, df, images_root, masks_root, augmentation=None):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.masks_root = masks_root
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image_filename = str(row["image_name"]).strip()
            if not image_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_filename += ".jpg"
            image_path = os.path.join(self.images_root, image_filename)

            mask_filename = str(row["mask_name"]).strip()
            if not mask_filename.lower().endswith((".png", ".jpg", ".jpeg")):
                mask_filename += ".png"
            mask_path = os.path.join(self.masks_root, mask_filename)

            image = np.array(Image.open(image_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = encode_mask(mask)

            if self.augmentation is not None:
                augmented = self.augmentation(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"].long()
            else:
                image = transforms.ToTensor()(image)
                image = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )(image)
                mask = torch.from_numpy(mask).long()

            return image, mask

        except Exception as e:
            print(f"❌ Skipping sample at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self.df))


# ======================
# Supported-classes mIoU setup
# ======================


min_val_images_per_class = 5 # "high support" threshold
raw_val_set = ForestSegmentation(val_df, images_root, masks_root, augmentation=None)


def class_presence_counts(dataset):
    present = defaultdict(int)
    for i in range(len(dataset)):
        _, m = dataset[i]
        vals = set(m.unique().cpu().numpy().tolist())
        vals.discard(ignore_index)
        for v in vals:
            present[int(v)] += 1
    return present

val_presence = class_presence_counts(raw_val_set)
supported_classes = [cid for cid in range(1, num_classes) if val_presence.get(cid, 0) >= min_val_images_per_class]
#print("Supported classes:", supported_classes)
#print("Val presence (images per class):", dict(sorted(val_presence.items())))

# ======================
# Model
# ======================
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
model = model.to(device)


# ======================
# Loaders
# ======================
batch_size = 4  # Change depending on resource availability (4 was maximum available)
val_batch_size = 1  # full-resolution validation; safer for memory and variable image sizes

train_set = ForestSegmentation(train_df, images_root, masks_root, augmentation=train_aug)
val_set = ForestSegmentation(val_df, images_root, masks_root, augmentation=val_aug)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


# ======================
# Loss / Optim / Scheduler
# ======================


class_weights = torch.ones(num_classes, dtype=torch.float32, device=device)




# background
class_weights[0] = 0.25

# moderately rare (missing classes default to 1.0)
class_weights[1] = 1.25
class_weights[7] = 1.25
class_weights[8] = 1.5
class_weights[9] = 1.5

# rare
class_weights[10] = 2.5
class_weights[11] = 2.25

# dominant classes
class_weights[12] = 1.0
class_weights[13] = 1.0


# stabilize extremes
class_weights = torch.clamp(class_weights, 0.25, 2.5)


# Optional: label smoothing can improve generalization; it may slightly raise CE numerically.
use_label_smoothing = False
label_smoothing = 0.05

if use_label_smoothing:
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
else:
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

lr_backbone = 5e-5
lr_head = 5e-4  # slightly lower than 1e-3 for stability
weight_decay = 1e-5

optimizer = torch.optim.AdamW(
    [
        {"params": model.backbone.parameters(), "lr": lr_backbone},
        {"params": model.classifier.parameters(), "lr": lr_head},
    ],
    weight_decay=weight_decay,
)

num_epochs = 1
scaler = GradScaler("cuda")

warmup_epochs = 5
min_lr_backbone = 1e-6
min_lr_head = 1e-5





def set_lrs(epoch: int):
    if epoch < warmup_epochs:
        warm = (epoch + 1) / warmup_epochs
        b_lr = lr_backbone * (0.1 + 0.9 * warm)
        h_lr = lr_head * (0.1 + 0.9 * warm)
    else:
        t = (epoch - warmup_epochs) / max(1, (num_epochs - warmup_epochs - 1))
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        b_lr = min_lr_backbone + (lr_backbone - min_lr_backbone) * cos
        h_lr = min_lr_head + (lr_head - min_lr_head) * cos

    optimizer.param_groups[0]["lr"] = b_lr
    optimizer.param_groups[1]["lr"] = h_lr


# ======================
# Evaluate (mIoU)
# ======================
def miou_over_subset(iou_per_class: np.ndarray, class_ids):
    vals = []
    for cid in class_ids:
        v = iou_per_class[cid]
        if not np.isnan(v):
            vals.append(float(v))
    return float(np.mean(vals)) if len(vals) else float("nan")


@torch.no_grad()
def evaluate(model, loader, num_classes: int, ignore_index: int, device):
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast("cuda"):
            logits = model(images)["out"]
        preds = torch.argmax(logits, dim=1)

        valid = masks != ignore_index
        gt = masks[valid].view(-1)
        pr = preds[valid].view(-1)

        k = gt * num_classes + pr
        binc = torch.bincount(k, minlength=num_classes * num_classes)
        conf += binc.view(num_classes, num_classes).cpu()

    tp = conf.diag()
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = tp + fp + fn

    iou = torch.where(denom > 0, tp.float() / denom.float(), torch.nan)
    miou = torch.nanmean(iou).item()

    return {"mIoU": miou, "IoU_per_class": iou.numpy(), "confusion": conf.numpy()}


@torch.no_grad()
def ignore_fraction(loader, ignore_index, device):
    total = 0
    ign = 0
    for _, masks in loader:
        masks = masks.to(device)
        total += masks.numel()
        ign += (masks == ignore_index).sum().item()
    return ign / max(1, total)


frac = ignore_fraction(val_loader, ignore_index, device)
print(f"Validation ignore fraction: {frac:.3%}")


# ======================
# Excel logging helpers (auto-add missing columns)
# ======================
def ensure_excel_columns(xlsx_path: str, sheet_name: str, required_headers):
    wb = load_workbook(xlsx_path)
    if sheet_name not in wb.sheetnames:
        wb.close()
        raise ValueError(f"Sheet '{sheet_name}' not found in {xlsx_path}")
    ws = wb[sheet_name]

    existing = [c.value for c in ws[1]]
    existing_set = set([h for h in existing if h is not None])

    for h in required_headers:
        if h not in existing_set:
            ws.cell(row=1, column=ws.max_column + 1, value=h)
            existing_set.add(h)

    ws.freeze_panes = "A2"
    for col in range(1, ws.max_column + 1):
        ws.cell(row=1, column=col).alignment = Alignment(wrap_text=True, vertical="top")

    wb.save(xlsx_path)
    wb.close()


def append_run_to_excel(xlsx_path: str, row_dict: dict, sheet_name: str = "Runs"):
    wb = load_workbook(xlsx_path)
    if sheet_name not in wb.sheetnames:
        wb.close()
        raise ValueError(f"Sheet '{sheet_name}' not found in {xlsx_path}")

    ws = wb[sheet_name]
    headers = [c.value for c in ws[1]]
    values = [row_dict.get(h, "") for h in headers]
    ws.append(values)

    last_row = ws.max_row
    for col_idx in range(1, len(headers) + 1):
        ws.cell(row=last_row, column=col_idx).alignment = Alignment(wrap_text=True, vertical="top")

    wb.save(xlsx_path)
    wb.close()

def ensure_sheet_with_headers(xlsx_path: str, sheet_name: str, required_headers):
    wb = load_workbook(xlsx_path)

    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
    else:
        ws = wb.create_sheet(title=sheet_name)

    existing = []
    if ws.max_row >= 1:
        existing = [c.value for c in ws[1]]

    existing_set = set([h for h in existing if h is not None])

    # if sheet is empty, write headers from scratch
    if ws.max_row == 1 and ws["A1"].value is None:
        for col_idx, h in enumerate(required_headers, start=1):
            ws.cell(row=1, column=col_idx, value=h)
    else:
        for h in required_headers:
            if h not in existing_set:
                ws.cell(row=1, column=ws.max_column + 1, value=h)

    ws.freeze_panes = "A2"
    for col in range(1, ws.max_column + 1):
        ws.cell(row=1, column=col).alignment = Alignment(wrap_text=True, vertical="top")

    wb.save(xlsx_path)
    wb.close()


def append_rows_to_excel(xlsx_path: str, rows: list, sheet_name: str):
    if not rows:
        return

    wb = load_workbook(xlsx_path)
    if sheet_name not in wb.sheetnames:
        wb.close()
        raise ValueError(f"Sheet '{sheet_name}' not found in {xlsx_path}")

    ws = wb[sheet_name]
    headers = [c.value for c in ws[1]]

    for row_dict in rows:
        values = [row_dict.get(h, "") for h in headers]
        ws.append(values)

        last_row = ws.max_row
        for col_idx in range(1, len(headers) + 1):
            ws.cell(row=last_row, column=col_idx).alignment = Alignment(
                wrap_text=True, vertical="top"
            )

    wb.save(xlsx_path)
    wb.close()


def per_class_confusion_rows(confusion: np.ndarray, run_id: str, timestamp_utc: str,
                             supported_classes=None, ignore_index=255):
    rows = []
    total = confusion.sum()

    for cid in range(confusion.shape[0]):
        tp = int(confusion[cid, cid])
        fp = int(confusion[:, cid].sum() - tp)
        fn = int(confusion[cid, :].sum() - tp)
        tn = int(total - tp - fp - fn)

        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else np.nan

        denom_precision = tp + fp
        precision = tp / denom_precision if denom_precision > 0 else np.nan

        denom_recall = tp + fn
        recall = tp / denom_recall if denom_recall > 0 else np.nan

        denom_f1 = 2 * tp + fp + fn
        f1 = (2 * tp) / denom_f1 if denom_f1 > 0 else np.nan

        support = int(confusion[cid, :].sum())   # GT pixels for this class
        predicted = int(confusion[:, cid].sum()) # predicted pixels for this class

        rows.append({
            "run_id": run_id,
            "timestamp_utc": timestamp_utc,
            "class_id": cid,
            "is_background": int(cid == 0),
            "is_supported_class": int(cid in (supported_classes or [])),
            "support_pixels": support,
            "predicted_pixels": predicted,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "iou": float(iou) if not np.isnan(iou) else "",
            "precision": float(precision) if not np.isnan(precision) else "",
            "recall": float(recall) if not np.isnan(recall) else "",
            "f1": float(f1) if not np.isnan(f1) else "",
        })

    return rows

def write_dataframe_to_sheet(xlsx_path: str, df: pd.DataFrame, sheet_name: str):
    wb = load_workbook(xlsx_path)

    if sheet_name in wb.sheetnames:
        del wb[sheet_name]

    ws = wb.create_sheet(title=sheet_name)

    # headers
    for c_idx, col_name in enumerate(df.columns, start=1):
        ws.cell(row=1, column=c_idx, value=col_name)

    # rows
    for r_idx, row in enumerate(df.itertuples(index=False), start=2):
        for c_idx, value in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    ws.freeze_panes = "B2"

    for col in range(1, ws.max_column + 1):
        ws.cell(row=1, column=col).alignment = Alignment(wrap_text=True, vertical="top")

    for row in range(2, ws.max_row + 1):
        ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True, vertical="top")

    wb.save(xlsx_path)
    wb.close()


def confusion_to_count_dataframe(confusion: np.ndarray, run_id: str, class_names=None):
    n = confusion.shape[0]

    if class_names is None:
        class_names = [f"class_{i:02d}" for i in range(n)]

    row_labels = [f"gt_{class_names[i]}" for i in range(n)]
    col_labels = [f"pred_{class_names[i]}" for i in range(n)]

    df = pd.DataFrame(confusion, index=row_labels, columns=col_labels)
    df.insert(0, "run_id", run_id)

    # row totals = total GT pixels for each class
    numeric_cols = [c for c in df.columns if c != "run_id"]
    df["row_total_gt_pixels"] = df[numeric_cols].sum(axis=1)

    return df.reset_index(names="ground_truth_class")


def confusion_to_percent_dataframe(confusion: np.ndarray, run_id: str, class_names=None):
    n = confusion.shape[0]

    if class_names is None:
        class_names = [f"class_{i:02d}" for i in range(n)]

    row_labels = [f"gt_{class_names[i]}" for i in range(n)]
    col_labels = [f"pred_{class_names[i]}" for i in range(n)]

    row_sums = confusion.sum(axis=1, keepdims=True)
    pct = np.divide(
        confusion * 100.0,
        row_sums,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_sums > 0,
    )

    df = pd.DataFrame(pct, index=row_labels, columns=col_labels)
    df.insert(0, "run_id", run_id)
    df["row_total_gt_pixels"] = row_sums.flatten()

    return df.reset_index(names="ground_truth_class")

def confusion_to_column_percent_dataframe(confusion: np.ndarray, run_id: str, class_names=None):
    n = confusion.shape[0]

    if class_names is None:
        class_names = [f"class_{i:02d}" for i in range(n)]

    row_labels = [f"gt_{class_names[i]}" for i in range(n)]
    col_labels = [f"pred_{class_names[i]}" for i in range(n)]

    col_sums = confusion.sum(axis=0, keepdims=True)
    pct = np.divide(
        confusion * 100.0,
        col_sums,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=col_sums > 0,
    )

    df = pd.DataFrame(pct, index=row_labels, columns=col_labels)
    df.insert(0, "run_id", run_id)

    return df.reset_index(names="ground_truth_class")


# ======================
# Run metadata / config JSON
# ======================
run_id = datetime.datetime.utcnow().strftime("COMPLEX_%Y%m%d_%H%M%S")   # Adjust to rename model
timestamp_utc = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

ckpt_last_name = f"run_{run_id}_last.pth"
ckpt_best_name = f"run_{run_id}_best.pth"
ckpt_last_path = os.path.join(models_dir, ckpt_last_name)
ckpt_best_path = os.path.join(models_dir, ckpt_best_name)
config_path = os.path.join(models_dir, f"run_{run_id}_config.json")

config = {
    "run_id": run_id,
    "timestamp_utc": timestamp_utc,
    "experiment_name": "train_1",
    "paths": {
        "index_csv": csv_file,
        "images_root": images_root,
        "masks_root": masks_root,
        "models_dir": models_dir,
        "log_xlsx": log_xlsx_path,
        "checkpoint_last": ckpt_last_path,
        "checkpoint_best": ckpt_best_path,
    },
    "data": {
        "train_partition": "train",
        "val_partition": "validation",
        "test_partition": "test",
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "train_crop_size": [CROP_H, CROP_W],
        "validation_eval_size": "full_resolution_no_crop",
        "deployment_overlap_count": 0,
        "min_val_images_per_class": int(min_val_images_per_class),
        "supported_classes": supported_classes,
    },
    "train": {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "amp": True,
        "loss": {
            "name": "CrossEntropyLoss",
            "class_weights": class_weights.detach().cpu().tolist(),
            "label_smoothing": label_smoothing if use_label_smoothing else 0.0,
        },
        "optimizer": {
            "name": "AdamW",
            "lr_backbone": lr_backbone,
            "lr_head": lr_head,
            "weight_decay": weight_decay,
        },
        "scheduler": {
            "name": "cosine_warmup",
            "warmup_epochs": warmup_epochs,
            "min_lr_backbone": min_lr_backbone,
            "min_lr_head": min_lr_head,
        },
        "selection_metric": "mIoU_supported",
        "grad_clip_norm": 1.0,
    },
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Saved config: {config_path}")


# ======================
# Training loop
# ======================
patience = 30  # stops training if no improvement after 30 epochs
min_delta = 0.0005
no_improve = 0

selection_metric_name = "mIoU_supported"  # or "mIoU_all"

best_metric = -1.0
best_epoch = -1
best_iou_per_class = None
best_confusion = None
best_miou_all = None
best_miou_supported = None
final_train_loss = None

epoch_history = []
for epoch in range(num_epochs):
    set_lrs(epoch)
    model.train()
    total_loss = 0.0

    b_lr = optimizer.param_groups[0]["lr"]
    h_lr = optimizer.param_groups[1]["lr"]
    print(f"\nEpoch {epoch+1}/{num_epochs} | LR backbone={b_lr:.2e} | LR head={h_lr:.2e}")

    for images, masks in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()

        # grad clipping (helps stability with class weights + crops)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach().item()

    avg_loss = total_loss / max(1, len(train_loader))
    final_train_loss = avg_loss

    val_stats = evaluate(model, val_loader, num_classes, ignore_index, device)
    val_miou_all_epoch = float(val_stats["mIoU"])
    val_miou_supported_epoch = miou_over_subset(val_stats["IoU_per_class"], supported_classes)

    print(
        f"Epoch {epoch+1}/{num_epochs} | train_loss={avg_loss:.6f} | "
        f"val_mIoU_all={val_miou_all_epoch:.4f} | val_mIoU_supported={val_miou_supported_epoch:.4f}"
    )

    epoch_history.append({
    "run_id": run_id,
    "timestamp_utc": timestamp_utc,
    "epoch": epoch + 1,
    "train_loss": float(avg_loss),
    "val_mIoU_all": float(val_miou_all_epoch),
    "val_mIoU_supported": float(val_miou_supported_epoch) if not np.isnan(val_miou_supported_epoch) else "",
    "selection_metric_name": selection_metric_name,
    "selection_metric_value": float(
        val_miou_supported_epoch if selection_metric_name == "mIoU_supported" else val_miou_all_epoch
    ) if not np.isnan(
        val_miou_supported_epoch if selection_metric_name == "mIoU_supported" else val_miou_all_epoch
    ) else "",
    "lr_backbone": float(b_lr),
    "lr_head": float(h_lr),
    "is_best_epoch": 0,   # updated below if this epoch becomes best
})


    torch.save(model.state_dict(), ckpt_last_path)

    metric = val_miou_supported_epoch if selection_metric_name == "mIoU_supported" else val_miou_all_epoch

    if metric > best_metric + min_delta:
        best_metric = metric
        best_epoch = epoch + 1
        best_iou_per_class = val_stats["IoU_per_class"].copy()
        best_confusion = val_stats["confusion"].copy()
        best_miou_all = val_miou_all_epoch
        best_miou_supported = val_miou_supported_epoch
        no_improve = 0
        torch.save(model.state_dict(), ckpt_best_path)
        print(f"  New best {selection_metric_name}: {best_metric:.4f} (epoch {best_epoch})")
        epoch_history[-1]["is_best_epoch"] = 1
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping: no improvement ≥ {min_delta} for {patience} epochs.")
            break

print(f"\nBest {selection_metric_name}: {best_metric:.4f} at epoch {best_epoch}")


# ======================
# After training: update JSON + append to Excel
# ======================
config["results"] = {
    "best_metric_name": selection_metric_name,
    "best_metric_value": float(best_metric),
    "best_epoch": int(best_epoch),
    "best_mIoU_all": float(best_miou_all) if best_miou_all is not None else None,
    "best_mIoU_supported": float(best_miou_supported) if best_miou_supported is not None else None,
    "supported_classes": supported_classes,
    "min_val_images_per_class": int(min_val_images_per_class),
    "final_train_loss": float(final_train_loss) if final_train_loss is not None else None,
    "checkpoint_best_path": ckpt_best_path,
    "checkpoint_last_path": ckpt_last_path,
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Updated config with results: {config_path}")

row_dict = {
    "run_id": run_id,
    "timestamp_utc": timestamp_utc,
    "experiment_name": config["experiment_name"],
    "dataset_index_csv": csv_file,
    "seed": "",
    "model_arch": "deeplabv3_resnet101",
    "backbone": "resnet101",
    "pretrained_weights": "DEFAULT",
    "num_classes": num_classes,
    "crop_h": CROP_H,
    "crop_w": CROP_W,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "optimizer": "AdamW",
    "weight_decay": weight_decay,
    "lr_backbone": lr_backbone,
    "lr_head": lr_head,
    "warmup_epochs": warmup_epochs,
    "min_lr_backbone": min_lr_backbone,
    "min_lr_head": min_lr_head,
    "scheduler": "cosine_warmup",
    "loss_fn": "CrossEntropyLoss",
    "label_smoothing": label_smoothing if use_label_smoothing else 0.0,
    "class_weights": str(class_weights.detach().cpu().tolist()),
    "amp": True,
    "train_aug": "PadIfNeeded+RandomCrop+Flip+Affine+BrightnessContrast+Normalize",
    "val_aug": "FullResolution+Normalize",
    "best_metric_name": selection_metric_name,
    "best_metric_value": float(best_metric),
    "best_epoch": int(best_epoch),
    "best_mIoU_all": float(best_miou_all) if best_miou_all is not None else "",
    "best_mIoU_supported": float(best_miou_supported) if best_miou_supported is not None else "",
    "min_val_images_per_class": int(min_val_images_per_class),
    "supported_classes": str(supported_classes),
    "final_train_loss": float(final_train_loss) if final_train_loss is not None else "",
    "checkpoint_best_path": ckpt_best_path,
    "checkpoint_last_path": ckpt_last_path,
    "notes": "",
}

if best_iou_per_class is not None:
    for cid in range(num_classes):
        v = best_iou_per_class[cid]
        row_dict[f"IoU_class_{cid:02d}"] = (float(v) if not np.isnan(v) else "")
else:
    for cid in range(num_classes):
        row_dict[f"IoU_class_{cid:02d}"] = ""

required_headers = list(row_dict.keys())
ensure_excel_columns(log_xlsx_path, "Runs", required_headers)

append_run_to_excel(log_xlsx_path, row_dict, sheet_name="Runs")
# ======================
# Per-class confusion stats -> new Excel sheet
# ======================
if best_confusion is not None:
    per_class_rows = per_class_confusion_rows(
        confusion=best_confusion,
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        supported_classes=supported_classes,
        ignore_index=ignore_index,
    )

    per_class_headers = [
        "run_id",
        "timestamp_utc",
        "class_id",
        "is_background",
        "is_supported_class",
        "support_pixels",
        "predicted_pixels",
        "tp",
        "fp",
        "fn",
        "tn",
        "iou",
        "precision",
        "recall",
        "f1",
    ]

    ensure_sheet_with_headers(log_xlsx_path, "PerClassConfusion", per_class_headers)
    append_rows_to_excel(log_xlsx_path, per_class_rows, sheet_name="PerClassConfusion")
    print(f"Appended per-class confusion stats to Excel sheet 'PerClassConfusion': {log_xlsx_path}")
else:
    print("No best confusion matrix available; skipped PerClassConfusion logging.")
print(f"Appended run to Excel: {log_xlsx_path}")

# ======================
# Per-epoch training history -> new Excel sheet
# ======================
if epoch_history:
    epoch_df = pd.DataFrame(epoch_history)
    write_dataframe_to_sheet(log_xlsx_path, epoch_df, "EpochMetrics")
    print(f"Wrote per-epoch metrics sheet: {log_xlsx_path} [EpochMetrics]")
else:
    print("No epoch history available; skipped EpochMetrics export.")

#====================
# Full confusion matrix sheets
# ======================
if best_confusion is not None:
    class_names = [


"background",                  # 0
"sky",                         # 1
"snow_ice",                    # 2
"bare",                        # 3
"ericaceous_shrub",
"briar_seedling",            # 5
"graminoid",                   # 6
"forb",                        # 7
"fern",                        # 8
"cryptogam",                   # 9
"vine",         # 10
"broadleaf_evergreen",         # 10
"conifer",                     # 12
"broadleaf_deciduous",         # 13


]


    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names length ({len(class_names)}) does not match num_classes ({num_classes})"
    )

    conf_counts_df = confusion_to_count_dataframe(
        confusion=best_confusion,
        run_id=run_id,
        class_names=class_names,
    )

    conf_percent_df = confusion_to_percent_dataframe(
        confusion=best_confusion,
        run_id=run_id,
        class_names=class_names,
    )

    conf_col_percent_df = confusion_to_column_percent_dataframe(
        confusion=best_confusion,
        run_id=run_id,
        class_names=class_names,
    )

    write_dataframe_to_sheet(log_xlsx_path, conf_counts_df, "ConfusionMatrix_Counts")
    write_dataframe_to_sheet(log_xlsx_path, conf_percent_df, "ConfusionMatrix_RowPercent")
    write_dataframe_to_sheet(log_xlsx_path, conf_col_percent_df, "ConfusionMatrix_ColPercent")

    print(f"Wrote confusion matrix count sheet: {log_xlsx_path} [ConfusionMatrix_Counts]")
    print(f"Wrote confusion matrix row-percent sheet: {log_xlsx_path} [ConfusionMatrix_RowPercent]")
    print(f"Wrote confusion matrix column-percent sheet: {log_xlsx_path} [ConfusionMatrix_ColPercent]")
else:
    print("No best confusion matrix available; skipped confusion matrix sheet export.")

