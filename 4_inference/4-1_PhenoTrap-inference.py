import csv
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.amp import autocast
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101


# ==========================
# CONFIG
# ==========================
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_PATH = (
    r"path-to-CNN-model.pth"
)

input_dir = (
    r"path-to-input-directory"
)

output_dir = (
    r"path-to-output-directory"
)


# --------------------------
# Output options
# --------------------------
EXPORT_MASKS = True

# Toggle this to control overlay export.
EXPORT_OVERLAYS = True

MASKS_SUBFOLDER = "masks"
OVERLAYS_SUBFOLDER = "overlays"

# Overlay transparency:
# 0.0 = original image only
# 1.0 = segmentation colours only
OVERLAY_ALPHA = 0.45

# Keep the original image visible for predicted background pixels.
TRANSPARENT_BACKGROUND = True

EXPORT_TASKS_CSV = True
TASKS_CSV_NAME = "index.csv"

SUBFOLDER_VALUE = os.path.basename(input_dir.rstrip("/\\"))
TASKS_CSV_PATH = os.path.join(output_dir, TASKS_CSV_NAME)


# torchvision Resize tuple uses (height, width).
INPUT_HEIGHT = 1076
INPUT_WIDTH = 1536
input_size = (INPUT_HEIGHT, INPUT_WIDTH)


# ======================
# CLASS MAP
# ======================
class_map = {
    "background": 0,  
    "sky": 1,
    "snow_ice": 2,
    "bare": 3,
    "ericaceous_shrub": 4,
    "briar_seedling": 5,
    "graminoid": 6,
    "forb": 7,
    "fern": 8,
    "cryptogam": 9,
    "vine": 10,
    "broadleaf_evergreen": 11,
    "conifer": 12,
    "broadleaf_deciduous": 13,
    "ignore": 255,
}

RAW_IGNORE_VALUE = 255
num_classes = 14
ignore_index = RAW_IGNORE_VALUE


# ======================
# CLASS COLOURS
# ======================
id2color = {
    0: (0, 0, 0),  # black
    1: (86, 180, 233), # light blue
    2: (255, 255, 255), # white
    3: (128, 128, 128), # grey
    4: (230, 159, 0), # orange
    5: (204, 121, 167), # pink
    6: (117, 112, 179), # purple
    7: (230, 80, 0), # red
    8: (240, 228, 66), # yellow
    9: (0, 114, 190), # blue
    10: (231, 41, 138), # magenta
    11: (166, 86, 40), # brown
    12: (27, 158, 119), # teal
    13: (102, 166, 30), # green
}


# ==========================
# UTILS
# ==========================
def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_mask(
    mask_np: np.ndarray,
    out_mask_path: str,
) -> None:
    """Save a class-index mask as an 8-bit PNG."""
    ensure_dir(os.path.dirname(out_mask_path))

    Image.fromarray(
        mask_np.astype(np.uint8),
        mode="L",
    ).save(out_mask_path)


def mask_to_rgb(mask_np: np.ndarray) -> np.ndarray:
    """Convert a class-index mask into an RGB colour image."""
    height, width = mask_np.shape

    color_mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    for class_id, color in id2color.items():
        color_mask[mask_np == class_id] = color

    return color_mask


def create_overlay(
    original: Image.Image,
    mask_np: np.ndarray,
    alpha: float = 0.45,
    transparent_background: bool = True,
) -> Image.Image:
    """
    Blend the colour-coded prediction mask with the original image.

    When transparent_background=True, class 0 pixels are left unchanged.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"OVERLAY_ALPHA must be between 0 and 1, received {alpha}"
        )

    original_np = np.asarray(
        original.convert("RGB"),
        dtype=np.uint8,
    )

    if mask_np.shape != original_np.shape[:2]:
        raise ValueError(
            "Mask and image dimensions do not match: "
            f"mask={mask_np.shape}, image={original_np.shape[:2]}"
        )

    color_mask = mask_to_rgb(mask_np)

    blended = (
        (1.0 - alpha) * original_np.astype(np.float32)
        + alpha * color_mask.astype(np.float32)
    )

    blended = np.clip(blended, 0, 255).astype(np.uint8)

    if transparent_background:
        background_pixels = mask_np == class_map["background"]
        blended[background_pixels] = original_np[background_pixels]

    return Image.fromarray(blended, mode="RGB")


def save_overlay(
    original: Image.Image,
    mask_np: np.ndarray,
    out_overlay_path: str,
    alpha: float,
    transparent_background: bool,
) -> None:
    """Create and save a segmentation overlay."""
    ensure_dir(os.path.dirname(out_overlay_path))

    overlay = create_overlay(
        original=original,
        mask_np=mask_np,
        alpha=alpha,
        transparent_background=transparent_background,
    )

    overlay.save(
        out_overlay_path,
        quality=95,
    )


# ==========================
# SETUP
# ==========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

ensure_dir(output_dir)

masks_dir = os.path.join(
    output_dir,
    MASKS_SUBFOLDER,
)

overlays_dir = os.path.join(
    output_dir,
    OVERLAYS_SUBFOLDER,
)

if EXPORT_MASKS:
    ensure_dir(masks_dir)

if EXPORT_OVERLAYS:
    ensure_dir(overlays_dir)


# ==========================
# MODEL
# ==========================
model = deeplabv3_resnet101(
    weights=None,
    aux_loss=False,
)

model.classifier[4] = nn.Conv2d(
    256,
    num_classes,
    kernel_size=1,
)

state = torch.load(
    MODEL_PATH,
    map_location=device,
)

if isinstance(state, dict) and "model" in state:
    state = state["model"]

load_result = model.load_state_dict(
    state,
    strict=False,
)

if load_result.missing_keys:
    print("Missing model keys:", load_result.missing_keys)

if load_result.unexpected_keys:
    print("Unexpected model keys:", load_result.unexpected_keys)

model.to(device)
model.eval()


# ==========================
# TRANSFORMS
# ==========================
transform = transforms.Compose([
    transforms.Resize(
        input_size,
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==========================
# RUN
# ==========================
tasks_rows = []

image_files = sorted([
    filename
    for filename in os.listdir(input_dir)
    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
])

print(f"Found {len(image_files)} images.")

for i, filename in enumerate(image_files, start=1):
    image_path = os.path.join(
        input_dir,
        filename,
    )

    try:
        with Image.open(image_path) as image:
            original = image.convert("RGB")

    except Exception as error:
        print(
            f"[{i}/{len(image_files)}] "
            f"Skipped {filename}: {error}"
        )
        continue

    # PIL size is (width, height).
    original_size = original.size

    # The transform performs the inference resize.
    input_tensor = (
        transform(original)
        .unsqueeze(0)
        .to(device)
    )

    # Predict.
    with torch.no_grad():
        with autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            logits = model(input_tensor)["out"]

        pred = torch.argmax(
            logits,
            dim=1,
        ).squeeze(0)

        pred_np = (
            pred.detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    # Resize prediction to the original image dimensions.
    mask_resized = Image.fromarray(
        pred_np,
        mode="L",
    ).resize(
        original_size,
        resample=Image.Resampling.NEAREST,
    )

    mask_np = np.asarray(
        mask_resized,
        dtype=np.uint8,
    )

    base = os.path.splitext(filename)[0]

    saved_outputs = []

    # Save class-index mask.
    if EXPORT_MASKS:
        out_mask_path = os.path.join(
            masks_dir,
            f"{base}.png",
        )

        save_mask(
            mask_np,
            out_mask_path,
        )

        saved_outputs.append("mask")

    # Save colour overlay.
    if EXPORT_OVERLAYS:
        out_overlay_path = os.path.join(
            overlays_dir,
            f"{base}_overlay.jpg",
        )

        save_overlay(
            original=original,
            mask_np=mask_np,
            out_overlay_path=out_overlay_path,
            alpha=OVERLAY_ALPHA,
            transparent_background=TRANSPARENT_BACKGROUND,
        )

        saved_outputs.append("overlay")

    output_description = ", ".join(saved_outputs)

    print(
        f"[{i}/{len(image_files)}] "
        f"Processed {filename}: {output_description}"
    )

    if EXPORT_TASKS_CSV:
        tasks_rows.append({
            "new_name": filename,
            "mask_name": f"{base}.png",
            "subfolder": SUBFOLDER_VALUE,
        })


# ==========================
# EXPORT TASKS CSV
# ==========================
if EXPORT_TASKS_CSV:
    ensure_dir(os.path.dirname(TASKS_CSV_PATH))

    with open(
        TASKS_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "new_name",
                "mask_name",
                "subfolder",
            ],
        )

        writer.writeheader()
        writer.writerows(tasks_rows)

    print(f"Wrote tasks CSV: {TASKS_CSV_PATH}")

print("All images processed.")
