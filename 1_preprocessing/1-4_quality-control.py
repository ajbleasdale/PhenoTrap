import os
import shutil
from pathlib import Path
from collections import Counter

import cv2
import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

ROOT_DIR = r"path-to-image-directory"

QC_FOLDER_NAME = "Quality_Control" #saves within  image-directory

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

MOVE_BAD_IMAGES = True
PRINT_RESULTS = True

# Main toggles
CHECK_BLUR = True
CHECK_BLOCKAGE = True
CHECK_NIR = True
CHECK_DARK_FAILED = True

# Blur settings
# Higher values = stricter blur removal
BLUR_THRESHOLD_FULL = 180.0
BLUR_THRESHOLD_TOP = 120.0

# Blockage / fog / condensation settings
BLOCKAGE_EDGE_RATIO_MAX = 0.018
BLOCKAGE_CONTRAST_MAX = 38.0
BLOCKAGE_BOTTOM_DARKNESS_MAX = 85.0
BLOCKAGE_BOTTOM_EDGE_RATIO_MAX = 0.010

# Dark failed image settings
DARK_MEAN_THRESHOLD = 30.0

# NIR settings
NIR_SATURATION_MEAN_MAX = 18.0
NIR_GRAYSCALE_CHANNEL_DIFF_MAX = 6.0
NIR_PINK_RATIO_THRESHOLD = 0.50


# ============================================================
# BASIC UTILITIES
# ============================================================

def is_image_file(path):
    return path.suffix.lower() in IMAGE_EXTENSIONS


def safe_destination(qc_dir, src_path):
    dest = qc_dir / src_path.name

    if not dest.exists():
        return dest

    stem = src_path.stem
    suffix = src_path.suffix

    i = 1
    while True:
        new_dest = qc_dir / f"{stem}__{i}{suffix}"
        if not new_dest.exists():
            return new_dest
        i += 1


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


# ============================================================
# QUALITY DETECTORS
# ============================================================

def laplacian_blur_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_ratio(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    return float(np.mean(edges > 0))


def detect_blur(image):
    """
    Detects general image blur.
    Uses Laplacian variance on full image and top section.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]

    full_score = laplacian_blur_score(gray)
    top = gray[: int(h * 0.45), :]
    top_score = laplacian_blur_score(top)

    is_blur = (
        full_score < BLUR_THRESHOLD_FULL
        and top_score < BLUR_THRESHOLD_TOP
    )

    return is_blur, {
        "blur_full": full_score,
        "blur_top": top_score,
    }


def detect_blockage(image):
    """
    Detects fogged lens, condensation, partial blockage, or low-information frames.

    This is designed to catch images like the one you showed:
    visible trees at top, but lower image dominated by low-texture blue/grey obstruction.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]

    full_std = float(np.std(gray))
    full_edges = edge_ratio(gray)

    bottom = gray[int(h * 0.45):, :]
    bottom_mean = float(np.mean(bottom))
    bottom_std = float(np.std(bottom))
    bottom_edges = edge_ratio(bottom)

    top = gray[: int(h * 0.35), :]
    top_edges = edge_ratio(top)

    # Full low-information frame
    full_blocked = (
        full_edges < BLOCKAGE_EDGE_RATIO_MAX
        and full_std < BLOCKAGE_CONTRAST_MAX
    )

    # Partial lower-lens blockage / condensation
    bottom_blocked = (
        bottom_edges < BLOCKAGE_BOTTOM_EDGE_RATIO_MAX
        and bottom_std < BLOCKAGE_CONTRAST_MAX
        and bottom_mean < BLOCKAGE_BOTTOM_DARKNESS_MAX
        and top_edges > bottom_edges * 1.8
    )

    is_blocked = full_blocked or bottom_blocked

    return is_blocked, {
        "full_std": full_std,
        "full_edges": full_edges,
        "bottom_mean": bottom_mean,
        "bottom_std": bottom_std,
        "bottom_edges": bottom_edges,
        "top_edges": top_edges,
    }


def detect_dark_failed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))

    return mean < DARK_MEAN_THRESHOLD, {
        "dark_mean": mean,
    }


def detect_nir(image):
    """
    Detects likely NIR / grayscale / false-colour images.
    """
    b, g, r = cv2.split(image)

    mean_channel_diff = float(
        np.mean(
            np.abs(r.astype(np.float32) - g.astype(np.float32))
            + np.abs(g.astype(np.float32) - b.astype(np.float32))
            + np.abs(r.astype(np.float32) - b.astype(np.float32))
        ) / 3.0
    )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_sat = float(np.mean(s))

    grayscale_like = (
        mean_channel_diff < NIR_GRAYSCALE_CHANNEL_DIFF_MAX
        and mean_sat < NIR_SATURATION_MEAN_MAX
    )

    pink_mask = (
        (((h >= 140) & (h <= 179)) | ((h >= 0) & (h <= 10)))
        & (s > 35)
        & (v > 40)
    )
    pink_ratio = float(np.mean(pink_mask))

    pink_nir = pink_ratio > NIR_PINK_RATIO_THRESHOLD

    is_nir = grayscale_like or pink_nir

    return is_nir, {
        "mean_saturation": mean_sat,
        "mean_channel_diff": mean_channel_diff,
        "pink_ratio": pink_ratio,
    }


# ============================================================
# MAIN QC
# ============================================================

def classify_image(image):
    reasons = []
    metrics = {}

    if CHECK_NIR:
        is_nir, m = detect_nir(image)
        metrics.update(m)
        if is_nir:
            reasons.append("NIR")

    if CHECK_DARK_FAILED:
        is_dark, m = detect_dark_failed(image)
        metrics.update(m)
        if is_dark:
            reasons.append("dark_failed")

    if CHECK_BLOCKAGE:
        is_blocked, m = detect_blockage(image)
        metrics.update(m)
        if is_blocked:
            reasons.append("blockage_or_obscured")

    if CHECK_BLUR:
        is_blur, m = detect_blur(image)
        metrics.update(m)
        if is_blur:
            reasons.append("blur")

    return reasons, metrics


def run_quality_control(root_dir):
    root_dir = Path(root_dir)
    qc_dir = root_dir / QC_FOLDER_NAME
    qc_dir.mkdir(exist_ok=True)

    counts = Counter()
    total_checked = 0
    total_moved = 0

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        if QC_FOLDER_NAME in path.parts:
            continue

        if not is_image_file(path):
            continue

        image = read_image(path)

        if image is None:
            counts["unreadable"] += 1
            continue

        total_checked += 1

        reasons, metrics = classify_image(image)

        if not reasons:
            if PRINT_RESULTS:
                print(f"[PASS] {path}")
            counts["pass"] += 1
            continue

        reason_text = "+".join(reasons)
        counts.update(reasons)
        counts["fail"] += 1

        if PRINT_RESULTS:
            print(f"[FAIL: {reason_text}] {path}")
            print("   ", {k: round(v, 3) for k, v in metrics.items()})

        if MOVE_BAD_IMAGES:
            dest = safe_destination(qc_dir, path)
            shutil.move(str(path), str(dest))
            total_moved += 1

    print("\n==============================")
    print("QUALITY CONTROL SUMMARY")
    print("==============================")
    print(f"Root folder: {root_dir}")
    print(f"Checked images: {total_checked}")
    print(f"Moved images: {total_moved}")
    print("\nCounts:")
    for key, value in counts.most_common():
        print(f"{key:>22}: {value}")


if __name__ == "__main__":
    run_quality_control(ROOT_DIR)
