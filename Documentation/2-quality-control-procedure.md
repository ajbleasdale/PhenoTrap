# Image Quality Control

## Overview

The `1-4_quality-control.py` script performs automated quality control of camera-trap imagery before image selection, annotation, model training, or inference. Images that meet one or more predefined rejection criteria are identified and moved to a `Quality_Control` folder, while images that pass quality control remain in their original location.

The procedure is designed to identify images that are unsuitable for subsequent image analysis while retaining valid environmental conditions such as snow cover.

## Quality-control criteria

Images are assessed for four types of quality issue:

### 1. Image blur

Blur is assessed using the variance of the Laplacian, which provides a measure of image sharpness based on spatial intensity changes. Sharpness is evaluated for both the complete image and the upper 45% of the image.

An image is classified as blurred when both:

* full-image Laplacian variance < `180`
* upper-image Laplacian variance < `120`

Using both measurements reduces the likelihood of rejecting an image based on a naturally low-texture region in only part of the scene.

### 2. Lens blockage, fogging, and condensation

Potentially obscured images are identified using image contrast and the proportion of pixels detected as edges.

Two forms of obstruction are assessed:

* **Full-image obstruction** – images with both low edge density and low overall contrast.
* **Partial lower-image obstruction** – images where the lower portion has low edge density, low contrast and low brightness relative to a more structurally detailed upper portion of the image.

This is intended to identify images affected by conditions such as lens fogging, condensation, or physical obstruction.

### 3. Near-infrared and false-colour images

Images produced using camera near-infrared (NIR) modes are excluded because their colour information is unsuitable for subsequent colour-based analyses.

Two types of NIR imagery are detected:

* **Grayscale-like NIR images**, identified from low differences between RGB channels combined with low colour saturation.
* **False-colour NIR images**, identified from a high proportion of pink/red pixels.

The default thresholds are:

* mean channel difference < `6`
* mean saturation < `18`
* pink/red pixel proportion > `0.50`

### 4. Failed dark images

Very dark images are identified from their mean grayscale pixel intensity.

Images with a mean grayscale intensity below `30` are classified as failed dark images.

## Snow-covered images

Snow cover is **not** treated as a quality-control failure. Images containing snow are retained provided that they do not meet another rejection criterion. This allows snow presence and seasonal snow-cover dynamics to be analysed in subsequent stages of the PhenoTrap workflow.

## Running quality control

Before running the script, set the root directory containing the images:

```python
ROOT_DIR = r"path\to\camera-trap\images"
```

The script recursively searches the root directory and its subdirectories for supported image formats:

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, and `.tiff`.

Each image is evaluated against all enabled quality-control criteria. Images that fail one or more checks are moved to:

```text
ROOT_DIR/Quality_Control/
```

Images passing all checks remain in their original directories.

If duplicate filenames occur when failed images are moved, a numerical suffix is automatically added to prevent existing files from being overwritten.

## Configuration

Individual quality-control checks can be enabled or disabled using:

```python
CHECK_BLUR = True
CHECK_BLOCKAGE = True
CHECK_NIR = True
CHECK_DARK_FAILED = True
```

Setting `MOVE_BAD_IMAGES = False` allows the quality-control procedure to be run without moving rejected images.

Threshold values are defined in the `USER SETTINGS` section of the script and can be adjusted where required. The default thresholds were selected for the imagery used in the development of PhenoTrap and may require adjustment for other camera systems, environments, image resolutions, or acquisition settings.

## Output

During processing, each image is reported as either `PASS` or `FAIL`. Failed images are accompanied by the detected rejection reason(s) and the associated image-quality metrics.

After processing, the script reports:

* total number of images checked;
* total number of images moved;
* number of images passing quality control;
* number failing quality control; and
* counts for each individual rejection criterion.

An image can fail more than one criterion; therefore, the sum of individual rejection categories may exceed the total number of failed images.

Unreadable image files are reported separately and are not automatically moved by the current procedure.
