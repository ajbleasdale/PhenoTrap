import os
import torch
import pandas as pd
import numpy as np

from PIL import Image, ImageFile
from tqdm import tqdm
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ======================
# Paths
# ======================
images_folder = r"path-to-image-directory"
model_path = r"path-to-CNN-model.pth"
output_csv = r"path-to-output-csv.csv"

# ======================
# Model settings
# ======================
num_classes = 14  # change if needed

class_names = {
    0: "background",
    1: "sky",
    2: "snow_ice",
    3: "bare",
    4: "ericaceous_shrub",
    5: "briar_seedling",
    6: "graminoid",
    7: "forb",
    8: "fern",
    9: "cryptogam",
    10: "vine",
    11: "broadleaf_evergreen",
    12: "conifer",
    13: "broadleaf_deciduous",

}

ignore_class_id = 14  # set to None if you do not want to exclude ignore from percentages

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================
# Image transform
# ======================
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # must match training size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# Load model
# ======================
model = deeplabv3_resnet101(
    weights=None,
    weights_backbone=None,
    num_classes=14,
    aux_loss=False
)

checkpoint = torch.load(model_path, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

# Remove auxiliary classifier weights because they are not needed for inference
checkpoint = {
    k: v for k, v in checkpoint.items()
    if not k.startswith("aux_classifier")
}

model.load_state_dict(checkpoint, strict=False)

model = model.to(device)
model.eval()

# ======================
# Valid image extensions
# ======================
valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

with torch.no_grad():
    for filename in tqdm(image_files, desc="Running inference"):

        # ======================
        # Extract date from filename
        # ======================
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)

        if date_match:
            date = date_match.group(1)
        else:
            date = None

        image_path = os.path.join(images_folder, filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        original_width, original_height = image.size

        input_tensor = transform(image).unsqueeze(0).to(device)

        output = model(input_tensor)["out"]
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        total_pixels = pred_mask.size

        if ignore_class_id is not None:
            valid_mask = pred_mask != ignore_class_id
            valid_pixels = int(valid_mask.sum())
        else:
            valid_pixels = total_pixels

        row = {
            "filename": filename,
            "date": date,
            "original_width": original_width,
            "original_height": original_height,
            "inference_width": pred_mask.shape[1],
            "inference_height": pred_mask.shape[0],
            "total_pixels": int(total_pixels),
            "valid_pixels_excluding_ignore": int(valid_pixels)
        }

        for class_id in range(num_classes):
            class_name = class_names.get(class_id, f"class_{class_id}")

            pixel_count = int((pred_mask == class_id).sum())

            if ignore_class_id is not None and class_id != ignore_class_id:
                percentage = (pixel_count / valid_pixels * 100) if valid_pixels > 0 else 0
            else:
                percentage = (pixel_count / total_pixels * 100) if total_pixels > 0 else 0

            row[f"{class_name}_pixels"] = pixel_count
            row[f"{class_name}_percent"] = percentage

        rows.append(row)

# ======================
# Save CSV
# ======================
results_df = pd.DataFrame(rows)
results_df.to_csv(output_csv, index=False)

print(f"\nSaved results to:")
print(output_csv)
