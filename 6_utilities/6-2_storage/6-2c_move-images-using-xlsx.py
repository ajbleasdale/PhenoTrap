
from pathlib import Path
import shutil
import pandas as pd

# --- settings ---
xlsx_file = r"path-to-image-list.xlsx"
source_folder = Path(r"path-to-source-directory")
destination_folder = Path(r"path-to-new-directory")

# Column in Excel containing image filenames
filename_column = "image_name"   # change this if needed

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

destination_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(xlsx_file)

wanted = set()

for value in df[filename_column].dropna().astype(str):
    name = Path(value.strip()).name
    wanted.add(name.lower())
    wanted.add(Path(name).stem.lower())

source_images = [
    p for p in source_folder.iterdir()
    if p.is_file() and p.suffix.lower() in image_exts
]

moved = 0
missing = []

for img in source_images:
    if img.name.lower() in wanted or img.stem.lower() in wanted:
        shutil.move(str(img), str(destination_folder / img.name))
        moved += 1

print(f"Moved {moved} images.")

print("\nFirst 10 Excel values:")
print(df[filename_column].dropna().astype(str).head(10).to_list())

print("\nFirst 10 source image files:")
print([p.name for p in source_images[:10]])
