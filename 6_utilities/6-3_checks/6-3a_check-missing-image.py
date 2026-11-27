import os
import pandas as pd

# =========================
# PATHS
# =========================
folder_path = r"path-to-image-folder"
csv_path = r"path-to-output-csv.csv"

# =========================
# SETTINGS
# =========================
csv_filename_column = "image_name"   # change if needed
valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp",".png"}

# =========================
# READ CSV
# =========================
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

csv_names = set(
    df[csv_filename_column]
    .astype(str)
    .str.strip()
    .str.lower()
)

# =========================
# READ FOLDER
# =========================
folder_names = set()

for fname in os.listdir(folder_path):
    full_path = os.path.join(folder_path, fname)
    if os.path.isfile(full_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            folder_names.add(fname.strip().lower())

# =========================
# COMPARE
# =========================
in_folder_not_csv = sorted(folder_names - csv_names)
in_csv_not_folder = sorted(csv_names - folder_names)

# =========================
# RESULTS
# =========================
print(f"Images in folder: {len(folder_names)}")
print(f"Filenames in CSV: {len(csv_names)}")
print()

print(f"In folder but not in CSV: {len(in_folder_not_csv)}")
for name in in_folder_not_csv[:20]:
    print("  ", name)

if len(in_folder_not_csv) > 20:
    print("  ...")

print()
print(f"In CSV but not in folder: {len(in_csv_not_folder)}")
for name in in_csv_not_folder[:20]:
    print("  ", name)

if len(in_csv_not_folder) > 20:
    print("  ...")
