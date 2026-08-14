import os
import pandas as pd

# ======================
# CONFIG — EDIT THESE
CSV_FILE = r"root-to-csv-file"
ROOT_DIR = r"root-to-image-directory" # Same folder that contains the image subfolders


# ======================
# LOAD CSV
# ======================
df = pd.read_csv(CSV_FILE)

required = {"filename", "new_id"} 
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV missing columns: {missing}")

# ======================
# INDEX ALL FILES ON DISK
# ======================
file_locations = {}  # maps filename → full path

for dirpath, _, filenames in os.walk(ROOT_DIR):
    for f in filenames:
        file_locations[f] = os.path.join(dirpath, f)

# ======================
# RENAME
# ======================
for _, row in df.iterrows():
    old = str(row["filename"]).strip()
    new = str(row["new_id"]).strip()

    if old not in file_locations:
        print(f"NOT FOUND: {old}")
        continue

    old_path = file_locations[old]
    new_path = os.path.join(os.path.dirname(old_path), new)

    try:
        os.rename(old_path, new_path)
        print(f"Renamed: {old} → {new}")
    except Exception as e:
        print(f"ERROR renaming {old_path}: {e}")

print("Done.")
