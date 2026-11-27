
import os
import pandas as pd

# ============================
# CONFIG — EDIT THESE
# ============================
INPUT_CSV = r"path-to-input-csv.csv"
OUTPUT_DIR = r"Dpath-to-image-directory"

# ============================
# SCRIPT
# ============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

if "folder_name" not in df.columns:
    raise ValueError("CSV must contain a 'folder_name' column.")

# Get unique folder names
folders = df["folder_name"].unique()

for folder in folders:
    # Clean folder name for file safety
    safe_name = str(folder).replace("/", "_").replace("\\", "_").replace(":", "_")

    subset = df[df["folder_name"] == folder]

    out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")
    subset.to_csv(out_path, index=False)

    print(f"Wrote {len(subset)} rows to {out_path}")

print("Done.")
