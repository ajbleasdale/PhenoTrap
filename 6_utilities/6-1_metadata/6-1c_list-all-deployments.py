
import pandas as pd
from pathlib import Path

# ==========================
# SETTINGS
# ==========================
INPUT_FOLDER = r"path-to-metadata-folder-xlsx"  # folder containing .xlsx files
OUTPUT_FOLDER = r"path-to-output.folder"        # separate output folder
OUTPUT_NAME = "output-file-name-xlsx"
COLUMN_NAME = "deployment_id"      # column to collect

# ==========================
# SETUP
# ==========================
in_path = Path(INPUT_FOLDER)
out_path = Path(OUTPUT_FOLDER)
out_path.mkdir(parents=True, exist_ok=True)

xlsx_files = sorted(in_path.glob("*.xlsx"))

unique_ids = set()
skipped = []

# ==========================
# COLLECT UNIQUE IDS
# ==========================
for f in xlsx_files:
    try:
        df = pd.read_excel(f)
        if COLUMN_NAME not in df.columns:
            skipped.append((f.name, "missing deployment_id column"))
            continue

        vals = df[COLUMN_NAME].dropna().astype(str).str.strip()
        unique_ids.update(v for v in vals if v)

    except Exception as e:
        skipped.append((f.name, str(e)))

# ==========================
# SAVE OUTPUT
# ==========================
out_df = pd.DataFrame({COLUMN_NAME: sorted(unique_ids)})
out_file = out_path / OUTPUT_NAME
out_df.to_excel(out_file, index=False)

# Optional: write a skip log
if skipped:
    log_df = pd.DataFrame(skipped, columns=["file", "reason"])
    log_df.to_csv(out_path / "skipped_files.csv", index=False)

print(f"Found {len(unique_ids)} unique deployment_id values.")
print(f"Saved: {out_file}")
if skipped:
    print(f"Some files were skipped. See: {out_path / 'skipped_files.csv'}")
