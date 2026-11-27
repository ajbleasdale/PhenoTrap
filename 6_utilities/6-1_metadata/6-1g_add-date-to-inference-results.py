import pandas as pd
import re

# ======================
# Input / Output
# ======================
input_csv = r"path-to-pixel-summary.csv"
output_csv = r"path-to-output-csv.csv"

# ======================
# Load data
# ======================
df = pd.read_csv(input_csv)

# ======================
# Extract date from filename
# ======================
df["date"] = df["filename"].str.extract(r'(\d{4}-\d{2}-\d{2})')

# Optional: convert to proper datetime format
df["date"] = pd.to_datetime(df["date"])

# ======================
# Save
# ======================
df.to_csv(output_csv, index=False)

print(f"Saved: {output_csv}")
