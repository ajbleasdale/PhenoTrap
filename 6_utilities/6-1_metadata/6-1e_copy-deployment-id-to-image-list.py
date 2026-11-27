
import pandas as pd

# --------------------------------------------------
# File paths
# --------------------------------------------------
source_file = r"path-to-deployment-metadata.csv"
target_file = r"path-to-image-list-metadata.csv"
output_file = r"path-to-output-csv.csv"

# --------------------------------------------------
# Settings
# --------------------------------------------------
match_column = "deployment_id"

columns_to_copy = [
    "latitude",
    "longitude",
]

# --------------------------------------------------
# Read CSV files
# --------------------------------------------------
source_df = pd.read_csv(source_file)
target_df = pd.read_csv(target_file)

# Remove accidental whitespace from column names
source_df.columns = source_df.columns.str.strip()
target_df.columns = target_df.columns.str.strip()

# --------------------------------------------------
# Use deployment_id as the lookup key
# --------------------------------------------------
source_lookup = source_df.set_index(match_column)[columns_to_copy]

# Copy values where deployment_id matches
for column in columns_to_copy:
    target_df[column] = target_df[match_column].map(source_lookup[column])

# --------------------------------------------------
# Save result
# --------------------------------------------------
target_df.to_csv(output_file, index=False)

print(f"Updated file saved to:\n{output_file}")
