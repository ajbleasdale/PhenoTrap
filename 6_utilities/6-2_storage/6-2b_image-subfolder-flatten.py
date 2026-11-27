
import os
from pathlib import Path
import shutil

# Root folder containing subfolders and image files
root_folder = Path(r"path-to-zipped-folder")
output_folder = Path(r"path-to-output-directory")
output_folder.mkdir(parents=True, exist_ok=True)

# Allowed image extensions (lowercase)
image_exts = {".jpg", ".jpeg", ".png"}

# Find all image files recursively
image_files = [p for p in root_folder.rglob("*") if p.suffix.lower() in image_exts]

print(f"Found {len(image_files)} image files.")

for file in image_files:
    try:
        print(f"Processing: {file}")

        # Start with original filename
        dest_name = file.name
        dest_path = output_folder / dest_name

        # If a file with same name already exists, add a suffix
        if dest_path.exists():
            stem = file.stem
            suffix = file.suffix
            counter = 1
            while True:
                new_name = f"{stem}_{counter}{suffix}"
                dest_path = output_folder / new_name
                if not dest_path.exists():
                    break
                counter += 1

        # Copy the image into the flat folder
        shutil.copy2(file, dest_path)
        print(f"Copied to: {dest_path}")

    except Exception as e:
        print(f"Error processing {file}: {e}")

print("All image files processed. Flat copies saved in:", output_folder)
