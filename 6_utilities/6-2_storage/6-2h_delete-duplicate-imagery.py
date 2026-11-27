from pathlib import Path

# =========================
# SET YOUR FOLDERS HERE
# =========================
folder_a = Path(r"path-to-target-folder")  # Files to check/delete from
folder_b = Path(r"path-to-reference-folder")  # Reference folder

deleted = 0

# Get all filenames in Folder B (including subfolders)
reference_files = {f.name for f in folder_b.rglob("*") if f.is_file()}

# Check Folder A
for file in folder_a.rglob("*"):
    if file.is_file() and file.name in reference_files:
        file.unlink()
        deleted += 1

print(f"Deleted from Folder A: {deleted}")
