import os
import shutil

# ==========================
# CONFIG
# ==========================
SRC_DIR = r"path-to-source-directory"
DST_DIR = r"path-to-output-directory"
SUFFIX = "_120000.jpg"   # example: ".JPG", "_120000.JPG"

# ==========================
# SCRIPT
# ==========================
os.makedirs(DST_DIR, exist_ok=True)

copied = 0

for fname in os.listdir(SRC_DIR):
    if fname.endswith(SUFFIX):
        src_path = os.path.join(SRC_DIR, fname)
        dst_path = os.path.join(DST_DIR, fname)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1

print(f"[INFO] Copied {copied} files to {DST_DIR}")
