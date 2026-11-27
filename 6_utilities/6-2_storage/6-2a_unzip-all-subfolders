import os
import zipfile

def unzip_all(main_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(main_folder):
        if filename.lower().endswith(".zip"):
            zip_path = os.path.join(main_folder, filename)

            # Create a folder named after the ZIP file (without extension)
            extract_dir = os.path.join(output_folder, os.path.splitext(filename)[0])
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            print(f"Unzipped: {filename} -> {extract_dir}")

# Example usage:
unzip_all(r"path-to-zipped-directory", r"path-to-output-directory")
