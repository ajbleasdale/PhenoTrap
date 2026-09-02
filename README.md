# Workflow

## 1. Pre-processing 
### 1_preprocessing
The following steps outline the preprocessing workflow for camera trap imagery used in the PhenoTrap package. This pipeline selects timelapse images, standardises filenames, and prepares images for downstream annotation, training and inference.

#### 1-1_timelapse-selection.py
#### 1-2_file-rename.py

Renaming is performed before cropping to:
allow cross-referencing of filenames with camera records (date and time)
avoid potential issues caused by loss or modification of image metadata during processing

#### 1-3_header-footer-removal.py
#### 1-4_quality-control.py

### further information
#### 1-A_naming-convention.md

## 2. Dataset Labelling 
### 2_annotation
#### 2-1_serve-images.py (optional for HITL labelling)
#### 2-2_label-studio-interface (for implementation into Label Studio software)
#### 2-3_JSON-combiner.py
#### 2-4_JSON-reclass.py




## 3. Segmentation Model Training 
### 3_training

## 4. Segmentation Inference 
### 4_inference


## 5. Model Applications 
### 5_application


## 6. Model Applications Graphical Visualisations
### 6_visualisation

## 7. Miscellaneous Helper Code
### 7_miscellaneous
