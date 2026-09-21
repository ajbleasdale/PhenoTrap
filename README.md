# README

## 1. Pre-processing 
The following steps outline the preprocessing workflow for camera trap imagery used in the PhenoTrap package. This pipeline selects timelapse images, standardises filenames, and prepares images for downstream annotation, training and inference.

Renaming is performed before cropping to:
allow cross-referencing of filenames with camera records (date and time)
avoid potential issues caused by loss or modification of image metadata during processing

## 2. Dataset Labelling 

## 3. Segmentation Model Training 

Several of these scripts are optional depending on the labelling procedure
2-1_serve-images.py - optional for HITL labelling or moving dataset to a new PC
2-2_label-studio-interface for implementation into Label Studio software - Not to run in Python
2-3_JSON-combiner.py - run if using multiple smaller label-studio projects
2-4_JSON-reclass.py - use for renaming classes or combining multiple classes into a single class


## 4. Segmentation Inference 

## 5. Model Applications 

## 6. Model Applications Graphical Visualisations

## 7. Miscellaneous Helper Code


## Workflow
Run scripts in the following order
### 1_preprocessing
##### 1-1_timelapse-selection.py
##### 1-2_file-rename.py
##### 1-3_header-footer-removal.py
##### 1-4_quality-control.py
##### 1-5a_image-selection-random.py
##### 1-5b_image-selection-exclusionary.py (for use if certain images are to be avoided)


### 2_annotation
##### 2-1_serve-images.py (optional for HITL labelling)
##### 2-2_label-studio-interface for implementation into Label Studio software - Not to run in Python 
##### 2-3_JSON-combiner.py
##### 2-4_JSON-reclass.py
##### 2-5_JSON-to-PNG-mask.py
##### 2-6_dataset-partition.py
##### 2-7_HITL-mask-to-JSON.py




### 3_training


### 4_inference


### 5_application



### 6_visualisation


### 7_miscellaneous

## further information
#### 1-A_naming-convention.md
