# PhenoTrap

PhenoTrap is a workflow for semantic segmentation of fixed camera-trap imagery
for vegetation and phenological monitoring. The workflow covers image
pre-processing, semantic annotation, model training, inference, and extraction
of ecological metrics from classified imagery.

PhenoTrap fine-tunes a DeeplabV3/ResNet-101 CNN to isolate 13 unique environmental and vegetation classes. 
Inference from PhenoTrap can be utilised for monitoring habitat scene composition, snow cover analysis and vegetation greenup

## Installation

### Requirements
- Python
- PyTorch
- Torchvision
- OpenCV
- Pillow
- pandas
- NumPy


The expected input strucure is as follows
input/
├── site-1/ <br/> 
│  ├── BE-LE_40_2023-04-12_160000.JPG <br/> 
│  └── BE-LE_532_2024-08-13_160000.JPG
└── site-2/
   ├── SE-NM_111_2020_03_24_120000.JPG
   └── SE-NM_43_2021_12_24_120000.JPG


## 1. Pre-processing 
The following steps outline the preprocessing workflow for camera trap imagery used in the PhenoTrap package. This pipeline selects timelapse images, standardises filenames, and prepares images for downstream annotation, training and inference.

Renaming is performed before cropping to:
allow cross-referencing of filenames with camera records (date and time)
avoid potential issues caused by loss or modification of image metadata during processing

## 2. Dataset Labelling 

Several of these scripts are optional depending on the labelling procedure
2-1_serve-images.py - optional for HITL labelling or moving dataset to a new PC
2-2_label-studio-interface for implementation into Label Studio software - Not to run in Python
2-3_JSON-combiner.py - run if using multiple smaller label-studio projects
2-4_JSON-reclass.py - use for renaming classes or combining multiple classes into a single class

## 3. Segmentation Model Training 
3-1_PhenoTrap training - used to fine-tune model
3-2_experiment-log-analysis - optional
experiment-log.xlsx IMPORTANT - download and save in experiment folder to ensure results or logged correctly


## 4. Segmentation Inference 

scripts to run inference, organise data and subsequent ROI isolation 

4-1_PhenoTrap-inferance.py
4-2_class-ROI-isolation.py
4-3_organise-ROI-to-class-folder.py
4-4a_quanitfy-class-from-model.py
4-4b_quanitfy-class-from-mask.py
4-5_reverse-ROI-to-mask.py (optional - emergency retrieval of single-channel masks)


## 5. Model Applications 

scripts for for further ecological applications either direct from inference data or from subsequent ROI isolation

5-1a_dataset-scene-composition.py
5-1b_deployment-scene-composition.py
5-2a_snow-cover.py
5-2b_snow-cover-comparison.py
5-3a_GCC-SS-SC-annual.py
5-3b_GCC-SS-MC-annual.py
5-3c_GCC-MS-SC-annual.py
5-3d_GCC-MS-MC-annual.py
5-3e_GCC-MS-MC-continuous.py
5-4a_GCC-breakpoints-threshold.py
5-4b_GCC-breakpoints-slope.py

## 6. PhenoTrap Utlities
Split into folders, code to help

metadata to add and save update metadata
6-2_storage change information folder structure etc
6-3_checks - to do any checks
6-4_visualisation - scripts to generate ancillary graphs and figures



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
##### 2-2_label-studio-interface (for implementation into Label Studio software - Not to run in Python)
##### 2-3_JSON-combiner.py
##### 2-4_JSON-reclass.py
##### 2-5_JSON-to-PNG-mask.py
##### 2-6_dataset-partition.py
##### 2-7_HITL-mask-to-JSON.py




### 3_training
##### 3-1_PhenoTrap-training.py
##### 3-2_experiment-log-analysis.py
##### experiment-log.xlsx (to download for saving experimental log data)


### 4_inference
##### 4-1_PhenoTrap-inferance.py
##### 4-2_class-ROI-isolation.py
##### 4-3_organise-ROI-to-class-folder.py
##### 4-4a_quanitfy-class-from-model.py
##### 4-4b_quanitfy-class-from-mask.py
##### 4-5_reverse-ROI-to-mask.py (emergency retrieval of single-channel masks)

### 5_application
#### 5-1a_dataset-scene-composition.py
#### 5-1b_deployment-scene-composition.py
#### 5-2a_snow-cover.py
#### 5-2b_snow-cover-comparison.py
#### 5-3a_GCC-SS-SC-annual.py
#### 5-3b_GCC-SS-MC-annual.py
#### 5-3c_GCC-MS-SC-annual.py
#### 5-3d_GCC-MS-MC-annual.py
#### 5-3e_GCC-MS-MC-continuous.py
#### 5-4a_GCC-breakpoints-threshold.py
#### 5-4b_GCC-breakpoints-slope.py




### 6_utilities




## Documentation
#### 1_naming-convention.md
#### 2_quality control.md
