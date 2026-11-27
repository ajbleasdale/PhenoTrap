PhenoTrap
PhenoTrap is a semantic segmentation workflow for extracting environmental and vegetation information from fixed wildlife camera-trap imagery, including images collected incidentally during wildlife monitoring. It uses a fine-tuned DeepLabV3-ResNet101 convolutional neural network to segment 13 environmental and vegetation classes.
The model was developed using 1,400 annotated training images from Nordmaling, Sweden (SE-NM), and Leuven, Belgium (BE-LE). The workflow supports habitat composition and visible vegetation-class diversity analysis, snow-cover monitoring, and targeted vegetation phenology assessments using the Green Chromatic Coordinate (GCC).
By utilising imagery already collected for wildlife studies, PhenoTrap provides environmental information that is spatially and temporally linked to wildlife observations and can support subsequent habitat-use studies.
Associated manuscript: PhenoTrap: Semantic Segmentation of Environmental and Vegetation Classes in Camera-Trap Imagery (submitted; not yet published).
<!-- Optional: upload Figure 3 to assets/segmentation-example.png, then uncomment the following image and caption.
![Original imagery, reference annotations and segmentation predictions](assets/segmentation-example.png)
*Example original images, reference annotation masks and predicted segmentation masks from SE-NM_84 and BE-LE_1087 (Figure 3).*
Contents
Workflow overview
Installation
Quick start
Segmentation classes
Training dataset and model performance
Repository structure
1. Pre-processing
2. Dataset labelling
3. Segmentation model training
4. Segmentation inference
5. Model applications
6. PhenoTrap utilities
Model availability
Limitations
Documentation
Citation and licence
Acknowledgements
Workflow Overview
Two pathways are available:
Apply the pre-trained model: pre-processing → segmentation inference → ecological applications.
Develop or adapt a model: pre-processing → dataset labelling → model training and assessment → segmentation inference → ecological applications.
Human-in-the-loop (HITL) annotation is optional. Utilities support all stages.
```text
PhenoTrap/
├── 1_preprocessing/
├── 2_annotation/
├── 3_training/
├── 4_inference/
├── 5_applications/
│   ├── 5-1_habitat-composition/
│   ├── 5-2_snow-cover/
│   └── 5-3_greenness/
├── 6_utilities/
│   ├── 6-1_metadata/
│   ├── 6-2_storage/
│   ├── 6-3_checks/
│   ├── 6-4_visualisation/
│   └── 6-5_troubleshooting/
├── Documentation/
├── LICENSE
├── requirements.txt
└── README.md
```
Installation
Requirements
Python 3.12
The workflow has been used on Windows and the Alvis high-performance computing (HPC) cluster provided by NAISS.
PyTorch <br/>
Torchvision <br/>
NumPy <br/>
pandas <br/>
Pillow <br/>
OpenCV <br/>
Albumentations <br/>
tqdm <br/>
Matplotlib <br/>
SciPy <br/>
Plotly <br/>
OpenPyXL <br/>
Flask <br/>
Flask-CORS <br/>
Dependencies can be installed using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Model training requires HPC resources for the workflow as developed. A CUDA-compatible GPU is recommended for inference. CPU execution has not been verified here.

Label Studio
Label Studio is required for the annotation workflow, but not for applying the pre-trained model. See Label Studio for installation details.
Quick start
To apply PhenoTrap to a new dataset:
Obtain the pre-trained model weights (see Model availability).
Install the Python dependencies.
Prepare a working copy of the imagery using Section 1. Standardised filenames and removal of camera headers and footers are required.
Review the input, output and model checkpoint paths and settings in `4-1_PhenoTrap-inference.py` before execution.
Run segmentation inference and select the relevant class quantification or ROI extraction scripts in Section 4.
Select the ecological analysis scripts in Section 5.
Sections 2 and 3 are not required when applying the pre-trained model. No specific camera model is required.
> [!WARNING]
> Some pre-processing and utility scripts may permanently modify, move or delete imagery. Always retain an unmodified backup and run scripts on working copies. Check paths and settings before execution.
Segmentation classes
The 13 target classes are encoded using integer pixel values. Background and ignore are additional labels, not target environmental or vegetation classes.
Pixel value	Class
0	`background`
1	`sky`
2	`snow_ice`
3	`bare`
4	`ericaceous_shrub`
5	`briar_seedling`
6	`graminoid`
7	`forb`
8	`fern`
9	`cryptogam`
10	`vine`
11	`broadleaf_evergreen`
12	`conifer`
13	`broadleaf_deciduous`
255	`ignore`
Training dataset and model performance
The annotated dataset comprises 2,000 images from SE-NM and BE-LE. Dataset partitioning is performed at deployment level to reduce leakage between subsets.
Subset	Images
Training (including 800 HITL-annotated images)	1,400
Validation	200
Testing	400
Total	2,000
<!-- Confirm validation/test counts before publication. -->
The model achieved the following mean Intersection over Union (mIoU) and frequency-weighted Intersection over Union (FWIoU) scores, as reported in the submitted manuscript.
Dataset	mIoU	FWIoU
Overall	0.676	0.756
BE-LE	0.654	0.717
SE-NM	0.424	0.787
Environmental classes (`sky`, `snow_ice` and `bare`) achieved the highest IoU values. Abundant vegetation classes, including `ericaceous_shrub`, `conifer` and `broadleaf_deciduous`, generally performed well, particularly where they were more strongly represented. Less-abundant classes, including `vine` and `broadleaf_evergreen`, generally performed more poorly.
<!-- Optional: upload Figure S6 and uncomment.
![Test-set segmentation examples](assets/segmentation-test-examples.png)
*Original camera-trap images and predicted segmentation masks from the test set, with associated FWIoU and mIoU scores (Figure S6).*
Repository Structure
The expected input structure is as follows: <br/>
input/ <br/>
├── site-1/ <br/>
│  ├── BE-LE_40_2023-04-12_160000.JPG <br/>
│  └── BE-LE_532_2024-08-13_160000.JPG <br/>
└── site-2/ <br/>
├── SE-NM_111_2020_03_24_120000.JPG <br/>
└── SE-NM_43_2021_12_24_120000.JPG <br/>

1. Pre-processing
This stage prepares raw camera-trap imagery for annotation, model training and inference. The workflow selects timelapse images, standardises filenames, removes unwanted image regions and poor-quality images, and selects images for subsequent annotation.
Renaming and header/footer removal are compulsory. Other pre-processing steps depend on the dataset and intended analysis. Renaming is performed before image cropping to retain a consistent link between each image and the original camera records, including the date and time of capture. This also avoids potential issues associated with the loss or modification of image metadata during subsequent processing.
Select the relevant scripts in the order listed below:
`1-1_timelapse-selection.py` – Selects timelapse images from the raw camera-trap dataset while excluding images generated by other trigger modes.
`1-2_file-rename.py` – Renames images using a standardised naming convention containing deployment and image capture information. See Naming Convention for further details.
`1-3_remove-header-footer.py` – Removes camera-generated headers and/or footers from images to prevent overlays such as timestamps, camera information or logos from influencing subsequent annotation and model training.
`1-4_quality-control.py` – Identifies and removes poor-quality imagery, including blurred, obscured, dark and near-infrared images. See Quality control for further details.
`1-5a_image-selection-random.py` – Randomly selects images from the pre-processed dataset for subsequent annotation.
`1-5b_image-selection-random-exclusionary.py` – Optional alternative to `1-5a`. Selects images while preventing images from the same deployment entering the annotation sample.

2. Dataset labelling
This stage converts selected camera-trap images into semantic segmentation
training data. Images can be annotated manually in Label Studio or through
the optional human-in-the-loop (HITL) workflow. HITL scripts support iterative annotation and correction rather than a compulsory linear sequence.
Run the required scripts in the following order:
`2-1_serve-images.py` – Optional; serves images for HITL annotation or when
transferring the annotation dataset to another computer.
`2-2_label-studio-labelling-interface` – Label Studio interface configuration. This is
imported into Label Studio and is not run as a Python script. See Labelling strategy for further details.
`2-3_JSON-combiner.py` – Optional; combines annotations exported from
multiple Label Studio projects.
`2-4_JSON-reclass.py` – Optional; renames classes or merges multiple
annotation classes.
`2-5a_JSON-to-PNG-mask-complex.py` – Converts Label Studio JSON annotations to
segmentation masks.
`2-6_dataset-partition.py` – Partitions the labelled dataset at deployment level for model development.
`2-7_create-mask-image-index.py` – Creates an index linking images and masks.
`2-8_HITL-link-overlay-image-mask-index.py` – Links overlays, images and masks for HITL annotation.
`2-9_HITL-mask-to-JSON.py` – Converts masks to JSON annotations.

3. Segmentation Model Training
This stage fine-tunes the semantic segmentation model using the annotated and partitioned dataset produced during the previous stage. Model training is performed using DeepLabV3 with a ResNet-101 backbone implemented in PyTorch and Torchvision.
Run the required scripts in the following order:
`3-1_PhenoTrap-training.py` – Fine-tunes the segmentation model using the prepared training and validation datasets. Training progress, model performance and experimental parameters are recorded during training, and trained model checkpoints are saved for subsequent inference.
`3-2_experiment-log-analysis.py` – Optional. Summarises and visualises results recorded across training experiments to assist with comparison of model configurations and training performance.

`3-3_PhenoTrap-test-assessment.py` – Assesses model performance on the test dataset.
`experiment-log.xlsx` – Template used by the training script to record experimental settings and model performance. Download and save a copy of this file in the experiment directory before training to ensure that results are logged correctly.
The trained model checkpoint produced during this stage is used for segmentation inference in Section 4.

4. Segmentation Inference

This stage applies a trained PhenoTrap segmentation model to camera-trap imagery and processes the resulting predictions for subsequent analysis. The workflow generates semantic segmentation predictions, isolates class-specific regions of interest (ROIs), organises extracted imagery by class, and quantifies the predicted coverage of each class.
Images are resized for model input and predictions are resized back to the original image dimensions during inference. Class coverage can be calculated without saving ROI imagery; class-specific greenness requires ROI extraction.
Select the scripts required for the intended output:
`4-1_PhenoTrap-inference.py` – Applies a trained segmentation model to camera-trap images to generate pixel-level semantic segmentation predictions.
`4-2_class-ROI-isolation.py` – Uses the model predictions to isolate class-specific regions of interest (ROIs) from the original camera-trap images.
`4-3_organise-ROI-to-class-folder.py` – Organises the extracted ROI images into class-specific folders for subsequent analysis.
`4-4a_quantify-class-from-model.py` – Quantifies the number and proportion of pixels assigned to each class directly from the model inference outputs.
`4-4b_quantify-class-from-ROI-mask.py` – Alternative quantification workflow that calculates class coverage from previously generated segmentation masks.

The outputs generated during this stage can be used directly for scene-composition analysis or processed further using the ecological applications described in Section 5.

5. Model Applications
This stage contains ecological applications that use outputs from the PhenoTrap segmentation workflow. These scripts are not intended to be run sequentially; instead, individual scripts can be selected depending on the analysis required. Applications currently include scene-composition analysis, snow-cover estimation, and vegetation greenness and phenology analysis using the Green Chromatic Coordinate (GCC).
5.1 Habitat composition
Habitat composition is expressed relative to the entire image, whereas vegetation composition uses valid vegetation-class pixels. Visible vegetation-class diversity is quantified using the Shannon diversity index (H′). The single-image index describes individual images, while the weighted index summarises the dataset. Increasing vegetation cover does not necessarily increase diversity when one class becomes dominant.
<!-- Optional: upload Figures S7 and S8 and uncomment.
![Examples of high and low vegetation-class diversity](assets/vegetation-diversity-examples.png)
*Examples of high and low visible vegetation-class diversity in SE-NM and BE-LE (Figure S7).*

![Temporal vegetation composition examples](assets/vegetation-composition-temporal.png)
*Vegetation composition and H′ at the beginning and end of deployments SE-NM_113 and BE-LE_138 (Figure S8).*
`5-1a_calculate-dataset-scene-composition.py` – Summarises predicted class composition across the dataset.
`5-1b_class-cover-single-deployment.py` – Summarises class cover for an individual deployment.
`5-1c_Shannon-index-weighted.py` – Calculates a weighted Shannon index across the dataset.
`5-1d_Shannon-index-single.py` – Calculates the Shannon index for an individual image.
`5-1e_composition-Shannon-single-deployment.py` – Summarises composition and Shannon diversity for an individual deployment.
5.2 Snow cover
Snow-cover proportion is calculated using ground-cover classes only. PhenoTrap estimates broadly matched previous manual estimates of temporal variation in snow cover.
`5-2a_calculate-snow-cover.py` – Calculates snow-cover estimates.
`5-2b_calculate-snow-cover-comparison.py` – Compares PhenoTrap snow-cover estimates with previous manual estimates.
5.3 Vegetation greenness and phenology
Class-specific GCC is calculated using available pixels within extracted ROIs; masked pixels are excluded. Observations are aggregated daily and smoothed using a Savitzky–Golay filter. Breakpoint information from a previous study is entered manually for threshold- and slope-based analyses. Comparison with whole-image GCC illustrates the differences between ROI-based greenness estimates and estimates obtained without segmentation.
`5-3a_calculate-greenness-whole-image.py` – Calculates whole-image greenness.
`5-3b_calculate-greenness-classes.py` – Calculates class-specific greenness.
`5-3c_greenness-curve-overall-continuous.py` – Produces overall continuous greenness curves.
`5-3d_greenness-curve-overall-annual.py` – Produces overall annual greenness curves.
`5-3e_greenness-curve-annual-single-site-single-class.py` – Produces annual greenness curves for a single site and class.
`5-3f_greenness-curve-annual-multi-site-single-class.py` – Produces annual greenness curves for multiple sites and a single class.
`5-3g_greenness-curve-annual-multi-site-multi-class.py` – Produces annual greenness curves for multiple sites and classes.
`5-3h_greenness-curve-annual-multi-site-all-classes.py` – Produces annual greenness curves for multiple sites and all classes.
`5-3i_greenness-curve-continuous-multi-site-all-classes.py` – Produces continuous greenness curves for multiple sites and all classes.
`5-3j_greenness-curve-breakpoints-threshold.py` – Supports threshold-based breakpoint analysis of greenness curves.
`5-3k_greenness-curve-breakpoints-slope.py` – Supports slope-based breakpoint analysis of greenness curves.
`5-3l_greenness-curve-compare-to-whole-image.py` – Compares greenness curves with whole-image greenness.
6. PhenoTrap Utilities
These scripts support metadata management, file organisation, dataset checks, visualisation and troubleshooting. They can be used independently as required. Review settings before running scripts that move or delete files.
6.1 Metadata
`6-1a_extract-camera-metadata.py` – Extracts camera metadata.
`6-1b_camera-trap-model-metadata.py` – Processes camera-trap model metadata.
`6-1c_list-all-deployments.py` – Lists deployments.
`6-1d_count-deployment-id.py` – Counts deployment IDs.
`6-1e_copy-deployment-id-to-image-list.py` – Adds deployment IDs to an image list.
`6-1f_list-images-add-date-time.py` – Lists images with date and time information.
`6-1g_add-date-to-inference-results.py` – Adds dates to inference results.
`6-1h_copy-image-location-to-csv.py` – Copies image location information to CSV.
`6-1i_rename-imagery-JSON.py` – Renames imagery references in JSON data.
6.2 Storage
`6-2a_unzip-all-subfolders.py` – Extracts archives in subfolders.
`6-2b_image-subfolder-flatten.py` – Flattens image subfolder structures.
`6-2c_move-images-using-xlsx.py` – Moves images using an Excel file.
`6-2d_move-images-using-suffix.py` – Moves images based on filename suffixes.
`6-2e_move-images-using-JSON.py` – Moves images using JSON data.
`6-2f_create-unique-deployment-folder-names.py` – Creates unique deployment folder names.
`6-2g_remove-unlisted-images.py` – Removes images absent from a reference list.
`6-2h_delete-duplicate-imagery.py` – Deletes duplicate imagery.
6.3 Checks
`6-3a_check-missing-image.py` – Checks for missing images.
`6-3b_check-mask-class.py` – Checks mask class information.
`6-3c_count-class-partition-sets.py` – Counts classes within dataset partitions.
`6-3d_count-class-whole-dataset.py` – Counts classes across the complete dataset.
6.4 Visualisation
`6-4a_dataset-timeline-annual.py` – Plots annual dataset timelines.
`6-4b_dataset-timeline-continuous.py` – Plots continuous dataset timelines.
`6-4c_plot-class-abundance-bar-chart.py` – Plots class abundance as a bar chart.
`6-4d_plot-training-curve.py` – Plots training curves.
`6-4e_plot-2-training-curves-comparison.py` – Compares two training curves.
`6-4f_plot-confusion-matrix.py` – Plots a confusion matrix.
`6-4g_plot-2-confusion-matrices.py` – Compares two confusion matrices.
`6-4h_plot-class-iou-bar.py` – Plots class-level intersection over union (IoU).
`6-4i_convert-greyscale-mask-to-colour.py` – Converts greyscale masks to colour.
`6-4j_greyscale-to-colour-legend.py` – Generates a colour legend for masks.
`6-4k_vegetation-composition-quadrant.py` – Produces a vegetation composition quadrant visualisation.
`6-4l_plot-rgb-manual-prediction-comparison.py` – Compares RGB imagery, manual annotations and predictions.
`6-4m_plot-example-inference-imagery.py` – Plots example inference imagery.
`6-4n_plot-camera-trap-image-examples.py` – Plots example camera-trap imagery.
6.5 Troubleshooting
`6-5a_deployment-length.py` – Examines deployment lengths.
`6-5b_annotation-mask-count-JSON.py` – Counts annotation masks using JSON data.
`6-5c_segmentation-class-abundance-error.py` – Examines segmentation class abundance errors.
`6-5d_savgol-sensitivity-analysis.py` – Examines sensitivity to Savitzky–Golay smoothing settings.
6.6 Additional utilities
<!-- Confirm the parent directory for these scripts before adding relative links. -->
`6-6a_find-missing-images.py` – Identifies missing images.
`6-6b_combine-pixel-summary.py` – Combines pixel summary data.
`6-6c_create-mask-from-ROI.py` – Reconstructs masks from class-specific ROI images.
Model Availability
The pre-trained model weights and an archived version of the GitHub repository are to be made available through Zenodo. The model can be applied to entirely new camera-trap datasets without retraining.
<!-- Insert the verified Zenodo DOI and checkpoint download link here when available. -->
The annotated dataset is not publicly distributed but is available upon request.
Limitations
Segmentation performance varies among classes and study sites. Performance on new datasets should be assessed before ecological interpretation, particularly for less-abundant classes. The suitability of ecological outputs depends on the segmentation accuracy of the relevant classes.
Image-derived cover represents visible cover within the camera field of view, not necessarily physical ground-area cover. The Shannon index represents visible vegetation-class diversity rather than species diversity. Class-specific GCC describes greenness of the visible predicted class and may be affected by segmentation errors and image conditions.

Documentation

Additional documentation is available in the `Documentation` directory:
Naming Convention
Quality Control
Labelling Strategy
Citation and licence
PhenoTrap is associated with the submitted manuscript PhenoTrap: Semantic Segmentation of Environmental and Vegetation Classes in Camera-Trap Imagery. Publication and archival citation details will be added when available.
The repository is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence.
Acknowledgements
The computations/data handling were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) at C3SE, on the Alvis cluster, partially funded by the Swedish Research Council (grant agreement no. 2022-06725).
