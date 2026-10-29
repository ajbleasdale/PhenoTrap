# Image Labelling

This guide describes the labelling procedure used to create semantic segmentation datasets. Annotation is performed in Label Studio using polygon annotations.
The aim is to produce consistent vegetation and environmental labels while recognising that camera-trap imagery can contain complex boundaries, distant vegetation and features that cannot always be identified with certainty.
Annotation workflow
•	Images are annotated individually using polygon annotations.

•	The entire image should be considered for annotation.
•	All target classes should be labelled where they can be reasonably identified.
•	Areas that cannot be identified with sufficient confidence can be assigned to Ignore.
•	Annotation boundaries should be moderately precise. Follow the major visible boundaries of an object or vegetation class, but do not spend excessive time tracing highly complex boundaries.
•	Small gaps or individual pixels do not need to be exhaustively annotated where doing so would require disproportionately complex polygons.

## General labelling rules
### Uncertainty
Labels should primarily be assigned according to the visible features in the image. Contextual information can also be used where the class is difficult to determine.
Useful contextual information includes:
•	surrounding vegetation and objects;
•	position within the image and vegetation structure;
•	season;
•	site characteristics;
•	images from earlier or later in the same camera-trap time series; and
•	the original image where the displayed or processed image makes features difficult to distinguish.
Small uncertain regions can generally be assigned to the most likely class where any error would affect only a small number of pixels.

Where a large region cannot be confidently identified, it should be assigned to Ignore rather than forcing an uncertain class assignment.
For distant or low-resolution vegetation, use visible features and contextual information where possible. The briar_seedling class accounts for some low-growing broadleaf vegetation that cannot reliably be classified more finely. Where there is insufficient evidence to assign a class, use Ignore.

The same labelling criteria should be applied to training, validation and testing imagery.

### Visible surfaces and overlapping classes
As a general rule, label the visible surface rather than the object underneath it.
This rule is relaxed where a covering does not substantially obscure the structure and identity of the underlying vegetation. For example, light or powdery snow through which the vegetation remains clearly identifiable can retain the vegetation class.

Examples include:
•	large accumulations of snow obscuring vegetation → snow_ice;
•	light/powdery snow with the underlying vegetation clearly visible → underlying vegetation class;
•	moss growing over rock → visible moss as cryptogam, exposed rock as bare;
•	vines covering tree foliage → visible vine-covered area as vine;
•	vegetation in shadow → underlying vegetation class where its features remain identifiable.

### Seasonal and dead vegetation
Where vegetation remains identifiable, seasonal changes should not normally result in a change of class.
For example:
•	brown or dead grass remains graminoid;
•	senescent or dead ferns remain fern;
•	senescent forbs remain forb;
•	thin stems that remain identifiable as living vegetation can remain within the appropriate vegetation class;
•	dead woody branches are background.
Special consideration is required for deciduous trees during winter; see Broadleaf deciduous.

### Scale and distance
The appropriate level of annotation detail depends on how clearly vegetation structures can be resolved.
For nearby vegetation, individual leaves or groups of leaves can be delineated where practical. Visible branches and sufficiently large gaps between foliage should not be included within the vegetation polygon.
For distant vegetation, crowns can be annotated as more continuous regions where individual leaves, shoots and small gaps cannot reasonably be resolved.
Dense groups of trees can therefore be annotated collectively where separating individual crowns is impractical.
If a distant tree cannot be confidently classified, for example as conifer or broadleaf deciduous, use ignore.
Sky and canopy gaps
Clearly identifiable sky should be labelled sky.
Within sparse conifer and deciduous crowns, aim to capture visible sky gaps where practical. However, very small or highly fragmented gaps do not need to be exhaustively delineated.
For nearby crowns, clearly visible gaps and spaces between individual leaves should be labelled as sky where sufficiently large to annotate.
For distant crowns, very small internal gaps can be incorporated into the tree polygon where individual crown components cannot reasonably be resolved.
________________________________________
## Class Guide
Class	Includes	Key identification and notes
background	Tree trunks, branches, stumps, exposed roots and non-target objects	Used for areas that do not belong to one of the target environmental classes
sky	Visible sky	Label clear canopy gaps but do not exhaustively trace very small fragmented gaps
snow_ice	Snow and ice	Includes heavy snow obscuring vegetation; light snow can retain the underlying vegetation class
bare	Bare soil, exposed rock, gravel, mud and leaf/needle litter	Includes surfaces without sufficient living vegetation cover
ericaceous_shrub	Heather, bilberry and similar low-growing shrubs	Important low-growing woody vegetation, particularly in boreal environments
briar_seedling	Briars and broadleaf woody seedlings	Functional class for low-growing woody broadleaf vegetation that cannot reliably be separated more finely
graminoid	Grasses, sedges and rushes	Includes green, senescent and dead graminoid vegetation
forb	Broadleaf herbaceous plants	Generally distinguished from briars/seedlings by herbaceous rather than woody stems
fern	Ferns	Includes green and senescent/dead ferns where identifiable
cryptogam	Bryophytes and terrestrial lichens	Includes mosses and substantial terrestrial lichen cover
vine	Woody vines	Label visible vine-covered vegetation as vine
broadleaf_evergreen	Evergreen broadleaf vegetation such as holly	Often darker, tougher and/or shinier foliage; particularly identifiable during winter
conifer	Coniferous trees	Includes all conifers
broadleaf_deciduous	Deciduous broadleaf trees	Includes distant leafless crowns where tree identity remains clear
ignore	Regions that should not contribute to training/evaluation	Used for substantial uncertain regions and specified excluded objects
Background
Background includes non-target features such as:
•	tree trunks;
•	large branches;
•	fallen branches;
•	tree stumps;
•	exposed roots; and
•	artificial objects and structures such as fences and buildings.
Small or thin dead woody debris can be incorporated into bare where it forms part of the ground surface. Larger identifiable branches remain background.
Pixels that are not covered by an annotation are also treated as background. The aim is therefore to minimise unintended gaps between annotations, although small gaps may remain where boundaries are particularly complex.
Ignore
ignore is used for regions that cannot be labelled reliably or that should deliberately be excluded from model training and evaluation.
This includes:
•	large regions where the vegetation or surface class cannot be confidently identified;
•	distant or low-resolution features where there is insufficient information to assign a class;
•	animals and people; and
•	regions where image quality, such as severe blur, darkness or overexposure, prevents reliable identification.
In the exported masks, Ignore pixels are assigned a value of 255. This value is excluded within both the training and testing scripts, meaning that Ignore pixels do not contribute to the training loss or model evaluation.
Uncertain regions that should be excluded should therefore be explicitly labelled as ignore rather than simply left unannotated, as unannotated pixels are treated as background.
Sky
Label as much clearly identifiable sky as practical.
This is particularly challenging within sparse conifer crowns and leafless deciduous vegetation. Clearly visible gaps should be annotated, but very small gaps do not need to be exhaustively traced.
For distant crowns, small gaps can be incorporatedated into the tree class where individual branches and gaps cannot reasonably be resolved.
Snow/Ice
Label visible snow and ice as snow_ice.
Where snow occurs on vegetation:
•	heavy snow / large accumulations: label the visible snow as snow_ice;
•	light or powdery snow: retain the vegetation class where the underlying structure remains clearly visible.
The aim is to distinguish snow that visually replaces the vegetation surface from snow that only partially covers an otherwise identifiable plant.
Bare
bare includes:
•	bare soil;
•	exposed rock;
•	gravel;
•	mud;
•	leaf litter; and
•	needle litter.
Dead herbaceous vegetation should not automatically become Bare. Where its vegetation type remains identifiable, retain the corresponding vegetation class.
Very sparse vegetation can be treated as Bare where only a small number of isolated blades or shoots occur over an otherwise bare surface.
Ericaceous shrubs
This class includes low-growing ericaceous shrubs such as:
•	heather; and
•	bilberry.
The class is intended to capture low-growing shrubs that can form substantial ground cover, particularly in boreal environments.
It should be distinguished from briar_seedling, which represents other low-growing woody broadleaf vegetation that cannot reliably be separated more finely.
Briar/Seedling
briar_seedling is a functional annotation class rather than a conventional taxonomic group.
It includes low-growing woody broadleaf vegetation such as:
•	broadleaf tree seedlings; and
•	briars such as bramble.
These vegetation types can be difficult to distinguish reliably from camera-trap imagery, particularly at greater distances.
Useful features include woody stems and relatively tough broad leaves.
The class can be distinguished from:
•	forb by the presence of woody rather than herbaceous stems;
•	broadleaf_deciduous primarily by growth form and location, with broadleaf deciduous generally representing tree crowns/overstory and briar_seedling representing low-growing vegetation;
•	ericaceous_shrub where characteristic ericaceous growth forms can be identified.
Where low-resolution foreground broadleaf vegetation cannot be separated more finely but appears consistent with this functional group, briar_seedling can be used.
Where there is insufficient evidence even for this distinction, use ignore.
Graminoid
Includes:
•	grasses;
•	sedges; and
•	rushes.
Graminoids are generally characterised by long, relatively narrow leaves or blades.
Brown, senescent or dead graminoid vegetation remains graminoid where its structure can still be identified.
When distinguishing graminoids from forbs, look for narrower leaves and grass-like structure. Broad leaves and identifiable flowers may provide evidence for forb.
Forbs
Forbs include broadleaf herbaceous vegetation.
Useful characteristics include:
•	broad leaves;
•	herbaceous rather than woody stems; and
•	identifiable flowers where present.
Forbs should be distinguished from briar_seedling, which generally has woody stems and tougher foliage.
Senescent or dead forb vegetation remains forb where it can still be identified.
Ferns
All identifiable ferns should be labelled fern.
Senescent and dead ferns remain within this class where their characteristic structure remains recognisable.
Cryptogams
The cryptogam class primarily includes:
•	bryophytes, including mosses, occurring both terrestrially and epiphytically; and
•	terrestrial lichens, including substantial ground-covering lichens such as reindeer lichens.
Where moss grows over rock or another substrate, label the visible moss as cryptogam and the surrounding exposed substrate according to its visible class, typically bare.
Epiphytic bryophytes, such as moss growing on tree trunks or branches, should also be labelled as cryptogam where they can be reasonably identified.
Lichens should only be labelled where they occur terrestrially. Epiphytic lichens growing on trunks or branches are not annotated. These can be numerous, small and difficult to delineate consistently, making exhaustive annotation disproportionately time-consuming for their relevance to vegetation green-up.
Vine
Label visible woody vine vegetation as vine.
Where a vine covers a tree, label the visible vine-covered area as vine. Areas of the underlying tree should only receive the tree class where its own foliage is visibly distinguishable.
Dark green leaf tissue and the position of foliage along climbing woody stems can help distinguish vines from bare branches.
Broadleaf Evergreen
Includes broadleaf evergreen vegetation such as holly.
Useful characteristics include:
•	persistent foliage during winter;
•	relatively dark green leaves; and
•	tougher and often shinier leaves than many deciduous species.
Winter imagery can be particularly useful for distinguishing broadleaf evergreen from broadleaf deciduous vegetation.
Conifer
All identifiable coniferous trees should be labelled conifer.
Visible trunks and sufficiently large exposed branches remain background.
For distant crowns, small internal gaps and fine branches can be incorporatedated into the crown where they cannot reasonably be separated.
Broadleaf Deciduous
Includes deciduous broadleaf trees.
During the growing season, annotate visible foliage while excluding clearly resolvable branches and sufficiently large gaps.
During winter, leafless deciduous crowns can still be labelled broadleaf_deciduous where:
•	the tree/crown can be confidently identified;
•	the crown is sufficiently distant that individual shoots cannot reasonably be separated; or
•	shoots are sufficiently dense that the crown forms a continuous visual structure.
For nearby trees where individual bare branches are clearly resolved, the branches themselves should remain background.
This means annotation is intentionally scale dependent: a distant leafless crown may form a broadleaf_deciduous polygon even though individual bare branches would be background when viewed at higher resolution.
________________________________________
Common Ambiguous Cases
Situation	Recommended annotation
Heavy snow obscuring vegetation	snow_ice
Light/powdery snow with vegetation clearly identifiable	Underlying vegetation class
Moss covering rock	Moss = cryptogam; exposed rock = bare
Vine covering tree foliage	Visible vine = vine
Vegetation in shadow but still identifiable	Underlying vegetation class
Completely obscured/unidentifiable vegetation	ignore
Dead/brown grass	graminoid
Senescent/dead fern	fern
Senescent forb	forb
Leaf or needle litter	bare
Fallen large branch	background
Small woody debris incorporated into ground surface	bare
Tree trunk/stump/exposed root	background
Distant identifiable leafless deciduous crown	broadleaf_deciduous
Distant tree with uncertain type	ignore
Tiny fragmented sky gaps	Do not need exhaustive annotation
Animal or person	ignore
Severe blur/darkness/overexposure preventing identification	ignore
Image-edge vegetation	Annotate normally where identifiable
________________________________________
Human-in-the-Loop Annotation
Overview
Human-in-the-loop (HITL) annotation can reduce the time required to expand an annotated dataset by using predictions from an existing trained model as initial annotations.
The general workflow is:
Select candidate imagery → run model inference → review prediction overlays → select images for annotation → convert predictions to Label Studio polygons → manually review and correct predictions → export corrected annotations → retrain and evaluate
Model predictions should always be treated as initial annotations rather than ground truth. The same labelling rules described above should be applied when reviewing HITL annotations.
1. Select candidate images
Begin by selecting a larger pool of images from the new dataset/deployments.
For example, when aiming to add approximately 250 annotated images, we have used a random sample of approximately 2,500 candidate images for initial model inference.
These numbers are examples from our workflow rather than fixed requirements and can be adjusted according to the required dataset size and available annotation time.
Where possible, retain several deployments independently from the HITL process in case additional validation or testing data are required. In our workflow, approximately 2–5 deployments may be retained for this purpose.
Images subsequently selected and corrected through HITL should be considered part of the model-development process rather than independent test data.
2. Run inference
Apply the existing trained segmentation model to the candidate images.
Generate both:
•	integer prediction masks; and
•	mask overlays showing predictions on the original images.
The overlays are useful for rapidly reviewing prediction quality without having to interpret integer-valued masks directly.
3. Select images for annotation
Review the prediction overlays and select a mixture of images.
Selection can include:
•	images where the model performs well;
•	images where the model performs poorly;
•	images containing rare or underrepresented classes;
•	images representing novel lighting or environmental conditions; and
•	images from new deployments.
There is a trade-off between annotation speed and potential information value.
Images with good predictions can be corrected quickly and are useful for efficiently increasing representation across different deployments and lighting conditions.
Poorly predicted images may require substantially more annotation effort but can provide examples of conditions or vegetation types on which the existing model currently struggles.
Where a prediction is particularly poor, it may be faster and more reliable to delete the predicted polygons and annotate the image from scratch.
Even predictions that appear accurate should be manually checked before being accepted.
4. Collect selected images and masks
After selecting the HITL images, collect the corresponding:
1.	original images;
2.	integer prediction masks; and
3.	prediction overlays.
The following script is used for this step:
2-7b_HITL-overlay-mask-image-index.py
This produces the corresponding files required for the subsequent Label Studio import workflow.
Important: Ensure that the image index refers to the actual subfolder containing the HITL images.
5. Copy images to the Label Studio image directory
Copy the selected HITL image folder into the directory used to serve the other Label Studio images.
The subfolder value in the image index must match the name of the folder containing the HITL images. For example:
image_name	mask_name	subfolder
SE-KB_1_2023-12-01_120000.jpg	SE-KB_1_2023-12-01_120000.png	SE-KB-HITL-images
In this example, the selected image should therefore be located within the SE-KB-HITL-images folder.
Ensure that the folder name and the subfolder entry match before generating the Label Studio JSON, as these are used to construct the image paths for import.
6. Convert prediction masks to Label Studio polygons
Predicted integer masks must be converted into polygon annotations that can be imported into Label Studio.
The following script performs this conversion:
2-8_HITL-mask-to-JSON.py
This creates a JSON file containing the model-generated annotations for import into Label Studio.
The local image server must be running so that Label Studio can access the images. Use:
2-1_serve-images-cors.py
This serves the image directory with the required CORS configuration for Label Studio.
7. Polygon extraction settings
Converting segmentation masks to editable polygons requires a balance between retaining boundary detail and producing polygons that can be efficiently corrected manually.
The following settings were used in our workflow:
# Polygon extraction tuning
MIN_AREA_PX     = 50
APPROX_EPS_FRAC = 0.0005

# Overlap, boundary and splitting controls
NO_OVERLAP      = True
ERODE_PX        = 1
SPLIT_OPEN_PX   = 2
MIN_AREA_PX
Controls the minimum region size retained during polygon extraction.
Very small predicted regions can generate large numbers of unnecessary polygons and increase annotation time.
APPROX_EPS_FRAC
Controls polygon simplification.
A polygon that is too complex can retain smooth or detailed boundaries but contain a very large number of points, making manual adjustment difficult.
Greater simplification makes polygons easier to edit but can reduce boundary detail.
The value above represents the setting used in our workflow rather than a universally optimal value.
NO_OVERLAP
Prevents pixels from being assigned to overlapping polygons from different classes.
Class priority is determined by PREDICT_CLASSES order.
ERODE_PX
Slightly shrinks predicted regions to reduce boundary overlaps.
A value of 0 disables erosion.
SPLIT_OPEN_PX
Applies morphological opening to help separate large connected prediction regions into more manageable components.
A value of 0 disables this operation.
8. Check overlapping polygons
During development, importing predictions into Label Studio sometimes resulted in polygons occurring within or overlapping other polygons.
These can cause problems during subsequent annotation or mask generation.
Where this occurs:
•	delete the incorrect overlapping polygon; or
•	manually adjust/trace the polygon around the appropriate boundary.
The overlap and boundary controls described above were introduced to reduce these cases, but imported predictions should still be checked manually.
9. Review and correct annotations
Review each imported prediction using the same class definitions and annotation rules used during manual labelling.
Corrections can include:
•	changing incorrect classes;
•	adjusting polygon boundaries;
•	adding missed regions;
•	deleting false detections;
•	resolving overlapping polygons; and
•	deleting predictions and annotating from scratch where correction would be more difficult than manual annotation.
Corrected HITL annotations should ultimately meet the same standard as annotations produced entirely manually.
10. Retrain in batches
HITL annotations can be accumulated into batches before retraining rather than continuously retraining after every small number of corrected images.
For example, several hundred corrected images may be accumulated before another training iteration.
The appropriate batch size depends on:
•	annotation availability;
•	size of the existing training dataset;
•	diversity of the newly added imagery;
•	model performance; and
•	computational cost of retraining.
After retraining, the updated model can be used to generate predictions for another HITL iteration.
11. Evaluate performance
Evaluate the retrained model on validation and/or test data that have not been incorporated into HITL training.
An improvement should not necessarily be expected after every HITL iteration.
A small number of additional training images may not provide sufficient new information to produce a measurable improvement. Similarly, where the validation or test dataset is small, changes in model performance may be difficult to distinguish reliably.
Evaluation should therefore consider both overall performance and class-specific performance, particularly for rare classes or conditions deliberately targeted during HITL.
________________________________________
Practical Principles
When annotating manually or correcting HITL predictions:
1.	Aim for consistency rather than pixel-perfect boundaries.
2.	Do not spend disproportionate amounts of time tracing highly complex polygons.
3.	Use contextual and temporal information where it helps identify vegetation.
4.	Use Ignore when substantial regions genuinely cannot be identified.
5.	Retain vegetation classes through seasonal changes where the vegetation remains identifiable.
6.	Adjust annotation detail to the spatial resolution of the vegetation.
7.	Treat HITL predictions as suggestions, not ground truth.
8.	Prioritise a mixture of annotation efficiency, dataset diversity and examples on which the model currently struggles.
