[//]: # Brief overview on what the package does

Camera traps are cameras with a motion sensor that take a picture
whenever something moves infront of them. They are often
used as an inexpensive and noninvasive method of observing animals 
in the wild, however they do have one major drawback. As they are
triggered by any movement, wind and changes in lighting can
severly clutter databases with images that do not have any animal.

Megadetector is an algorithm developed by Microsoft that can automatically
detect and draw a bounding box around an animal in an image. This package 
provides an easy-to-use function that identifies the species of the animals
that were detected by the algorithm and puts them into folders that are
sorted by species. It was originally designed to work for a model that
was trained to categorise animals in Hong Kong however it should be easy
to adapt it for any model trained with Keras 
(see [the 'modify pipeline' section] (#-How-to-modify-package-for-new-model) 
for details).

## How to run program (maybe it should be called "Quick Start" instead?)

### Download model

[//]: # Tell user to either email Calvin for model or to train their own and change the code

### Run function

```python
pip install "path\to\package"
```

```python
from classify_animals.main import main as classy_func

classy_func(
    bb_results_path = 'path\to\Megadetector\output\json\string', 
    image_dir = 'path\to\raw\images',
    model_path = 'path\to\trained\model'
)
```

## Inputs

[//]: # Talk about CSV file and 
The program requires three files/directories to run.
- *Bounding boxes* - These are the animal detections that were found
    after running Megadetector. (See (this section)[#-Megadetector]
    to find out how to get this).
- *Images* - Uncropped images that were fed into
	megadetector. The path passed into the classification function
	should be the same as what was passed into Megadetector's detection
	function when the bounding boxes were generated.
- *Model* - Model that was trained to classify cropped images of animals.
    It should be saved using either Keras's H5 format or some other format
	that can be read by Keras's 'model.load_model' function

### Megadetector

[//]: # Talk about how it needs Megadetector's detections as an input
[//]: # Talk about version of megedetector used by the model
[//]: # and add link to their github

## Outputs

This program has two outputs. The first is a CSV file called 'Animal Labels'
that lists the species of animal detected for each image. The data
dictionary for this file is as follows:

- *original_image_path* - File path of the original uncropped image
    relative to the directory that was passed into 'image_dir'
- *cropped_image_path* - File path to the cropped image relative to
    the folder called 'cropped_images'
- *bb_conf* - Confidence level of bounding box of the animal given
    by Megadetector
- *species* - Species of the animal as predicted by the model
- *class_ent* - Level of uncertainty in the model's prediction for
    the species of the animal

The second is a folder called 'cropped_images' that contains the cropped
images of animals. It contains a subfolder for each species that was
detected by the algorithm. Additionally, if a threshold value for the
level of uncertainty in the species classifications then it will also 
contain a subfolder called "zz_unknown" for the animals that it could
not classify.

Given as is, the function is designed to classify animals into only 1 of 15
different species, but it can be modified to include any number of categories
as long as the output of a neural network is given by the node with the
highest value. 

### Output folder

The function will create a folder within which all of the intermediary files
that are required to process the data as well as both final outputs will be stored.
All intermediary files will be deleted once the program has processed all of the
data.

By default, the folder will be created in the working directory and be called
"Animal Detections". To change where the output is stored then provide
the path to the directory by using the optional parameter 'working_data_dir'
of the function.

*Warning* please do not open, move or delete any files in
this directory while the program is still running as this will likely cause
it to crash.

[//]: # Talk about output CSV and sorted cropped data
[//]: # Talk about categories of animals

[//]: # Talk about optional working directory file


## Optional Parameters

[//]: # Batch sizes
[//]: # Warn that if keep_crops = False then if more than one
[//]: # animal is detected in an image then there will be no
[//]: # way of knowing which label refers to which animal

## How to modify package for new model

[//]: # Talk about modifying config's global variable
[//]: # Talk about changing the image preprocessing function


## Example?

## Citation

[//]: # Tell people how to cite this code

## Acknowledgements 

	[//]: # (thank the people who provided the data?)
	[//]: # Acknowledge that some of the code and that
	[//]: # the idea for the pipeline came from Megadetector