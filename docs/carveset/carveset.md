# CarveSet Dataset V1.0
We have collected an extensive dataset covering the most common categories of objects intended for background removal. It includes about 179 object names belonging to 8 different categories. (CarveSet subset)

Categories: people, animals, cars, household items, everyday objects, electronics, children's toys, clothing, kitchen utensils.

Some examples of objects: animals, man, woman, group of people, dishes, furniture, car, cosmetics, and others.

Total number of images in the dataset: **20 155**.

Dataset Split:

-   Test set: **2 000** image pairs.
-   Validation set: **2 000** image pairs.
-   Training set: **16 155** image pairs.

The annotation was done in a semi-automatic mode with subsequent human verification and a special classifier.

## Information about the image database in the dataset
<div align="center"> <img src="../imgs/carveset/carveset_pair_example.png"></div>

1.  **CarveSet** - contains 4 583 high-quality images with a size of approximately 2500x2500 pixels, collected from open sources.
    

<div align="center"> <img src="../imgs/carveset/duts_hd.png"></div>

2.  **DUTS-HD** - consists of 15 572 images, magnified 4 times from the [DUTS](http://saliencydetection.net/duts) dataset, with a size of approximately 1600x1600 pixels. The dataset was re-annotated with controlled enhancement of the output mask. The images were upscaled, which added new details (see figure).

Two versions of the dataset are provided:

1.  A merged image database, which is randomly divided into 3 sets: train, val, test. Along with the datasets, tables with paths to all images and alpha channel and trimap maps are provided.
    
2.  Separate subsets "CarveSet" and "DUTS-HD" are placed in individual folders. The "DUTS-HD" dataset is presented in its original structure, divided into test and training sets.


## Dataset File Structure

###  Separated version of the structure - carveset_no_split_29_07.zip

-   `SHARE_carveset_all` - subset of CarveSet data.
    -   `images` - folder containing RGB images.
    -   `masks` - folder containing annotations (masks).
-   `all_images.csv` - table with file paths.
-   `SHARE_duts-hd` - subset of DUTS-HD data.
    -   `train` - training set from the original split of the DUTS dataset. Nested structure similar to `SHARE_carveset_all`.
    -   `test` - testing set from the original split of the DUTS dataset. Nested structure similar to `SHARE_carveset_all`.

### Merged version of the structure - carveset_splitted.zip

-   `SPLITTED_carveset_duts` - merged image database.
    -   `test` - folder containing the testing set.
    -   `train` - folder containing the training set.
        -   `images` - folder containing RGB images.
        -   `masks` - folder containing annotations (masks).
        -   `trimaps` - folder containing trimap annotations.
    -   `val` - folder containing the validation set.
    -   `test.csv`, `train.csv`, `val.csv` - tables with file paths.

## Download:

The dataset is provided under [special terms of use.](./terms_of_use.pdf)
In short, the dataset is provided for non-commercial research purposes only.
By downloading the dataset, you agree to the terms of use.

1.  CarveSet (split version): [Google Drive](https://drive.google.com/file/d/16CAhyDqWiUN-D3u5GpQqjg5RGFICi4eb/view?usp=drive_link)
2.  CarveSet (unsplit version): [Google Drive](https://drive.google.com/file/d/1JoiQfuiJPh9_AEH5QNpURAPfa6xB9Xng/view?usp=drive_link)