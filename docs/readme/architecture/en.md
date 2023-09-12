## Basic description of the project architecture
The software package is a solution for automatically removing backgrounds from images using neural networks.

Architecturally, the software consists of the following components:
* Preprocessing module;
* Segmentation module;
* Post-processing module.

During processing, the image is transmitted sequentially through the specified modules, possibly as part of a package of images (depending on the configuration set by the user during use)

## Description of module functionality
### Preprocessing module
It consists of scene and object classifiers, which are used depending on the complexity of the processed image and determine the segmentation modules used in the future.
### Segmentation module
The segmentation module is presented in two options: TRACER and ISNet, the specific module is selected during preprocessing.
The segmentation module creates a mask from the image, which represents the probability for each pixel in the image to be part of the main object in the image.
### Post-processing module
The post-processing module eliminates errors during segmentation, bringing the mask to the required quality. During post-processing, two models are used sequentially: CascadePSP and FBA Matting. They implement fundamentally different approaches and refine only the edges of the mask in those areas where the probability returned by the segmentation module does not allow us to judge with sufficient confidence whether a pixel belongs to the image. After post-processing, the mask is applied to the images, leaving visible only those pixels of the input image that are included in it.


