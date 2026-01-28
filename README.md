# Pneumo-Fluor-Toolkit
This branch contains the denoising pipeline for fluorescence microscopy images.  
It uses Noise2Void (self-supervised, no clean ground truth required) together with a U-Net for background suppression and structure enhancement.  
For volumetric datasets, a 3D convolutional model is included to exploit z-context and reduce slice-to-slice noise.  
Outputs from this branch are intended as standardized inputs for downstream segmentation and feature extraction.
