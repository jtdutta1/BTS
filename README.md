# Automatic Segmentation of Brain Tumours in Flair, T1 weighted, T2 weighted, T1 weighted with contrast enhancements volumetric MRI scans of the brain using Time-Distributed 2D Convolutional Neural Networks placed in an U-Net architecture.

We design an unique way to train a U-Net on lower end Hardware. 

### What is a U-Net?
A **U-Net** is a CNN architecture designed by Ronneberger et al. ([link](https://arxiv.org/pdf/1505.04597.pdf)) which is used to segment medical images. 
![U-Net architecture](/images/unet1.png )

### Points about the data.

![4-types of scans](/images/4-types-of-scan.jpg)
This represents the 4 types of scans present in the dataset. From top-left clockwise:-
  1. Flair
  2. T1
  3. T2
  4. T1c
 Out of these 4 T1c outlines the tumour regions efficiently. Below is the ground truth along with a colored image representing the different modalities in the tumour.
 ![tumour](/images/Ground-Truth.png) ![colored-tumour](/images/Colored-Ground-Truth.png)
 
 ### Our model's architecture
