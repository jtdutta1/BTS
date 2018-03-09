## Automatic Detection and Segmentation of Graded Brain Tumor in Flair, T1, T1c, T2, &amp; OT weighted volume MRI scans of the brain using Deep Multi-Scale &amp; Multi-Modal Convolutional Neural Networks

### About the Data 

 

> The data is multimodal.

> The data is also in multi color format.

> The data has 4 types of format of MRI scans, FLAIR, T1w, T2wand T1wc.  

 

### How we are reading the Data 

 

> We are taking a single slice of the data initially. We are also working with the T1wc type MRI scan data. 

> It is then fed through a filter called Curvature Flow which filters the image of noise while still preserving most of the data.

>The T1wc data along with the output data of the same slice index is stored in a numpy array. We are also pickling this numpy array for faster access and pipelining our work flow. 

### Tumour Stability

**For model 0:**

- For top-down view tumour visible for the frames 50-100
- For side view the tumour visible for frames 110-60
- For back view the tumour is visible for frames 70-150

**For model 1:**

- For top-down view tumour visible for the frames 70-115
- For side view tumour visible for frames 160-120
- For back view the tumour is visible for frames 40-100

**For model 2:**

- For top-down view tumour visible for the frames 60-135
- For side view tumour visible for frames 165-65
- For back view the tumour is visible for frames 40-115

### Ideal Slices

- Top down: 70-100 
