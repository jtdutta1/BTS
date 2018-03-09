import os
import cv2
import SimpleITK
import numpy as np
import matplotlib.pyplot as plt

def sitk_show(img, title=None, margin=0.0, dpi=40):
	nda = SimpleITK.GetArrayFromImage(img)
	figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
	extent = (0, nda.shape[1], nda.shape[0], 0)
	fig = plt.figure(figsize=figsize, dpi=dpi)
	ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
	plt.set_cmap("gray")
	ax.imshow(nda,extent=extent,interpolation=None)
	if title:
		plt.title(title)
	plt.show()

filenameT1c = "./BRATS2015_Training/HGG/brats_2013_pat0001_1/VSD.Brain.XX.O.MR_T1c.54514/VSD.Brain.XX.O.MR_T1c.54514.mha"

imgT1cOriginal = SimpleITK.ReadImage(filenameT1c)

nda = SimpleITK.GetArrayFromImage(imgT1cOriginal[:, :, 88])

imgT1cSmooth = SimpleITK.CurvatureFlow(image1=imgT1cOriginal,
                                       timeStep=0.125,
                                       numberOfIterations=5)

sitk_show(SimpleITK.Tile(imgT1cSmooth[:, :, 88]))