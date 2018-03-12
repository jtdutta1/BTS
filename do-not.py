# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import imageio
from image_preprocess import imagepreprocess as improc
import numpy as np
import

'''<<<hgg/lgg>>>_<<<patitent_no>>>_<<<MRI_type>>>_<<<slice_no>>>.png'''

def create_folder(path):
    graded_folder_list = os.listdir(path)
    saving_path = 'training_data'
    count = 1
    for l in graded_folder_list:
        path_to_patient_list = os.path.join(path,l)
        patient_folders_list = os.listdir(path_to_patient_list)
        for l1 in patient_folders_list:
            path_to_brain_scan = os.path.join(path_to_patient_list,l1)
            brain_scans_list = os.listdir(path_to_brain_scans)
            for l2 in brain_scans_list:
                path_to_the_brain_image
            img_pre = improc(path_to_brain_scan)
            top_view = img_pre._getTopView()
            for i in top_view:
                imageio.imwrite(l+'_'+str(count)+)