# -*- coding: utf-8 -*-
"""
@author jtdut

@version 1.4

File parser



"""
import os
import sys
import imageio
from image_preprocess import imagepreprocess as improc
#import



'''<<<hgg/lgg>>>_<<<patitent_no>>>_<<<MRI_type>>>_<<<slice_no>>>.png'''



def create_folder(path):
    graded_folder_list = os.listdir(path)
    skipped = 0
    for l in graded_folder_list:
        count = 1
        path_to_patient_list = os.path.join(path,l)
        patient_folders_list = os.listdir(path_to_patient_list)
        for l1 in patient_folders_list:
            path_to_patient = os.path.join(path_to_patient_list,l1)
            brain_scans_folders_list = os.listdir(path_to_patient)
            for l2 in brain_scans_folders_list:
                path_to_patient_info = os.path.join(path_to_patient,l2)
                folder_items_list = os.listdir(path_to_patient_info)
                scan_type = ""
                path_to_brain_scan = ''
                for l3 in folder_items_list:                    
                    if('Flair' in l3):
                        scan_type = 'Flair'
                    elif('T2' in l3):
                        scan_type = 'T2'
                    elif('T1c' in l3):
                        scan_type = 'T1c'
                    elif('T1' in l3):
                        scan_type = 'T1'
                    elif('OT' in l3):
                        scan_type = 'OT'
                    else:
                        scan_type = ''
                    path_to_brain_scan = os.path.join(path_to_patient_info,l3)
                if(len(scan_type)>0):
                    #print(path_to_brain_scan)
                    img_pre = improc(path=path_to_brain_scan)
                    top_view = img_pre._getTopView()
                    #print('scan_type= ',scan_type)
                    '''accept = input('Continue?')
                    if(accept == 'no'):
                        sys.exit(0)'''
                    if(scan_type == 'OT'):
                        for i in range(5,151):
                            temp = i - 5
                            image_file_name = 'labels/'+l+'_'+str(count)+'_'+scan_type+'_'+str(temp)+'.png'
                            imageio.imwrite(image_file_name,top_view[i])
                    else:
                        for i in range(5,151):
                            temp = i - 5
                            image_file_name = 'features/'+l+'_'+str(count)+'_'+scan_type+'_'+str(temp)+'.png'
                            imageio.imwrite(image_file_name,top_view[i])
                else:
                    skipped = skipped + 1
            count = count + 1
    print('\nSkiped: ',skipped)
    
if(__name__ == '__main__'):
    create_folder('BRATS2015_Training')