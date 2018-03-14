import os
import sys
import imageio
from image_preprocess import imagepreprocess as improc
    
path = 'BRATS2015_Training/LGG'
patient_list = os.listdir(path)
for i in  range(46,54):
    path_to_patient_info = os.path.join(path,patient_list[i])
    folder_items_list = os.listdir(path_to_patient_info)
    for l in folder_items_list:
        path_to_brain_image = os.path.join(path_to_patient_info,l)
        #print(path_to_brain_image)
        brain_scan_folder_list = os.listdir(path_to_brain_image)
        scan_type = ""
        path_to_brain_scan = ''
        for l3 in brain_scan_folder_list:
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
            path_to_brain_scan = os.path.join(path_to_brain_image,l3)
            print(path_to_brain_scan)
        if(len(scan_type)>0):
            img_pre = improc(path=path_to_brain_scan)
            top_view = img_pre._getTopView()
                        
            if(scan_type == 'OT'):
                for j in range(5,151):
                    temp = j - 5
                    image_file_name = 'labels/LGG_'+str(i)+'_'+scan_type+'_'+str(temp)+'.png'
                    imageio.imwrite(image_file_name,top_view[i])
            else:
                for j in range(5,151):
                    temp = j - 5
                    image_file_name = 'features/LGG_'+str(i)+'_'+scan_type+'_'+str(temp)+'.png'
                    imageio.imwrite(image_file_name,top_view[i])