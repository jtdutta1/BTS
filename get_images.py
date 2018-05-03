import os
#import warning
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
    

def get_images(path, filter = False, slices = None, crop = False, valid_set = 0.1):
    """
    Returns a list T1c MRI scans pre-filtered if the condition is set and splits them into training and validation
    sets.

    INPUTS:-
    path: A string to the path of the root folder which has the HGG and LGG MRI scans.

    filter: Boolean. Sets a Curvature flow filter to all the images if the condition is set.(Optional)
    By default the filter will not be applied.

    scans: List of string(s). Tells the function to only accept the type(s) of scan provided in the parameter.(Optional)
    By default it returns all the types of scans.
    
    slices: List. If value is provided, it will send the slices between and including x and y, where x and y have to 
    be passed to slices in a list format [x, y]
    
    crop: Boolean. If set True, get_images will return cropped slices of resolution (192 X 192)

    valid_set: Float. Seperates a portion of the data as provided by this parameter for testing purpose.(Optional)
    By default it keep 10% of the data for testing.

    OUTPUT:-
    A tuple of 2 dictionaries. Train and Test. Each dictionary has two keys, 'features' and 'labels'.
    """
    graded_folder_list = os.listdir(path)
    if(slices == None):
        slices = [0, 155]
    features, labels = [], []
    for scan_type_dir in graded_folder_list:
        print('Extracting HGG')
        path_to_patient_dir = os.path.join(path, scan_type_dir)
        patient_list = os.listdir(path_to_patient_dir)
        for patient in tqdm(patient_list):
            path_to_patient_folder = os.path.join(path_to_patient_dir,patient)
            brain_scan_folders_list = os.listdir(path_to_patient_folder)
            necessary_brain_scan_folders_list = __necessary_folders__(brain_scan_folders_list)
            for brain_scan_folder in necessary_brain_scan_folders_list:
                path_to_patient_mri_scan = os.path.join(path_to_patient_folder,brain_scan_folder)
                folder_items = os.listdir(path_to_patient_mri_scan)
                mha_file = [x for x in folder_items if('.mha' in x)][0]
                cropped_rows = [0, 240]
                cropped_collum = [0, 240]
                if(crop):
                    cropped_rows = [30, 222]
                    cropped_collum = [21, 213]
                mri_scan = os.path.join(path_to_patient_mri_scan, mha_file)
                if('OT' in brain_scan_folder):
                    labels.append(__get_labels__(mri_scan)[slices[0] : slices[1],
                                  cropped_rows[0] : cropped_rows[1],
                                  cropped_collum[0] : cropped_collum[1]])
                else:
                    features.append(__get_features__(mri_scan, filter)[slices[0] : slices[1],
                                    cropped_rows[0] : cropped_rows[1],
                                    cropped_collum[0] : cropped_collum[1]])
    random_indices = __generate_random_indices__(len(labels))
    training_up_limit = int(valid_set * len(random_indices))
    train_features = [features[random_indices[x]] for x in range(0,training_up_limit)]
    train_labels = [labels[random_indices[x]] for x in range(0, training_up_limit)]
    test_features = [features[random_indices[x]] for x in range(training_up_limit, len(features))]
    test_labels = [labels[random_indices[x]] for x in range(training_up_limit, len(labels))]
    train = {'features': train_features, 'labels': train_labels}
    test = {'features': test_features, 'labels': test_labels}
    return (train, test)
    

def __necessary_folders__(brain_scan_folders_list):
    return [x for x in brain_scan_folders_list if ('T1c' in x) or ('OT' in x)]

def __get_features__(path_to_mri_scans, filter):
    image = sitk.ReadImage(path_to_mri_scans)
    if(filter):
        image = sitk.CurvatureFlow(image1 = image, timeStep = 0.125, numberOfIterations = 5)
    return sitk.GetArrayFromImage(image)

def __get_labels__(path_to_mri_scans):
    image = sitk.ReadImage(path_to_mri_scans)
    return sitk.GetArrayFromImage(image)

def __generate_random_indices__(size):
    """
    Returns a list of random integers between 0(inclusive) and size(exclusive)

    INPUTS:-
    size: int. Determines the upper limit of the random numbers and size of the list

    OUTPUT:-
    List of integers.
    """
    index = []
    while(len(index) < size):
        num = np.random.randint(0, size)
        if(not num in index):
            index.append(num)
    return index
