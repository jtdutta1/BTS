import os
import pickle
import numpy as np
import SimpleITK as st

path = 'mha_files'

features = []
labels = []
list_dir = os.listdir(path)
len_list_dir = len(list_dir)

for filename in list_dir:
    path_1 = os.path.join(path, filename)
    l1 = os.listdir(path_1)
    img_feature_path, img_label_path = os.path.join(path_1, l1[0]), os.path.join(path_1, l1[1])
    
    print(img_feature_path)
    
    imgT1c = st.ReadImage(img_feature_path)
    imgOT = st.ReadImage(img_label_path)
    
    imgT1cSmooth = st.CurvatureFlow(image1=imgT1c,
                                          timeStep=0.125,                                       
                                          numberOfIterations=5)
    imgOTSmooth = st.CurvatureFlow(image1=imgOT,
                                          timeStep=0.125,                                       
                                          numberOfIterations=5)
    
    T1c_nda = st.GetArrayFromImage(imgT1cSmooth[:,:,:])
    OT_nda = st.GetArrayFromImage(imgOTSmooth[:,:,:])
    
    features.append(T1c_nda)
    labels.append(OT_nda)

features_train = np.array(features)
labels_train = np.array(labels)

dataset = {'features' : features_train, 'labels' : labels_train}

with open('brats_dataset.pickle','wb') as foo:
    pickle.dump(dataset, foo, protocol=pickle.HIGHEST_PROTOCOL)