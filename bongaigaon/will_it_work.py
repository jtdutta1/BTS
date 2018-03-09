import os
import pickle
import numpy as np
import SimpleITK as st

path = 'HGG'

Flair_features = []
T1_features = []
T1c_features = []
T2_features = []
OT_labels = []

list_dir = os.listdir(path)

for filename in list_dir:
    path_HGG = os.path.join(path, filename)
    listAll = os.listdir(path_HGG)

    Flair_path = os.path.join(path_HGG, listAll[0])
    T1_path = os.path.join(path_HGG, listAll[1])
    T1c_path = os.path.join(path_HGG, listAll[2])
    T2_path = os.path.join(path_HGG, listAll[3])
    OT_path = os.path.join(path_HGG, listAll[4])
    
    print(path_HGG)

    imgFlair = st.ReadImage(Flair_path)
    imgT1 = st.ReadImage(T1_path)
    imgT1c = st.ReadImage(T1c_path)
    imgT2 = st.ReadImage(T2_path)
    imgOT = st.ReadImage(OT_path)
    
    imgFlairSmooth = st.CurvatureFlow(image1=imgFlair,
                                          timeStep=0.125,                                       
                                          numberOfIterations=5)
    imgT1Smooth = st.CurvatureFlow(image1=imgT1,
                                          timeStep=0.125,                                       
                                          numberOfIterations=5)
    imgT1cSmooth = st.CurvatureFlow(image1=imgT1c,
                                          timeStep=0.125,                                       
                                          numberOfIterations=5)
    imgT2Smooth = st.CurvatureFlow(image1=imgT2,
                                          timeStep=0.125,                                       
                                          numberOfIterations=5)
    imgOTSmooth = st.CurvatureFlow(image1=imgOT,
                                          timeStep=0.125,                                       
                                          numberOfIterations=5)
    
    Flair_nda = st.GetArrayFromImage(imgFlairSmooth[:,:,:])
    T1_nda = st.GetArrayFromImage(imgT1Smooth[:,:,:])
    T1c_nda = st.GetArrayFromImage(imgT1cSmooth[:,:,:])
    T2_nda = st.GetArrayFromImage(imgT2Smooth[:,:,:])
    OT_nda = st.GetArrayFromImage(imgOTSmooth[:,:,:])
    
    Flair_features.append(Flair_nda)
    T1_features.append(T1_nda)
    T1c_features.append(T1c_nda)
    T2_features.append(T2_nda)
    OT_labels.append(OT_nda)

Flair_train = np.array(Flair_features)
T1_train = np.array(T1_features)
T1c_train = np.array(T1c_features)
T2_train = np.array(T2_features)
OT_train = np.array(OT_labels)

dataset = {'features[0]' : Flair_train,
           'features[1]' : T1_train,
           'features[2]' : T1c_train,
           'features[3]' : T2_train,
           'labels' : OT_train}

with open('brats_dataset.pickle','wb') as data:
    pickle.dump(dataset, data, protocol=pickle.HIGHEST_PROTOCOL)