import os
import pickle
import numpy as np
import SimpleITK as st

path = 'HGG'
count = 0
features = []
labels = []

list_dir = os.listdir(path)
len_list_dir = len(list_dir)

#filenameT1c = os.path.join(path, '0/VSD.Brain.XX.O.MR_T1c.54514.mha')
for filename in list_dir:
    print('Iteration #'+str(count)+'/#'+str(len_list_dir))
    path_1 = os.path.join(path, filename)
    l1 = os.listdir(path_1)
    img_feature_path, img_label_path = os.path.join(path_1, l1[0]), os.path.join(path_1, l1[1])
    #print(img_feature_path,' ', img_label_path)
    
    imgT1cOriginal = st.ReadImage(img_feature_path)
    imgOT = st.ReadImage(img_label_path)
    nda = st.GetArrayFromImage(imgT1cOriginal[:, :, 50:100])
    #print(nda.shape)
    nda1 = st.GetArrayFromImage(imgOT[:, :, 50:100])
    imgT1cSmooth = st.CurvatureFlow(image1=imgT1cOriginal,
                                          timeStep=0.125,                                       
                                          numberOfIterations=5)
    
    pew = st.GetArrayFromImage(imgT1cSmooth[:,:,88])
    features.append(pew)
    #print(features)
    labels.append(nda1)
    count = count + 1
    
features_train = np.array(features)
labels_train = np.array(labels)
#print(features_train.shape)
#print(labels_train.shape)
dataset = {'features_train' : features_train, 'labels_train' : labels_train}

with open('dataset.pickle','wb') as foo:
    pickle.dump(dataset, foo, protocol = pickle.HIGHEST_PROTOCOL)

    
    

