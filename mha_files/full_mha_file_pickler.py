
# coding: utf-8

# In[1]:


import os
import numpy as np
import pickle
import SimpleITK as st


# In[2]:


path = 'mha_files'
count = 0
features = []
labels = []

list_dir = os.listdir(path)
len_list_dir = len(list_dir)


# In[3]:


for filename in list_dir:
    print('Iteration #'+str(count)+'/#'+str(len_list_dir))
    path_1 = os.path.join(path, filename)
    l1 = os.listdir(path_1)
    img_feature_path, img_label_path = os.path.join(path_1, l1[0]), os.path.join(path_1, l1[1])
    #print(img_feature_path,' ', img_label_path)
    
    imgT1cOriginal = st.ReadImage(img_feature_path)
    imgOT = st.ReadImage(img_label_path)
    nda = st.GetArrayFromImage(imgT1cOriginal)
    #print(nda.shape)
    nda1 = st.GetArrayFromImage(imgOT[:,:,5:149])
    imgT1cSmooth = st.CurvatureFlow(image1=imgT1cOriginal,
                                          timeStep=0.125,                                       
                                          numberOfIterations=5)
    
    pew = st.GetArrayFromImage(imgT1cSmooth[:,:,5:149])
    features.append(pew)
    #print(features)
    labels.append(nda1)
    count = count + 1


# In[ ]:


features = np.array(features)


# In[ ]:


labels = np.array(labels)


# In[ ]:


dataset = {'features' : features, 'labels' : labels}


# In[ ]:


with open('dataset.pickle','wb') as foo:
    pickle.dump(dataset, foo, protocol = pickle.HIGHEST_PROTOCOL)

