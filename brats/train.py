# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 00:07:09 2018

@author: jtdut
"""

import pickle 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

with open('dataset.pickle','rb') as fun:
    dataset = pickle.load(fun)
features, labels = dataset['features_train'], dataset['labels_train']
#features.shape = (features.shape[0], features.shape[1] * features.shape[2])
#labels.shape = (labels.shape[0], labels.shape[1] * labels.shape[2])
#print(features_train.shape)

features_train, labels_train, features_test, labels_test = features[0:15], labels[0:15], features[15:20], labels[15:20]

print(features_train.shape)
print(features_test.shape)
cluster = KMeans(n_clusters = 4)
cluster.fit(features_train[0])
for e in features_train:
    cluster.fit_transform(e)
pred = cluster.predict(features_train[5])
print(pred)
#print(r2_score(labels_train[5],pred))

#plt.scatter(features_train,labels_train, color='b')