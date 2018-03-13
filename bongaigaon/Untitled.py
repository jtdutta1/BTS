import cv2
import keras
import numpy as np
from keras.utils import *
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras import backend as K
import matplotlib.pyplot as plt

json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("weights.h5")
print("Loaded model from disk")

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

model.compile(optimizer=Adam(lr=1e-4),
              loss=dice_coef_loss,
              metrics=[dice_coef])

img = cv2.imread('2D_images_train/0_1.png')
