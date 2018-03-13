
# coding: utf-8

# In[1]:


import cv2
import keras
import numpy as np
from keras.utils import *
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras import backend as K


# In[ ]:


x_train_path = './2D_images_train/'
x_test_path = './2D_images_test/'


# In[3]:


x_train = []
x_test = []
y_train = []
y_test = []
for i in range(200):
    for j in range(1,52):
        x_train_img_path = x_train_path+str(i)+'_'+str(j)+'.png'
        x_train_img = cv2.imread(x_train_img_path)
        x_train.append(x_train_img)

        y_train_img_path = x_train_path+str(i)+'_'+str(j)+'.png'
        y_train_img = cv2.imread(y_train_img_path)
        y_train.append(y_train_img)
    
for i in range(200,220):
    for j in range(1,52):
        x_test_img_path = x_test_path+str(i)+'_'+str(j)+'.png'
        x_test_img = cv2.imread(x_test_img_path)
        x_test.append(x_test_img)

        y_test_img_path = x_test_path+str(i)+'_'+str(j)+'.png'
        y_test_img = cv2.imread(y_test_img_path)
        y_test.append(y_test_img)
    
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[ ]:


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


# In[ ]:


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(240, 240, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Dropout(0.5))

model.add(Conv2DTranspose(256, (3, 3)))
model.add(UpSampling2D(size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2DTranspose(128, (3, 3)))
model.add(UpSampling2D(size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2DTranspose(64, (3, 3)))
model.add(UpSampling2D(size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2DTranspose(32, (3, 3)))
model.add(UpSampling2D(size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2DTranspose(16, (3, 3)))
model.add(Conv2DTranspose(8, (3, 3)))
model.add(Conv2D(3, (1, 1)))

model.compile(optimizer=Adam(lr=1e-4),
              loss=dice_coef_loss,
              metrics=[dice_coef])

checkpointer = ModelCheckpoint(filepath='/tmp/weights.h5', verbose=1, save_best_only=True)


# In[ ]:

try:
	model.fit(x_train, y_train, batch_size=2, epochs=50, callbacks=[checkpointer])
	score = model.evaluate(x_test, y_test, batch_size=2)
finally:
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	model.save_weights("weights.h5")
	print("Saved model to disk")