import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
import os


train_dir = 'C:/Users/mepiy/Downloads/aerial-cactus-identification/Aerial_Cactus_Classification_kaggle/train/train'
test_dir = 'C:/Users/mepiy/Downloads/aerial-cactus-identification/Aerial_Cactus_Classification_kaggle/test/test'
train_df_dir = 'C:/Users/mepiy/Downloads/aerial-cactus-identification/Aerial_Cactus_Classification_kaggle/train.csv'

df = pd.read_csv(train_df_dir)

im = cv2.imread(train_dir +'/'+ str(df['id'][1]))
plt.imshow(im)


vgg16_net = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(32, 32, 3))

vgg16_net.trainable = False
model = Sequential()
model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer= 'Adam', metrics=['accuracy'])

X_tr = []
Y_tr = []
image_id = df['id'].values
for i in image_id:
    X_tr.append(cv2.imread(train_dir +'/'+ str(i)))
    Y_tr.append(df[df['id'] == str(i)]['has_cactus'].values[0])

X_tr = np.array(X_tr)
X_tr = X_tr.astype('float32')
X_tr /= 255
Y_tr = np.asarray(Y_tr)

batch_size = 32
nb_epoch = 10

history = model.fit(X_tr, Y_tr,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=0.1,
              shuffle=True,
              verbose=2)