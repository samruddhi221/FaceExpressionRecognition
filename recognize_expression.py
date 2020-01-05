import face_recognition
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
import os
from keras.layers.advanced_activations import ELU
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import regularizers
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
import cv2
import argparse

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


parser = argparse.ArgumentParser(description='Face emotion recognition')

parser.add_argument("--image", dest="image_path", required=False, help="transfer learn from some_model.pth", metavar="MODEL NAME", default="test_images/sameya.JPG")

args = parser.parse_args()

image = face_recognition.load_image_file(args.image_path)

(h,w,c) = image.shape

aspect_ratio = float(w)/float(h)

image = cv2.resize(image, dsize=(1024,int(1024/aspect_ratio) ), interpolation=cv2.INTER_CUBIC) 
face_locations = face_recognition.face_locations(image)

print(face_locations)
plt.imshow(image)
plt.show()

def baseline_model(drop_rate=0.1, regularization_val=0.0001):
  # Create the model
  model = Sequential()

  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                   kernel_regularizer=regularizers.l2(regularization_val), 
                   padding='same',
                   input_shape=(48,48,1)))
  # model.add(BatchNormalization())

  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
                   kernel_regularizer=regularizers.l2(regularization_val), 
                   padding='same') )
  # model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', 
                   kernel_regularizer=regularizers.l2(regularization_val), 
                   padding='same'))
  # model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', 
                   kernel_regularizer=regularizers.l2(regularization_val), 
                   padding='same'))
  # model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))

  # model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', 
  #                  kernel_regularizer=regularizers.l2(regularization_val)))
  # model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', 
  #                  kernel_regularizer=regularizers.l2(regularization_val)))
  # model.add(MaxPooling2D(pool_size=(2, 2)))


  model.add(Conv2D(7, kernel_size=(1, 1), activation='relu', 
                   kernel_regularizer=regularizers.l2(regularization_val), 
                   padding='same'))
  # model.add(BatchNormalization())

  model.add(Conv2D(7, kernel_size=(4, 4), activation='relu', 
                   kernel_regularizer=regularizers.l2(regularization_val), 
                   padding='same'))
  # model.add(BatchNormalization())

  model.add(Flatten())

  # model.add(Dense(4096, activation='relu'))
  # model.add(Dropout(drop_rate))
  # model.add(Dense(4096, activation='relu'))
  # model.add(Dropout(drop_rate))

  model.add(Dense(7, activation='relu'))

  model.add(Activation("softmax"))

  model.summary()

  return model


model_ = baseline_model(drop_rate=0.0, regularization_val=0.0)
model_.load_weights("trained_model.h5")

EMOTIONS = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

for face_location in face_locations:
    top = face_location[0] #y1
    right = face_location[1] #x2
    bottom = face_location[2] #y2
    left = face_location[3] #x1
        
    face_crop = image[top:bottom, left:right]
    # plt.imshow(face_crop)
    # plt.show() 

    input_image = cv2.resize(face_crop, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image = np.reshape(input_image, [1, input_image.shape[0], input_image.shape[1], 1])
    input_image = input_image.astype(np.float32) * 1.0/255.0
    print(input_image.shape)
    emotion_pred = model_.predict_proba(input_image)
    emotion_id = np.argmax(emotion_pred)
    probability = np.max(emotion_pred)
    print("{} P:{}".format(EMOTIONS[emotion_id],probability))
    

    # Draw a box around the face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(image, (left, bottom - 3), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, EMOTIONS[emotion_id], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

plt.imshow(image)
plt.show()
