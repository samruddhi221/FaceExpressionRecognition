import face_recognition
import matplotlib.pyplot as plt

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder
# from keras.preprocessing.image import ImageDataGenerator

image = face_recognition.load_image_file("friends.jpg")
face_locations = face_recognition.face_locations(image)

print(face_locations)
plt.imshow(image)
plt.show()

for face_location in face_locations:
    top = face_location[0] #y1
    right = face_location[1] #x2
    bottom = face_location[2] #y2
    left = face_location[3] #x1
        
    face_crop = image[top:bottom, left:right]
    plt.imshow(face_crop)
    plt.show() 