# 20181027 Dog Breed Prediction (Projet 7)

# Here the pic tested is n02116738-African_hunting_dog/n02116738_2083.jpg

# Put in the "test" pictures folder :
# sauvegarde_0.h5 (Resnet50),
# sauvegarde_82.h5 (Dense120),
# class_frame.csv (labels),
# & this .py

import numpy as np
import pandas as pd
import h5py
import xml.etree.ElementTree as ET
from PIL import Image
import cv2

from keras import models
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array


# Focus on the dog using Annotation
def reshape(img, dirs, name_tmp):
    name_tronc = name_tmp[:-4]
    name_annot = "../Annotation/" + dirs +"/" + name_tronc  
    
    annot = ET.parse(name_annot)
    root = annot.getroot()

    xmin = root[5][4][0].text
    xmin = int(xmin)
    ymin = root[5][4][1].text
    ymin = int(ymin)
    xmax = root[5][4][2].text
    xmax = int(xmax)
    ymax = root[5][4][3].text
    ymax = int(ymax)

    crop_rectangle = (xmin, ymin, xmax, ymax)

    cropped_im = img.crop(crop_rectangle)
    return cropped_im


# CNN creation with pretrained weight from Resnet50 + Transfer Learning
model0_test = models.load_model('sauvegarde_0.h5')
model_test = models.load_model('sauvegarde_82.h5')

# class labels
class_frame = pd.read_csv('class_frame.csv')

#preprocess
name_file = 'n02116738-African_hunting_dog/n02116738_2083.jpg'

array = name_file.split("/")
i = 0
for word in array:
    if i == 0:
        dirs = word
    if i == 1:
        name_tmp = word
    i = i + 1
    
img = image.load_img(name_file, target_size=(224, 224))
img = img_to_array(img)
cv2.imwrite("image_tmp.jpg", img)
image_focus_tmp = Image.open("image_tmp.jpg")
img = reshape(image_focus_tmp, dirs, name_tmp)
img = img_to_array(img)
img = cv2.resize(img, dsize=(224, 224))
img = img_to_array(img)
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  
img = preprocess_input(img)

# feaures + prediction
features = model0_test.predict(img)
outcome = model_test.predict(features)
race_pred = outcome.argmax()

# find label
name = class_frame['class'].loc[race_pred]
name = name[10:]
print("Predicted dog  breed:", name)
