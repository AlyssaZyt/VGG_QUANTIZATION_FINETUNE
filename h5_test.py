import os
import numpy as np
from keras import backend as K
from tensorflow.keras.models import load_model
K.set_image_dim_ordering('th')
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

filePath = 'path to the test data'
pathDir = os.listdir(filePath)
img_width = 150
img_height = 150

def pred(img_path):
    Acc = 0
    test_model = load_model(img_path)

    for label, allDir in enumerate(pathDir):
        acc = 0
        imgpaths = filePath + allDir
        imgpath = os.listdir(imgpaths)
        for path in imgpath:
            img = load_img(imgpaths + '/' + path, target_size = (150, 150))
            x = img_to_array(img)
            x = np.expand_dims(x, axis = 0)
            x = preprocess_input(x)
            preds = test_model.predict(x)
            preds = np.where(preds >= 0.5, 1, 0)
            if preds == label:
                acc += 1
                Acc += 1
        print(label, acc / 50) #the accuracy of one of the categories
        print('Acc: ', Acc / 50)