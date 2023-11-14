import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
import numpy as np
import time
import os
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

tf.disable_v2_behavior()
path = 'path to the test data'
names = sorted(os.listdir(path))
config = tf.ConfigProto()
sess = tf.Session(config = config)
with gfile.FastGFile('path to the pb model', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name = '')
    opname = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]

#get the input tensor
x = tf.get_default_graph().get_tensor_by_name('name of the input tensor')
print('input: ', x)
#get the predicting tensor
pred = tf.get_default_graph().get_tensor_by_name('name of the output tensor')
print('pred: ', pred)

Acc = 0
for label, allDir in enumerate(names):
    acc = 0
    imgpaths = path + allDir
    imgpath = os.listdir(imgpaths)
    for p in imgpath:
        img = load_img(imgpaths + '/' +p, target_size = (150, 150))
        image = img_to_array(img)
        image = np.expand_dims(image, axis = 0)
        image = preprocess_input(image)
        preds = sess.run(pred, feed_dict = {x: image})
        preds = np.where(preds >= 0.5, 1, 0)
        if preds == label:
            acc += 1
            Acc += 1
    print(label, acc / 50) #the accuracy of one of the categories
    print('Acc: ', Acc / 50)