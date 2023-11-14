#!/usr/bin/python
# coding:utf8

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.applications.vgg16 import VGG16

# include_top：是否保留顶层的3个全连接网络
# 'imagenet'代表加载预训练权重
model = VGG16(include_top=False, weights='imagenet')
# 加载pre-model的权重
model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

datagen = ImageDataGenerator(rescale=1./255)
# 训练集图像生成器,以文件夹路径为参数,生成经过数据提升/归一化后的数据
train_generator = datagen.flow_from_directory('train',
                                              target_size=(150, 150),
                                              batch_size=32,
                                              class_mode=None,
                                              shuffle=False)
#　验证集图像生成器
validation_generator = datagen.flow_from_directory('validation',
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode=None,
                                                    shuffle=False)

# 得到bottleneck feature
# 使用一个生成器作为数据源预测模型，生成器应返回与test_on_batch的输入数据相同类型的数据
bottleneck_features_train = model.predict_generator(train_generator, steps=200)
print (bottleneck_features_train)
# steps是生成器要返回数据的轮数
# 将得到的特征记录在numpy array里
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
bottleneck_features_validation = model.predict_generator(validation_generator, steps=80)
# 一个epoch有800张图片,验证集
np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)