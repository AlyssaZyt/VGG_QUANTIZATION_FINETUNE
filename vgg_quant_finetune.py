from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_model_optimization as tfmot

model_path = 'path to the h5 model'
model = load_model(model_path)
quant_model = tfmot.quantization.keras.quantize_model(model)

quant_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'],
              run_eagerly = True)

#training with low learning rates
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('train',
                                                    target_size=(150,150),
                                                    batch_size=16,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory('validation',
                                                        target_size=(150,150),
                                                        batch_size=16,
                                                        class_mode='binary')

quant_model.fit_generator(train_generator,
                    steps_per_epoch=10,
                    epochs=50,
                    validation_data=validation_generator,
                    validation_steps=10)

quant_model.save('path to the quantized h5 model')