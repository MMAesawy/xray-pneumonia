#python main.py | qsub -d /home/u14035/Pneumonia
import sys
import os
import numpy as np
import pickle
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import regularizers
from datetime import datetime

img_width = 250
img_height = 325

from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto(intra_op_parallelism_threads=64, inter_op_parallelism_threads=2, allow_soft_placement=True,  device_count = {'CPU': 64})
session = tf.Session(config=config)
K.set_session(session)

os.environ["OMP_NUM_THREADS"] = "64"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=25, width_shift_range=0.25,
                                   height_shift_range=0.25,shear_range=0.25, zoom_range=0.15)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory('chest_xray/train', color_mode='grayscale',
                                                    target_size=(img_width, img_height), batch_size=64, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('chest_xray/test', target_size=(img_width, img_height), batch_size=64,
                                                        class_mode='binary', color_mode='grayscale')

kernel_sizes = 3

model = models.Sequential()
model.add(layers.Conv2D(8, (kernel_sizes, kernel_sizes), activation='relu', input_shape=(img_width, img_height, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (kernel_sizes, kernel_sizes), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (kernel_sizes, kernel_sizes), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (kernel_sizes, kernel_sizes), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(16, (kernel_sizes, kernel_sizes), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(16, (2, 2), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.010)))
model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.010)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(0.0001), metrics=['acc'])
model.summary()

history = model.fit_generator(train_generator, steps_per_epoch=200, epochs=30, validation_data=validation_generator,
                              validation_steps=10, verbose=1)
print('Saving...')
with open('f_010_4_512' + str(kernel_sizes) +'.pkl', 'wb') as f:
    pickle.dump(history.history, f)
model.save('f_010_4_512' + str(kernel_sizes) + '.h5')




