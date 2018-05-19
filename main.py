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

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory('chest_xray_aug/train', color_mode='grayscale',
                                                    target_size=(img_width, img_height), batch_size=64, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('chest_xray_aug/test', target_size=(img_width, img_height), batch_size=64,
                                                        class_mode='binary', color_mode='grayscale')

kernel_sizes = 3
dense_count = 512
learning_rate = 0.00001
suffix = '1'

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
model.add(layers.Dense(dense_count, activation='relu', kernel_regularizer=regularizers.l2(0.010)))
model.add(layers.Dense(dense_count, activation='relu', kernel_regularizer=regularizers.l2(0.010)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate), metrics=['acc'])
model.summary()

history = model.fit_generator(train_generator, steps_per_epoch=160, epochs=50, validation_data=validation_generator,
                              validation_steps=10, verbose=1)
print('Saving...')
with open('ff'+ str(dense_count) + '_' + suffix + '.pkl', 'wb') as f:
    pickle.dump(history.history, f)
model.save('ff'+ str(dense_count) +'_' + suffix + '.h5')




