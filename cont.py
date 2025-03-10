
import os
import numpy as np
import pickle
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
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

model = models.load_model('ff512_2.h5')

history = model.fit_generator(train_generator, steps_per_epoch=160, epochs=50, validation_data=validation_generator,
                              validation_steps=10, verbose=1)
model.save('ff512_2_2.h5')
with open('ff512_2_2.pkl', 'wb') as f:
    pickle.dump(history.history, f)

