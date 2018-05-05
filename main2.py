#python main.py | qsub -d /home/u14035/Pneumonia
import os
import numpy as np
import pickle
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from datetime import datetime

img_width = 750
img_height = 1000

'''
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

train_generator = train_datagen.flow_from_directory('chest_xray_3class/train', color_mode='grayscale',
                                                    target_size=(img_width, img_height), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('chest_xray_3class/test', target_size=(img_width, img_height), batch_size=32,
                                                        class_mode='binary', color_mode='grayscale')
'''
model = models.Sequential()
model.add(layers.Conv2D(16, (150, 150), activation='relu', input_shape=(img_width, img_height, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (60, 60), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (30, 30), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (10, 10), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
model.summary()
'''
histories=[]
for i in range(5):
    print()
    print('Starting era ', (i+1), ' at ', str(datetime.now()))
    history = model.fit_generator(train_generator, steps_per_epoch=159, epochs=10, validation_data=validation_generator,
                              validation_steps=19, verbose=1)
    histories.append(history)
    print('Saving...')
    try:
        with open('history_3class_1_'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(history, f)
    except:
        pass
    model.save('3class_1000x1300_augdrop_1_' + str(i) + '.h5')
'''



