import os
import numpy as np
import re
from PIL import Image

def get_images(directory):
    files = os.listdir(directory)
    im_list = []
    im_labels = []
    for i, f in enumerate(files):
        print("Processing file ", (i + 1), "/", len(files), ": ", f)
        if f[0] == '.':
            continue
        im = Image.open(directory + '/' + f)
        im_np = np.asarray(im.resize((1000, 1300), Image.NEAREST), dtype=np.float32)
        im_np /= 255.0
        if re.search('bacteria', f): # bacterial pneumonia
            im_labels.append(np.array((0., 1., 0.)))
        elif re.search('virus', f): # viral pneumonia
            im_labels.append(np.array((0., 0., 1.)))
        else: # normal
            im_labels.append(np.array((1., 0., 0.)))
        im_list.append(im_np)
    return im_list, im_labels



for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

import os
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory('./chest_xray/train', color_mode='grayscale',
                                                    target_size=(100, 130), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('./chest_xray/test', target_size=(100, 130), batch_size=32,
                                                        class_mode='binary', color_mode='grayscale')

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 130, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
model.summary()



history = model.fit_generator(train_generator, steps_per_epoch=159, epochs=30, validation_data=validation_generator,
                              validation_steps=19, verbose=1)