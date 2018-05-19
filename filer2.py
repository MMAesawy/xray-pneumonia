from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import numpy as np
import os

batch = 64

img_width = 250
img_height = 325
train_datagen = ImageDataGenerator(rotation_range=17, zca_whitening=True, width_shift_range=0.12,
                                   height_shift_range=0.12,shear_range=0.15, zoom_range=0.15)
train_generator = train_datagen.flow_from_directory('chest_xray/train/', color_mode='grayscale',
                                                    target_size=(img_width, img_height), batch_size=batch, class_mode='binary')

bs = [x for x in range(0, 100, batch)]
for s in bs:
    (x, y) = train_generator.next()
    for i, im in enumerate(x):
        imag = image.array_to_img(np.concatenate([im,im,im], axis=2))
        path = "chest_xray_aug/train2"
        if y[i] == train_generator.class_indices['NORMAL']:
            path = os.path.join(path, 'NORMAL')
        else:
            path = os.path.join(path, 'PNEUMONIA')
        path = os.path.join(path, str(s+i)+'.png')
        imag.save(path, mode = 'grayscale')

'''
import os
import random
files = os.listdir('chest_xray_aug/train/PNEUMONIA')
random.shuffle(files)
for fname in files[5000:]:
    os.remove(os.path.join('chest_xray_aug/train/PNEUMONIA', fname))
'''



