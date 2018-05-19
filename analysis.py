import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns


import sys
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import regularizers
from datetime import datetime
histories = []

main_path = 'ff512_2_2'
histories.append(pickle.load(open(main_path + '.pkl', 'rb')))
#histories.append(pickle.load(open('f_010_4_5123_cont.pkl', 'rb')))
#histories.append(pickle.load(open('f_010_4_5123_cont2.pkl', 'rb')))
#histories.append(pickle.load(open('f_010_4_5123_cont3.pkl', 'rb')))
#histories.append(pickle.load(open('f_010_4_5123_cont4.pkl', 'rb')))
#histories.append(pickle.load(open('f_010_4_5123_cont5.pkl', 'rb')))

img_width = 250
img_height = 325
model = models.load_model(main_path + '.h5')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory('chest_xray/val', target_size=(img_width, img_height), batch_size=1,
                                                        class_mode='binary', color_mode='grayscale')


test_loss, test_acc = model.evaluate_generator(test_generator, steps=116)
print (model.summary())
print(model.optimizer.get_config())
print('test acc:', test_acc)


acc = histories[0]['acc']
val_acc = histories[0]['val_acc']
loss = histories[0]['loss']
val_loss = histories[0]['val_loss']

for i in range(1, len(histories)):
    acc += histories[i]['acc']
    val_acc += histories[i]['val_acc']
    loss += histories[i]['loss']
    val_loss += histories[i]['val_loss']


epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc', color='green')
plt.plot(epochs, val_acc, 'b', label='Validation acc', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss', color='green')
plt.plot(epochs, val_loss, 'b', label='Validation loss', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



test_path = './chest_xray/val/PNEUMONIA/person405_bacteria_1817.jpeg'
train_path = './chest_xray/val/NORMAL/IM-0487-0001.jpeg'

img = image.load_img(train_path,  target_size=(img_width, img_height), grayscale=True)

img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor /= 255.0
print(model.predict(img_tensor, 1))
print(img_tensor.shape)
plt.figure()
np.broadcast_to(img_tensor[0], (img_tensor[0].shape[0], img_tensor[0].shape[1], img_tensor[0].shape[2], img_tensor[0].shape[3]+2))
plt.grid(False)
plt.imshow(img_tensor[0])
plt.show()
model.summary()
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs = layer_outputs)

activations = activation_model.predict(img_tensor)

print(model.predict(img_tensor))

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
images_per_row = 8
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    #print(layer_activation.shape)
    width = layer_activation.shape[1]
    height = layer_activation.shape[2]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((width * n_cols, images_per_row * height))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :,col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std() + 1e-4
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * width : (col + 1) * width, row * height : (row + 1) * height] = channel_image
    plt.figure(figsize=(1. / width * display_grid.shape[1],
                        1. / height * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()
