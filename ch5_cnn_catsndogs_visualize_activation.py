import os

base_dir = '/mac/okehara/dev/keras_teset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')


# from keras import layers
# from keras import models
#
# model = models.Sequential()
# model.add(layers.Conv2D(filters=32,
#                         kernel_size=(3, 3),
#                         activation='relu',
#                         input_shape=(150, 150, 3)))
# model.add(layers.MaxPool2D((2,2)))
# model.add(layers.Conv2D(64,
#                         (3,3),
#                         activation='relu'))
# model.add(layers.MaxPool2D((2,2)))
# model.add(layers.Conv2D(128,
#                         (3,3),
#                         activation='relu'))
# model.add(layers.MaxPool2D((2,2)))
# model.add(layers.Conv2D(128,
#                         (3,3),
#                         activation='relu'))
# model.add(layers.MaxPool2D((2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
#
# print(model.summary())
#
# from keras import optimizers
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['acc'])
# model.save('cats_and_dogs_small_2.h5')

from keras import layers
from keras import models

model = models.load_model('cats_and_dogs_small_2.h5')
print(model.summary())


from keras.preprocessing import image
import numpy as np

img_path = 'test/cats/cat.1700.jpg'
img = image.load_img(img_path,
                     target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)

import matplotlib.pyplot as plt

# plt.imshow(img_tensor[0])
# plt.show()

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
# first_layer_activation = activations[0]
# print(first_layer_activation.shape)
# plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
# plt.show()

# visualize activation
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

image_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):

    print(layer_activation.shape)

    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features//image_per_row
    display_grid = np.zeros((size*n_cols, image_per_row*size))

    for col in range(n_cols):
        for row in range(image_per_row):
            channel_image = layer_activation[0,
                                            :, :,
                                            col*image_per_row+row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col*size:(col+1)*size,
                        row*size:(row+1)*size] = channel_image

    scale = 1./size
    plt.figure(figsize=(scale*display_grid.shape[1],
                        scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
