import os, shutil

org_data_path = '/mac/okehara/dev/keras_teset/org/train'
base_dir = '/mac/okehara/dev/keras_teset'

train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)

val_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(val_cats_dir)
val_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(val_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)

# fnames = [f'cat.{i}.jpg' for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(org_data_path, fname)
#     dst = os.path.join(train_cats_dir, fname)
#     shutil.copy(src, dst)
#
# fnames = [f'cat.{i}.jpg' for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(org_data_path, fname)
#     dst = os.path.join(val_cats_dir, fname)
#     shutil.copy(src, dst)
#
# fnames = [f'cat.{i}.jpg' for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(org_data_path, fname)
#     dst = os.path.join(test_cats_dir, fname)
#     shutil.copy(src, dst)
#
# fnames = [f'dog.{i}.jpg' for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(org_data_path, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copy(src, dst)
#
# fnames = [f'dog.{i}.jpg' for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(org_data_path, fname)
#     dst = os.path.join(val_dogs_dir, fname)
#     shutil.copy(src, dst)

fnames = [f'dog.{i}.jpg' for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(org_data_path, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copy(src, dst)

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(filters=32,
                        kernel_size=(3, 3),
                        activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,
                        (3,3),
                        activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,
                        (3,3),
                        activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,
                        (3,3),
                        activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
val_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# for data_batch, label_batch in train_generator:
#     print('data batch shape : ', data_batch.shape)
#     print('label batch shape : ', label_batch.shape)
#     break

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=val_generator,
    validation_steps=50
)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.figure()
plt.plot(epochs, acc, 'bo', label='Training ACC')
plt.plot(epochs, val_acc, 'b', label='Val ACC')
plt.title('Training and Validation ACC')
plt.legend()

plt.show()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Val loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()

model.save('cats_and_dogs_small_1.h5')
