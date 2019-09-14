from keras import Input, layers
from keras.models import Model

input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor, output_tensor)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
print(model.summary())

import numpy as np
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

from IPython import embed; embed()

model.fit(x_train, y_train, epochs=10, batch_size=128)
score = model.evaluate(x_train, y_train)
