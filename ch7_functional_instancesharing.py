from keras import layers
from keras import applications
from keras import Input
from keras.models import Model

xception_base = applications.Xception(weights=None,
                                      include_top=False)
left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))

left_features = xception_base(left_input)
right_features = xception_base(right_input)

merged_features = layers.concatenate([left_features,
                                      right_features], axis=-1)

output = layers.Dense(1)(merged_features)

model = Model([left_input, right_input], output)

print(model.summary())
