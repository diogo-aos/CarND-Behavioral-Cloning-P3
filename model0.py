from keras.models import Model
from keras.layers import Flatten, Dense, Input, Lambda
import pickle


train_fn = 'normal_anticlockwise'

print('loading data from:', train_fn + '.p')
with open(train_fn + '.p', 'rb') as f:
    X_train, y_train = pickle.load(f)

print('train set shape:', X_train.shape)
im_shape = X_train.shape[1:]

inp = Input(shape=im_shape)
x = inp

normalize = lambda pix: pix / 255.0 - 0.5
x = Lambda(normalize, input_shape=im_shape)(x)  # normalize
x = Flatten()(x)
x = Dense(1)(x)


model = Model([inp], [x])
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=10, validation_split=0.2, shuffle=True)

model.save('model.h5')
