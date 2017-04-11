from keras.models import Model
from keras.layers import Flatten, Dense, Input
import pickle


train_fn = 'normal_anticlockwise'
with open(train_fn + '.p', 'rb') as f:
    X_train, y_train = pickle.load(f)

print(X_train.shape)

inp = Input(shape=X_train.shape[1:])

x = Flatten()(inp)
x = Dense(1024)(x)
x = Dense(128)(x)
x = Dense(1)(x)

model = Model([inp], [x])
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=10, validation_split=0.2, shuffle=True)

model.save('model.h5')
