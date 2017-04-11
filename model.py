from keras.models import Model
from keras.layers import Flatten, Dense, Input, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
import pickle
import os.path
import sys

from sklearn.model_selection import train_test_split

import get_data

EPOCHS = 10
BATCH_SIZE = 128
DEVIATION = 5.0  # in degrees, float
CROP_TOP = 0.35
CROP_BOT = 0.15
BRIGHTNESS = 0.5
SHADOW = 0.5
FLIP = 0.5

load_all_data = True

if '-gen' in sys.argv:
    load_all_data = False

if '-e' in sys.argv:
    EPOCHS = int(sys.argv[sys.argv.index('-e') + 1])

if '-bs' in sys.argv:
    BATCH_SIZE = int(sys.argv[sys.argv.index('-bs') + 1])

if '-deviation' in sys.argv:
    DEVIATION = float(sys.argv[sys.argv.index('-deviation') + 1])

if '-crop_top' in sys.argv:
    CROP_TOP = float(sys.argv[sys.argv.index('-crop_top') + 1])

if '-crop_bot' in sys.argv:
    CROP_BOT = float(sys.argv[sys.argv.index('-crop_bot') + 1])

if '-brightness' in sys.argv:
    BRIGHTNESS = float(sys.argv[sys.argv.index('-brightness') + 1])

if '-shadow' in sys.argv:
    SHADOW = float(sys.argv[sys.argv.index('-shadow') + 1])

if '-flip' in sys.argv:
    FLIP = float(sys.argv[sys.argv.index('-flip') + 1])

if '-model' not in sys.argv:
    print('must specify a model file')
    sys.exit(0)

model_file = sys.argv[sys.argv.index('-model') +1]
if not os.path.exists(model_file):
    print('model file does not exist')
    sys.exit(0)

print('preparing data...')

train, val, n_samples = get_data.get_data(deviation=DEVIATION,
                                          crop_top=CROP_TOP, crop_bot=CROP_BOT,
                                          random_brightness=BRIGHTNESS,
                                          random_shadow=SHADOW,
                                          random_flip=FLIP)
    
print('defining model...')

#im_shape = (160, 320, 3)
#im_shape = X_train[0].shape
#im_shape = None, None, 3
im_shape = train[0][0].shape  # only works loading all data to memory

# input layer is common to all models
inp = Input(shape=im_shape)
x = inp


# read model code from file
with open(model_file) as mf:
    model_code = mf.read()

# execute code
exec(model_code)

# final layer defined here
x = Dense(1)(x)

# compile model
model = Model([inp], [x])
model.compile(loss='mse', optimizer='adam')

# save model after each epoch
model_fn = 'model.{epoch:02d}-{val_loss:.2f}.h5'
cb = ModelCheckpoint(model_fn, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

print('training model...')
while True:
    if not load_all_data:
        model.fit_generator(train, samples_per_epoch=,
                            validation_data=validation,
                            nb_val_samples=len(validation_samples),
                            nb_epoch=EPOCHS, callbacks=[cb])
    else:
        X_train, y_train = train
        X_val, y_val = val
        model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val),
                  shuffle=True, callbacks=[cb])
    epoch_io = input('how many epochs to train more (enter anything that is not an int to exit):')
    EPOCHS = int(epoch_io)

