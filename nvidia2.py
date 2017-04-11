from keras.backend import tf as ktf
import numpy as np

h, w, c = im_shape
CROP_TOP = 0.35
CROP_BOT = 0.15
BRIGHTNESS_DELTA = 30
CONTRAST_LOWER, CONTRAST_UPPER = 0, 30

#croped_shape = (h - int(CROP_BOT * h)) - int(CROP_TOP*h), w, c
croped_shape = None, None, 3


def convert_yuv(im):
    for i, x in enumerate(im):
        im[i,...] = cv2.cvtColor(x, cv2.COLOR_BGR2YUV)
    return im


def random_apply(prob, func, *args, **kwargs):
    def wraped(im):
        if np.random.uniform() > prob:
            return func(im, *args, **kwargs)
        else:
            return im
    return wraped


normalize = lambda pix: pix / 255.0 - 0.5

# ((crop n pixels on top, crop n on bottom), (crop n on left, crop n on right))
#x = Cropping2D(cropping=((int(CROP_TOP*h), int(CROP_BOT * h)),(0,0)),
#               input_shape=im_shape)(x)  # crop
#x = Cropping2D(cropping=((int(CROP_TOP*h), int(CROP_BOT * h)), (0,0)))(x)  # crop

#x = Lambda(convert_yuv)(x)  # normalize

# random transformations
# x = Lambda(lambda im: ktf.image.rgb_to_hsv(im))(x)
# x = Lambda(lambda im: ktf.image.random_brightness(im, BRIGHTNESS_DELTA))(x)
# #x = Lambda(lambda im: ktf.image.random_contrast(im, CONTRAST_DELTA))(x)
# x = Lambda(lambda im: ktf.image.hsv_to_rgb(im))(x)
# x = Lambda(lambda im: ktf.image.random_flip_left_right(im))(x)

# x = Lambda(random_apply(0.3, ktf.image.random_brightness, BRIGHTNESS_DELTA), (None, None, 3))(x)
# x = Lambda(random_apply(0.3, ktf.image.random_contrast, CONTRAST_LOWER, CONTRAST_UPPER))(x)
# x = Lambda(random_apply(0.5, ktf.image.flip_left_right))(x)
x = Lambda(normalize)(x)  # normalize

x = Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu')(x)
x = Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu')(x)
x = Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation='relu')(x)
x = Dropout(0.5)(x)
x = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)

x = Flatten()(x)

x = Dense(1164, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(100, activation='relu')(x)
x = Dense(10, activation='relu')(x)
