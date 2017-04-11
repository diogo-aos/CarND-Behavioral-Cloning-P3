CROP_TOP = 0.35
CROP_BOT = 0.15

normalize = lambda pix: pix / 255.0 - 0.5
x = Lambda(normalize, input_shape=im_shape)(x)  # normalize

# ((crop n pixels on top, crop n on bottom), (crop n on left, crop n on right))
#x = Cropping2D(cropping=((int(CROP_TOP*h), int(CROP_BOT * h)),(0,0)),
#               input_shape=im_shape)(x)  # crop


x = Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu')(x)
x = Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu')(x)
x = Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)

x = Flatten()(x)

x = Dense(1164, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(10, activation='relu')(x)
