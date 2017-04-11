
normalize = lambda pix: pix / 255.0 - 0.5
x = Lambda(normalize, input_shape=im_shape)(x)  # normalize
x = Conv2D(filters=6, kernel_size=(5,5), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(filters=16, kernel_size=(5,5), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Flatten()(x)
x = Dense(120)(x)
x = Dense(84)(x)
