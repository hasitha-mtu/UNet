from tensorflow import keras
from keras import Model
from keras.layers import (Input,
                          Conv2D,
                          MaxPooling2D,
                          Conv2DTranspose,
                          concatenate, BatchNormalization, Dropout,
                          RandomFlip, RandomRotation, RandomZoom, Rescaling)


def get_model():
    data_augmentation = keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.2)
    ])
    inputs = Input(shape=(256, 256, 3), name="UnetModelInput")
    a = data_augmentation(inputs)
    b1 = BatchNormalization()(a)
    c1 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(b1)
    d1 = Dropout(0.2)(c1)
    c2 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d1)
    p1 = MaxPooling2D((2, 2))(c2)

    b2 = BatchNormalization()(p1)
    c3 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(b2)
    d2 = Dropout(0.2)(c3)
    c4 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d2)
    p2 = MaxPooling2D((2, 2))(c4)

    b3 = BatchNormalization()(p2)
    c5 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(b3)
    d3 = Dropout(0.2)(c5)
    c6 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d3)
    p3 = MaxPooling2D((2, 2))(c6)

    b4 = BatchNormalization()(p3)
    c7 = Conv2D(512, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(b4)
    d4 = Dropout(0.2)(c7)
    c8 = Conv2D(512, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d4)
    p4 = MaxPooling2D((2, 2))(c8)

    c9 = Conv2D(512, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(p4)
    d5 = Dropout(0.2)(c9)
    c10 = Conv2D(1024, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d5)

    b5 = BatchNormalization()(c10)
    u1 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(b5)
    u1 = concatenate([u1, c8])
    c11 = Conv2D(512, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(u1)
    d6 = Dropout(0.2)(c11)
    c12 = Conv2D(512, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d6)

    b6 = BatchNormalization()(c12)
    u2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(b6)
    u2 = concatenate([u2, c6])
    c13 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(u2)
    d7 = Dropout(0.2)(c13)
    c14 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d7)

    b7 = BatchNormalization()(c14)
    u3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(b7)
    u3 = concatenate([u3, c4])
    c15 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(u3)
    d8 = Dropout(0.2)(c15)
    c16 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d8)

    b8 = BatchNormalization()(c16)
    u4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b8)
    u4 = concatenate([u4, c2])
    c17 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(u4)
    d9 = Dropout(0.2)(c17)
    c18 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d9)

    outputs = Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')(c18)

    model = Model(inputs=[inputs], outputs=[outputs], name="AttentionUNet")

    # NOTE: Compile the model in the main program to make it easy to test with various loss functions
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(f"Model information : {model.summary()}")
    keras.utils.plot_model(model, "unet_model3.png", show_shapes=True)

    return model

def get_model2():
    data_augmentation = keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.2)
    ])
    inputs = Input(shape=(256, 256, 3), name="UnetModelInput")
    a = data_augmentation(inputs)
    b1 = BatchNormalization()(a)
    c1 = Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(b1)
    d1 = Dropout(0.2)(c1)
    c2 = Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d1)
    p1 = MaxPooling2D((2, 2))(c2)

    b2 = BatchNormalization()(p1)
    c3 = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(b2)
    d2 = Dropout(0.2)(c3)
    c4 = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d2)
    p2 = MaxPooling2D((2, 2))(c4)

    b3 = BatchNormalization()(p2)
    c5 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(b3)
    d3 = Dropout(0.2)(c5)
    c6 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d3)
    p3 = MaxPooling2D((2, 2))(c6)

    b4 = BatchNormalization()(p3)
    c7 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(b4)
    d4 = Dropout(0.2)(c7)
    c8 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d4)
    p4 = MaxPooling2D((2, 2))(c8)

    c9 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(p4)
    d5 = Dropout(0.2)(c9)
    c10 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d5)

    b5 = BatchNormalization()(c10)
    u1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(b5)
    u1 = concatenate([u1, c8])
    c11 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(u1)
    d6 = Dropout(0.2)(c11)
    c12 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d6)

    b6 = BatchNormalization()(c12)
    u2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b6)
    u2 = concatenate([u2, c6])
    c13 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(u2)
    d7 = Dropout(0.2)(c13)
    c14 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d7)

    b7 = BatchNormalization()(c14)
    u3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(b7)
    u3 = concatenate([u3, c4])
    c15 = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(u3)
    d8 = Dropout(0.2)(c15)
    c16 = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d8)

    b8 = BatchNormalization()(c16)
    u4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(b8)
    u4 = concatenate([u4, c2])
    c17 = Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(u4)
    d9 = Dropout(0.2)(c17)
    c18 = Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(d9)

    outputs = Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')(c18)

    model = Model(inputs=[inputs], outputs=[outputs], name="AttentionUNet")

    # NOTE: Compile the model in the main program to make it easy to test with various loss functions
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(f"Model information : {model.summary()}")
    keras.utils.plot_model(model, "unet_model4.png", show_shapes=True)

    return model

if __name__ == "__main__":
    get_model2()