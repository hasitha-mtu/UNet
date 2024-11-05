from keras import Model
import keras.utils
from keras.layers import (Input,
                          Conv2D,
                          MaxPooling2D,
                          Conv2DTranspose,
                          concatenate)


def get_model():
    inputs = Input(shape=(256, 256, 3), name="UnetModelInput")
    c1 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(inputs)
    c2 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(p1)
    c4 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(c3)
    p2 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(p2)
    c6 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(c5)
    p3 = MaxPooling2D((2, 2))(c6)

    c7 = Conv2D(512, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(p3)
    c8 = Conv2D(512, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(c7)
    p4 = MaxPooling2D((2, 2))(c8)

    c9 = Conv2D(512, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(p4)
    c10 = Conv2D(1024, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(c9)

    u1 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(c10)
    u1 = concatenate([u1, c8])
    c11 = Conv2D(512, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(u1)
    c12 = Conv2D(512, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(c11)

    u2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c12)
    u2 = concatenate([u2, c6])
    c13 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(u2)
    c14 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(c13)

    u3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c14)
    u3 = concatenate([u3, c4])
    c15 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(u3)
    c16 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(c15)

    u4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c16)
    u4 = concatenate([u4, c2])
    c17 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(u4)
    c18 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal', padding='same')(c17)

    outputs = Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')(c18)

    model = Model(inputs=[inputs], outputs=[outputs], name="AttentionUNet")

    # NOTE: Compile the model in the main program to make it easy to test with various loss functions
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(f"Model information : {model.summary()}")
    # keras.utils.plot_model(model, "unet_model3.png", show_shapes=True)

    return model

if __name__ == "__main__":
    get_model()