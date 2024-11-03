from keras.layers import (Input,
                          Conv2D)
from keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model

from decoder import Decoder
from encoder import Encoder


def get_model():
    unet_inputs = Input(shape=(256, 256, 3), name="UNetInput")
    # Encoder Network : Down sampling phase
    p1, c1 = Encoder(64, 0.1, name="Encoder1")(unet_inputs)
    p2, c2 = Encoder(128, 0.1, name="Encoder2")(p1)
    p3, c3 = Encoder(256, 0.2, name="Encoder3")(p2)
    p4, c4 = Encoder(512, 0.2, name="Encoder4")(p3)

    # Encoding layer : Latent representation
    e = Encoder(512, 0.3, pooling=False)(p4)

    # Decoder Network : Up sampling phase
    d1 = Decoder(512, 0.2, name="Decoder1")([e, c4])
    d2 = Decoder(256, 0.2, name="Decoder2")([d1, c3])
    d3 = Decoder(128, 0.1, name="Decoder3")([d2, c2])
    d4 = Decoder(64, 0.1, name="Decoder4")([d3, c1])

    # Output
    unet_out = Conv2D(3,
                      kernel_size = 3,
                      padding='same',
                      activation='sigmoid',
                      )(d4)

    print(f" unet_out : {unet_out} type of unet_out is {type(unet_out)}")

    # Model
    UNet = Model(
        inputs = unet_inputs,
        outputs = unet_out,
        name = "AttentionUNet"
    )

    print(f" UNet : {UNet} type of UNet is {type(UNet)}")

    UNet.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return UNet

def visualize_model(model):
    plot_model(model, "UNet-WaterBody.png", show_shapes=True)

if __name__ == "__main__":
    model = get_model(),
    print("model : ",model.summary())
