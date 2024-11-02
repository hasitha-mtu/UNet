from keras.layers import (Layer,
                              BatchNormalization,
                              Conv2DTranspose,
                              concatenate)

from encoder import Encoder


class Decoder(Layer):
    def __init__(self, filters, rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate

        self.bn = BatchNormalization()
        self.cT = Conv2DTranspose(filters,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer='he_normal')
        self.net = Encoder(filters, rate, pooling=False)

    def call(self, X):
        x, skip_x = X
        x = self.bn(x)
        x = self.cT(x)
        x = concatenate([x, skip_x])
        x = self.net(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filter": self.filters,
            "rate": self.rate
        }