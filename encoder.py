from keras.layers import (Layer,
                              BatchNormalization,
                              Conv2D,
                              Dropout,
                              MaxPooling2D)

class Encoder(Layer):
    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate
        self.pooling = pooling

        self.bn = BatchNormalization()
        self.c1 = Conv2D(filters,
                         kernel_size=3,
                         padding='same',
                         activation='relu',
                         kernel_initializer='he_normal')
        self.drop = Dropout(rate)
        self.c2 = Conv2D(filters,
                         kernel_size=3,
                         padding='same',
                         activation='relu',
                         kernel_initializer='he_normal')
        self.pool = MaxPooling2D()

    def call(self, x):
        x = self.bn(x)
        x = self.c1(x)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y,x
        else:
            return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "rate": self.rate,
            "pooling": self.pooling
        }