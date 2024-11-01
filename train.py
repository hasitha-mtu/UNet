from math import floor

from numpy.random import randint
from keras.src.callbacks import (Callback,
                                 ModelCheckpoint,
                                 CSVLogger)
import tensorflow as tf
from data import load_data
import matplotlib.pyplot as plt

from data_viz import show_image
from model import get_model


def train_model(path):
    (X_train, y_train), (X_val, y_val) = load_data(path)

    class ShowProgress(Callback):
        def on_epoch_end(self, epoch, logs=None):
            id = randint(len(X_val))
            image = X_val[id]
            mask = y_val[id]
            pred_mask = self.model(tf.expand_dims(image, axis=0))[0]

            plt.figure(figsize=(10, 8))
            plt.subplot(1, 3, 1)
            show_image(image, title="Original Image")

            plt.subplot(1, 3, 2)
            show_image(mask, title="Original Mask")

            plt.subplot(1, 3, 3)
            show_image(pred_mask, title="Predicted Mask")

            plt.tight_layout()
            plt.show()

    cbs = [
        CSVLogger('unet_logs.csv', separator=',', append=False),
        ModelCheckpoint("UNet-WaterBodySegmentation.keras", save_best_only=True),
        ShowProgress()
    ]

    model = get_model()
    print('model :', model)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.optimizers.Adam()
    )

    print(f'Model information: {model.summary()}')

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=cbs
    )

    for i in range(20):
        id = randint(len(X_val))
        image = X_val[id]
        mask = y_val[id]
        pred_mask = model.predict(tf.expand_dims(image, axis=0))[0]
        post_process = (pred_mask[:, :, 0] > 0.5).astype('int')

        plt.figure(figsize=(10, 8))
        plt.subplot(1, 4, 1)
        show_image(image, title="Original Image")

        plt.subplot(1, 4, 2)
        show_image(mask, title="Original Mask")

        plt.subplot(1, 4, 3)
        show_image(pred_mask, title="Predicted Mask")

        plt.subplot(1, 4, 4)
        show_image(post_process, title="Post=Processed Mask")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    train_model("./input/water_segmentation_dataset/water_v1/JPEGImages/ADE20K")
