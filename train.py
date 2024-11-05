import keras.callbacks_v1
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import (Callback,
                             ModelCheckpoint,
                             CSVLogger)
from numpy.random import randint

from data import load_data
from data_viz import show_image
from model import get_model1 as model11
from model import get_model2 as model12
from model2 import get_model as model2
from unet_model import get_model as unet

class LossHistory(Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []
    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))
    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses, label="Training loss for each batch")
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plot_at_epoch_{epoch}")
        self.per_batch_losses = []

def train_model(path):
    (X_train, y_train), (X_val, y_val) = load_data(path)
    print(f'X_train shape : {X_train.shape}')
    print(f'y_train shape : {y_train.shape}')

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

    tensorboard = keras.callbacks_v1.TensorBoard(
        log_dir="logs/"
    )
    cbs = [
        CSVLogger('unet_logs.csv', separator=',', append=False),
        ModelCheckpoint("UNet-WaterBodySegmentation.keras", save_best_only=True),
        # ShowProgress(),
        LossHistory(),
        tensorboard
    ]

    model = model12()
    print('model :', model)

    print(f'Model information: {model.summary()}')

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=cbs
    )


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    train_model("./input/water_segmentation_dataset/water_v1/JPEGImages/ADE20K")
    # with tf.device('/CPU:0'):
    #     train_model("./input/water_segmentation_dataset/water_v1/JPEGImages/ADE20K")


