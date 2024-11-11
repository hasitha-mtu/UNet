import os
from datetime import datetime

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
from model import get_model3 as model13
from model2 import get_model1 as model21
from model2 import get_model2 as model22
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
        plt.savefig(f"plots/plot_at_epoch_{epoch}")
        self.per_batch_losses = []

def train_model(path):
    (X_train, y_train), (X_val, y_val) = load_data(path)
    print(f'X_train shape : {X_train.shape}')
    print(f'y_train shape : {y_train.shape}')

    tensorboard = keras.callbacks.TensorBoard(
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        histogram_freq=1
    )
    cbs = [
        CSVLogger('logs/unet_logs.csv', separator=',', append=False),
        ModelCheckpoint("ckpt/ckpt-{epoch}", save_freq="epoch"),
        LossHistory(),
        tensorboard
    ]
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = make_or_restore_model()

    history = model.fit(
                    X_train,
                    y_train,
                    epochs=50,
                    batch_size=16,
                    validation_data=(X_val, y_val),
                    callbacks=cbs
                )

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()
    return None

def make_or_restore_model():
    checkpoints = ["ckpt/" + name for name in os.listdir("ckpt")]
    print(f"Checkpoints: {checkpoints}")
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Restoring from {latest_checkpoint}")
        return keras.models.load_model(latest_checkpoint)
    else:
        print("Creating fresh model")
        return model11()

# if __name__ == "__main__":
#     print(tf.config.list_physical_devices('GPU'))
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     if len(physical_devices) > 0:
#         tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)
#         train_model("./input/water_segmentation_dataset/water_v1/JPEGImages/ADE20K")

# if __name__ == "__main__":
#     with tf.device('/CPU:0'):
#         train_model("./input/water_segmentation_dataset/water_v1/JPEGImages/ADE20K")

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        train_model("./input/water_segmentation_dataset/water_v1/JPEGImages/ADE20K")