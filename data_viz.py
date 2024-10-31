import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops.np_random import random

from data import load_data
from numpy.random import randint

def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

def show_image_set(images, masks):
    for (image, mask) in zip(images, masks):
        plt.figure(figsize=(len(images),8))
        plt.subplot(1, 2, 1)
        show_image(image, title='Image')
        plt.subplot(1, 2, 2)
        show_image(mask, title='Mask')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    (x_train, y_train),(x_test, y_test) = load_data("./input/water_segmentation_dataset/water_v1/JPEGImages/ADE20K")
    indexes = []
    for i in range(10):
        index = randint(1000)
        indexes.append(index)
    images = x_train[indexes]
    masks = y_train[indexes]
    show_image_set(images, masks)
