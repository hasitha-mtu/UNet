import os

import numpy as np
from keras.utils import load_img, img_to_array
from tensorflow.image import resize
from glob import glob
from tqdm import tqdm
import random

def load_image(path: str):
    img = load_img(path)
    img_array = img_to_array(img)
    normalized_img_array = img_array/255.
    resized_img = resize(normalized_img_array, (256, 256))
    return resized_img

def load_data(image_path: str):
    total_images = len(os.listdir(image_path))
    print(f'total number of images in path is {total_images}')
    all_image_paths = sorted(glob(image_path+"/*.png"))
    train_paths = all_image_paths[:1000]
    x_train, y_train = load_images(train_paths)
    test_paths = all_image_paths[1000:]
    x_test, y_test = load_images(test_paths)
    print('all images and masks loaded')
    return (x_train, y_train),(x_test, y_test)

def load_images(paths):
    images = np.zeros(shape=(len(paths), 256, 256, 3))
    masks = np.zeros(shape=(len(paths), 256, 256, 3))
    for i, path in tqdm(enumerate(paths), total=len(paths), desc="Loading"):
        image = load_image(path)
        images[i] = image
        mask_path = path.replace("JPEGImages", "Annotations")
        mask = load_image(mask_path)
        masks[i] = mask
    return images, masks

def load_drone_dataset(path, file_extension = "jpg"):
    total_images = len(os.listdir(path))
    print(f'total number of images in path is {total_images}')
    all_image_paths = sorted(glob(path + "/*."+file_extension))
    random.Random(1337).shuffle(all_image_paths)
    print(all_image_paths)
    train_paths = all_image_paths[:300]
    print(f"train image count : {len(train_paths)}")
    x_train, y_train = load_drone_images(train_paths)
    test_paths = all_image_paths[300:]
    print(f"test image count : {len(test_paths)}")
    x_test, y_test = load_drone_images(test_paths)
    return (x_train, y_train),(x_test, y_test)

def load_drone_images(paths):
    images = np.zeros(shape=(len(paths), 256, 256, 3))
    masks = np.zeros(shape=(len(paths), 256, 256, 3))
    for i, path in tqdm(enumerate(paths), total=len(paths), desc="Loading"):
        image = load_image(path)
        images[i] = image
        mask_path = path.replace("images", "annotations")
        mask_path = mask_path.replace("jpg", "png")
        mask = load_image(mask_path)
        # print(f"image path : {path}")
        # print(f"mask path : {mask_path}")
        masks[i] = mask
    return images, masks

if __name__ == "__main__":
    print('load image for testing')
    # load_image("./input/water_segmentation_dataset/water_v1/JPEGImages/ADE20K/ADE_train_00000004.png")
    # load_data("./input/water_segmentation_dataset/water_v1/JPEGImages/ADE20K")
    (x_train, y_train),(x_test, y_test) = load_drone_dataset("input/drone_dataset/images")
    print(f"train data shape: {x_train.shape}")
    print(f"test data shape: {x_test.shape}")
