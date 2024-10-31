from tensorflow.image import resize
from keras.preprocessing.image import load_img, img_to_array

def load_image(path: str):
    img = load_img(path)
    img_array = img_to_array(img)
    normalized_img_array = img_array/255.
    resized_img = resize(normalized_img_array, (256, 256))
    return resized_img

if __name__ == "__main__":
    load_image("./input/water_segmentation_dataset/water_v1/JPEGImages/ADE20K/ADE_train_00000004.png")