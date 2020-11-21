import os

from keras.preprocessing.image import ImageDataGenerator

from utils import DNNConfigurer

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


def rename_files():
    print("Renaming files")
    directory_content_t = DNNConfigurer["data_files"]["CONTENT_ROOT"]
    directory_style_t = DNNConfigurer["data_files"]["STYLE_ROOT"]
    i = 1
    for filename in os.listdir(directory_content_t):
        filename = os.path.join(directory_content_t, filename)
        new_file_name = directory_content_t + "content_" + str(i) + ".jpg"
        os.rename(filename, new_file_name)
        i += 1

    i = 1
    for filename in os.listdir(directory_style_t):
        filename = os.path.join(directory_style_t, filename)
        new_file_name = directory_style_t + "style_" + str(i) + ".jpg"
        os.rename(filename, new_file_name)
        i += 1


if __name__ == '__main__':
    rename_files()
