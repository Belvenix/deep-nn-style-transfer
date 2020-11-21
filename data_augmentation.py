import os

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from utils import DNNConfigurer


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

    # i = 1
    # for filename in os.listdir(directory_style_t):
    #     filename = os.path.join(directory_style_t, filename)
    #     new_file_name = directory_style_t + "style_" + str(i) + ".jpg"
    #     os.rename(filename, new_file_name)
    #     i += 1
    print("Finished renaming")


def augment_content():
    print("Start data augmentation")
    datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0.5,
        zoom_range=0.9,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    cr = DNNConfigurer["data_files"]["CONTENT_ROOT"]
    root = DNNConfigurer["data_files"]["IMAGE_ROOT"] + '/augmented/'
    for filename in os.listdir(cr):
        d = str(filename).split("_")[1].split(".")[0]
        print(d)
        if int(d) < 74:
            continue
        filename_ext = os.path.join(cr, filename)
        filename_crop = str(filename).split(".")[0]
        print("Generating extra data for: " + str(filename_ext))
        img = load_img(filename_ext)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        aug_name = filename_crop + '_augmented'
        for batch in datagen.flow(x, batch_size=1, save_to_dir=root, save_prefix=aug_name, save_format='jpg'):
            i += 1
            if i > 3:
                break
    print("Finish data augmentation")


if __name__ == '__main__':
    # rename_files()
    augment_content()
