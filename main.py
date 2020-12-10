import os
import pprint
import time

import torch
import torchvision.models as models

from StyleTransfer.style_transfer_class import StyleTransfer
from StyleTransfer.utility_functions import image_loader, save_tensor
# -- CONSTANTS --
from utils import DNNConfigurer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imsize = (512, 512) if torch.cuda.is_available() else (300, 300)
RESULTS_PATH = "images/results/"
IMAGES_PATH = "images/"
CONTENT = "content/"
STYLE = "style/"

if __name__ == '__main__':
    times = []
    try:
        # Pretty printer used for nice display of architecture
        pp = pprint.PrettyPrinter(indent=4)

        # we dont need the last fully connected layers and adaptive avgpool so we copy only CNN part of VGG19
        # We send it to GPU and set it to run in eval() mode as in Style Transfer we won't need
        # to train the network
        model_vgg19 = models.vgg19(pretrained=True).features.to(device).eval()
        model_vgg16 = models.vgg16(pretrained=True).features.to(device).eval()
        pprint.pprint(model_vgg19)
        pprint.pprint(model_vgg16)

        # Define after which layers we want to input our content/style layers
        # they will enable us to compute the style and content losses during forward propagation
        content_layers_req = ["Conv2d_10"]  # pick layer near the end
        style_layers_req = ["Conv2d_1", "Conv2d_3", "Conv2d_5", "Conv2d_9", "Conv2d_11"]  # pick layers after pooling

        # VGG19 specific mean and std used to normalize images during it's training
        # We will normalize our images using those same values to ensure best results
        # Change this if other model is loaded
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Load the images as preprocessed tensors

        content_tensor_image = image_loader(CONTENT, "content_1.jpg")
        style_tensor_image = image_loader(STYLE, "style_1.jpg")

        # Assert that they're same size

        assert content_tensor_image.size() == style_tensor_image.size(), 'Images are not the same size!'

        # Run style transfer
        # Main loop going over all images
        directory_content_t = DNNConfigurer["data_files"]["CONTENT_ROOT"]
        directory_style_t = DNNConfigurer["data_files"]["STYLE_ROOT"]

        for filename_c in os.listdir(directory_content_t):
            for filename_s in os.listdir(directory_style_t):
                # Load the images as preprocessed tensors

                content_tensor_image = image_loader(CONTENT, filename_c)
                style_tensor_image = image_loader(STYLE, filename_s)

                # Assert that they're same size

                assert content_tensor_image.size() == style_tensor_image.size(), 'Images are not the same size!'

                input_image = content_tensor_image.clone()

                style_transfer_module = StyleTransfer(model_vgg16, content_tensor_image, style_tensor_image,
                                                      input_image,
                                                      mean, std, content_layers_req, style_layers_req)
                start = time.time()
                result = style_transfer_module.train_model(num_steps=15)
                duration = time.time() - start
                times.append(duration)
                save_tensor(filename_c + '___' + filename_s, result)

    finally:
        print(times)
