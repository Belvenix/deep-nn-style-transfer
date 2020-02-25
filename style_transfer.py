import matplotlib.pyplot as plt
from PIL import Image

import torchvision.models as models
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim

import numpy as np
import torch
import pprint
import copy
from collections import OrderedDict

from Layers.NormalizeLayer import NormalizeLayer
from Layers.ContentLayer import ContentLayer
from Layers.StyleLayer import StyleLayer

# -- CONSTANTS --
imsize = (300, 300)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = "images/results/"
IMAGES_PATH = "images/"


# -- UTILITY FUNCTIONS --


def resize(pil_image):
    """Resize the image to size specified in the beginning of code"""
    resized_image = pil_image.resize(imsize)
    return resized_image


def image_loader(file_name):
    """Loads the images from disk as preprocessed tensors"""
    image = Image.open(IMAGES_PATH + file_name)
    resized_image = resize(image)
    tensor = (transforms.ToTensor()(resized_image)).to(device)
    return tensor


def to_image(tensor):
    """Converts tensor to PIL image"""
    image = transforms.ToPILImage()(tensor.cpu())
    return image


def show_tensor(tensor):
    """Helper function to display the pytorch tensor as image."""
    to_image(tensor).show()


def save_tensor(file_name, tensor):
    """Helper function to save pytorch tensor as jpg image."""
    image = to_image(tensor)
    image.save(RESULTS_PATH + file_name, format='JPEG')


# Now to properly implement the style transfer we need to:
# 1. Define the custom content layer that will allow us to compute content loss (ContentLayer.py)

# 2. Define the style content layer that will allow us to compute content loss (StyleLayer.py)
# 2.1 Define the computation of Gram matrix (StyleLayer.py)

# 3. Create normalization layer to ensure that images that are passed
# to the network have the same mean and standard deviation as the ones
# VGG was trained on (NormalizeLayer.py)

# 4. Rebuild the VGG19 by inserting the custom content and style layers
# after chosen layers in original VGG19
# this way we can access the activations values of layers in VGG19
# and compute the style and content losses inside content/style layers

# 5. Define the LBFGS optimizer and pass the input image with gradients
# enabled to it

# 6. Write training function


# 4. Rebuild the network
def rebuild_model(nn_model, content_image, style_image,
                  normalize_mean, normalize_std,
                  content_layers_req, style_layers_req):
    """ Creates new model that is a modified copy of input nn_model.

        Inserts StyleLayer and ContentLayer after layers specified in
        content_layers_req and style_layers_req lists.

        Check out those variables in the main function on the end of file
        to see the naming structure of layers.

        Returns modified model and lists of StyleLayers and ContentLayers
        inserted into modified model"""
    # Deepcopy the model

    model_copy = copy.deepcopy(nn_model)

    # Create a new model that will be modified version of input model
    # starts with normalization layer to ensure all images that are
    # inserted are normalized like the ones original model was trained on

    # Check out torch.nn.Sequential and remember to make sure it resides on correct device
    # Also ensure that input images are on the same device

    new_model = nn.Sequential().to(device)
    new_model.add_module("Normalize_1", NormalizeLayer(normalize_mean, normalize_std))

    # We need to keep track of the losses in content/style layers
    # to compute the total loss therefore we keep those in a list and return it
    # at the end of the function
    # This will let us access loss values in those layers

    content_layers_list = []
    style_layers_list = []

    # Loop over the layers in original network

    i = 0
    layer_i = 0
    last_significant_layer = 0

    for layer in model_copy.children():
        # The layers in vgg are not numerated so we have to add numeration
        # to copied layers so we can append our content and style layers to it

        # Check which instance this layer is to name it appropiately
        # In vgg we only use nn.Conv2d,  nn.ReLU, nn.MaxPool2d
        # For naming conventions use "Conv2d_{}".format(i) and appropiately for other instances

        i += 1

        if isinstance(layer, nn.Conv2d):
            layer_i += 1

        name = (type(layer).__name__ + "_{}").format(layer_i)

        # Layer has now numerated name so we can find it easily
        # Add it to our model

        new_model.add_module(name, layer)

        # After adding check if it is a layer after which we should add our content
        # or style layer
        # Check for content layers
        if name in content_layers_req:
            # Get the activations for original content image in this layer
            # and detach the from pytorch's graph

            content_activations = new_model.forward(content_image).detach()

            # Create the content layer

            content_layer = ContentLayer(content_activations)

            # Append it to the module with proper name

            i += 1
            content_layer_name = (type(content_layer).__name__ + "_{}").format(layer_i)
            new_model.add_module(content_layer_name, content_layer)
            content_layers_list.append(content_layer)
            last_significant_layer = i + 1

        # Check for style layers
        if name in style_layers_req:
            # Get the activations for original style image in this layer
            # and detach the from pytorch's graph
            style_activations = new_model.forward(style_image).detach()

            # Create the style layer

            style_layer = StyleLayer(style_activations)

            # Append it to the module with proper name

            i += 1
            style_layer_name = (type(style_layer).__name__ + "_{}").format(layer_i)
            new_model.add_module(style_layer_name, style_layer)
            style_layers_list.append(style_layer)
            last_significant_layer = i + 1

    # Add this point our new model is the same as input model but with
    # StyleLayers and ContentLayers inserted after required layers
    # we don't need any layers after the last style or content layer
    # so we need to delete them from the model

    significant_layers = list(new_model.named_children())[0:last_significant_layer]
    new_model = nn.Sequential(OrderedDict(significant_layers))

    # return model and StyleLayer and ContentLayer lists
    return new_model, style_layers_list, content_layers_list


# 5. Define the optimizer
def get_optimizer(input_img):
    """Uses LBFGS as proposed by Gatys himself because it gives best results"""
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# 6. Write training function
def style_transfer(nn_model, style_layers, content_layers):
    """Runs the style transfer on input image"""
    # Add code here
    pass


if __name__ == '__main__':
    # Pretty printer used for nice display of architecture
    pp = pprint.PrettyPrinter(indent=4)

    # we dont need the last fully connected layers and adaptive avgpool so we copy only CNN part of VGG19
    # We send it to GPU and set it to run in eval() mode as in Style Transfer we won't need
    # to train the network
    model = models.vgg19(pretrained=True).features.to(device).eval()
    pprint.pprint(model)

    # Define after which layers we want to input our content/style layers
    # they will enable us to compute the style and content losses during forward propagation
    content_layers_req = ["Conv2d_5"]  # pick layer near the middle
    style_layers_req = ["Conv2d_1", "Conv2d_2", "Conv2d_3", "Conv2d_4", "Conv2d_5", "Conv2d_6", "Conv2d_7", "Conv2d_8"]

    # VGG19 specific mean and std used to normalize images during it's training
    # We will normalize our images using those same values to ensure best results
    # Change this if other model is loaded
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Load the images as preprocessed tensors

    content_tensor_image = image_loader('mountain-landscape-2031539_1280.jpg')
    style_tensor_image = image_loader('Starry-Night-canvas-Vincent-van-Gogh-New-1889.jpg')

    # Assert that they're same size

    assert content_tensor_image.size() == style_tensor_image.size(), 'Images are not the same size!'

    # Display them

    show_tensor(content_tensor_image)
    show_tensor(style_tensor_image)

    new_model, style_layers, content_layers = rebuild_model(model, content_tensor_image.unsqueeze(0),
                                                            style_tensor_image.unsqueeze(0), mean, std,
                                                            content_layers_req, style_layers_req)
    pprint.pprint(new_model)
    # Run style transfer

    # Add code here

    # Show results

    # Add code here
