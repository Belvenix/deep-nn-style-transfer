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

from Layers.NormalizeLayer import NormalizeLayer
from Layers.ContentLayer import ContentLayer
from Layers.StyleLayer import StyleLayer

# -- CONSTANTS --
imsize = (300, 300)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = "images/results/"
IMAGES_PATH = "images/"

# -- UTILITY FUNCTIONS --


def resize(...):
    """Resize the image to size specified in the beginning of code"""
    
    # Add code here


def image_loader(...):
    """Loads the images from disk as preprocessed tensors"""
       
    # Add code here


def to_image(...):
    """Converts tensor to PIL image"""
    
    # Add code here


def show_tensor(...):
    """
    Helper function to display the pytorch tensor as image.
    """
    
    # Add code here

def save_tensor(...):
    """
    Helper function to save pytorch tensor as jpg image.
    """
    
    # Add code here

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
def rebuild_model(...):
                  
    # Add code here


# 5. Define the optimizer
def get_optimizer(...):
    """Uses LBFGS as proposed by Gatys himself because it gives best results"""
      # Add code here


# 6. Write training function
def style_transfer(...):
    """Runs the style transfer on input image"""
      # Add code here


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
    
    # Add code here
    
    # Assert that they're same size
    
    # Add code here
    
    # Display them

    # Add code here
    
    # Run style transfer
 
    # Add code here
    
    # Show results
    
    # Add code here

    # Testing the gitHub ~ Jakub