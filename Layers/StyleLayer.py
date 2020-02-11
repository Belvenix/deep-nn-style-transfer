
import torch.nn as nn
import torch.nn.functional as F
import torch


def gram_matrix(activations):
    """
    Compute the Gram matrix of filters to extract the style
    :param activations: features from neural network layer
    :return: Normalized Gram matrix
    """
    # Add code here


class StyleLayer(nn.Module):
    """
    Custom style layer used to access the feature space
    of previous neural network layer and compute the style loss
    for input image
    """
    def __init__(self, target_activations):
        super(StyleLayer, self).__init__()
        # Compute the gram matrix of target activations for style image
        # and define loss
        
        # Add code here

    def forward(self, generated_activations):
        # Compute the gram matrix for generated activations,
        # compute the style loss
        # and pass the activations forward in neural network
        
        # Add code here
