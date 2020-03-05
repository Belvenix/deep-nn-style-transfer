import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(activations):
    """
    Compute the Gram matrix of filters to extract the style
    :param activations: features from neural network layer
    :return: Normalized Gram matrix
    """
    # Get the shape of activations
    n, c, h, w = activations.size()

    # Resize the activations to 2D matrix of size (n*c, h*w)
    activations = activations.view(n*c, h*w)
    # or activations = torch.reshape(activations,(n*c,h*w))

    # Compute gram matrix
    g_matrix = torch.mm(activations, activations.t())
    # Normalize the matrix
    return g_matrix


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
        
        # Detach the target content from the computation graph to ensure
        # it stays constant and does not throw errors during computation
        # and define loss
        # Reference link: https://pytorch.org/docs/stable/autograd.html?highlight=detach#torch.Tensor.detach
        
        # Add code here
        self.target_activations = target_activations.detach()
        self.target_gram = gram_matrix(self.target_activations)
        self.loss = 0

    def forward(self, generated_activations):
        # Compute the gram matrix for generated activations,
        # compute the style loss
        # and pass the activations forward in neural network
        
        # Compute the gram matrix for generated activations
        G = gram_matrix(generated_activations)
        
        # Compute the style loss
        self.loss = F.mse_loss(G, self.target_gram)
        
        # Pass activations forward in neural network
        return generated_activations
