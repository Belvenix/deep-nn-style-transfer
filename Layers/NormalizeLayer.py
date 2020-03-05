# "VGG networks are trained on images with each channel normalized by
# mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]"

import torch.nn as nn
import torch


class NormalizeLayer(nn.Module):
    """
    Layer used to normalize the input data.
    Useful for transfer learning to ensure that our data is
    normalized in the same way as the training data for model was
    """
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        # Reshape the mean and std to [C, 1, 1] to work with pytorch tensors of shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mean = torch.tensor(mean).reshape(-1, 1, 1).to(device)
        self.std = torch.tensor(std).reshape(-1, 1, 1).to(device)

    def forward(self, image):
        # Normalize the image
        norm_image = (image - self.mean) / self.std
        return norm_image

