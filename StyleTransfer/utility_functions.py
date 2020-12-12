import torch

import torchvision.transforms as transforms
from PIL import Image

# -- CONSTANTS --
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imsize = (512, 512) if torch.cuda.is_available() else (300, 300)
RESULTS_PATH = "images/results/"
IMAGES_PATH = "images/"
CONTENT = "content/"
STYLE = "style/"
# -- UTILITY FUNCTIONS --


def resize(pil_image):
    """Resize the image to size specified in the beginning of code"""
    resized_image = pil_image.resize(imsize)
    return resized_image


def image_loader(img_type, file_name, IMAGES_PATH="images/"):
    """Loads the images from disk as preprocessed tensors"""
    image = Image.open(IMAGES_PATH + img_type + file_name)
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
