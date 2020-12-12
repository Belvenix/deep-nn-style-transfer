import pprint
import torch
import torchvision.models as models

from StyleTransfer.style_transfer_class import StyleTransfer
from StyleTransfer.utility_functions import image_loader, show_tensor, save_tensor
from e_stat.EBase.tools import TOOLS


# -- CONSTANTS --
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imsize = (512, 512) if torch.cuda.is_available() else (300, 300)
imsize_one_dim = 512 if torch.cuda.is_available() else 300
RESULTS_PATH = "images/results/"
IMAGES_PATH = "evaluate_files/"
CONTENT = "content/"
STYLE = "style/"
RESULT = "result/"

content_file = "content1.jpg"
style_file = "style93.jpg"
result_file = "style93@content1@cos.jpg"



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
    content_layers_req = ["Conv2d_10"]  # pick layer near the end
    style_layers_req = ["Conv2d_1", "Conv2d_3", "Conv2d_5", "Conv2d_9", "Conv2d_13"]  # pick layers after pooling

    # VGG19 specific mean and std used to normalize images during it's training
    # We will normalize our images using those same values to ensure best results
    # Change this if other model is loaded
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Load the images as preprocessed tensors

    content_tensor_image = image_loader(CONTENT, content_file, IMAGES_PATH)
    style_tensor_image = image_loader(STYLE, style_file, IMAGES_PATH)
    result_tensor_image = image_loader(RESULT, result_file, IMAGES_PATH)

    # Assert that they're same size

    assert content_tensor_image.size() == style_tensor_image.size(), 'Images are not the same size!'
    assert result_tensor_image.size() == style_tensor_image.size(), 'Images are not the same size!'

    # Run style transfer

    input_image = content_tensor_image.clone()

    style_transfer_module = StyleTransfer(model, content_tensor_image, style_tensor_image, input_image,
                            mean, std, content_layers_req, style_layers_req)

    style_transfer_nn_model = style_transfer_module.nn_model

    E_StatisticsTools = TOOLS(style_transfer_nn_model, imsize_one_dim)

    PCA_basis = E_StatisticsTools.my_PCA_Basis_Generater(style_tensor_image, style_transfer_module.style_layers)

    # Once the PCA_basis is generated, we need the following information to generate Base E statisics:
    style_dir = './images/style'  # the style target images
    source_dir = './test_sample/'  # sample images(synthesized images we want to quantify)
    source_list = 'sample.txt'  # lsit of sample images
    outputfile = 'E_BaseTEST.txt'  # the name of output file
    iteration = 0
    # This will generate a text file with each row represneting the information of one sample image with
    # 5 Base E statistics corresponding to 5 critical layers in VGG
    E_StatisticsTools.my_E_Basic_Statistics(source_list, outputfile,
                                          model, style_transfer_module,
                                         content_tensor_image, style_tensor_image, result_tensor_image, mean, std,
                                         content_layers_req, style_layers_req, iteration)

