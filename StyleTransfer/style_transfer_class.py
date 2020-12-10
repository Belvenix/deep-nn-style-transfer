import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from Layers.ContentLayer import ContentLayer
from Layers.NormalizeLayer import NormalizeLayer
from Layers.StyleLayer import StyleLayer

# -- CONSTANTS --
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imsize = (512, 512) if torch.cuda.is_available() else (300, 300)


class StyleTransfer:
    def __init__(self, nn_model, content_image, style_image, input_image, normalize_mean, normalize_std,
                 content_layers_req, style_layers_req, style_weight=100000, content_weight=1):

        """Runs the style transfer on input image"""
        # Get the rebuilded model and style and content layers

        new_model, style_layers, content_layers = self.rebuild_model(nn_model, content_image.unsqueeze(0),
                                                                     style_image.unsqueeze(0), normalize_mean,
                                                                     normalize_std,
                                                                     content_layers_req, style_layers_req)
        self._input_image = input_image
        self.model = new_model
        self.style_layers = style_layers
        self.content_layers = content_layers
        self._style_layer_weight = 1 / len(style_layers)
        self._style_weight = style_weight
        self._content_weight = content_weight

        # Get the LBFGS optimizer
        self._input_batch = input_image.unsqueeze(0)
        self._optimizer = self.get_optimizer(self._input_batch)


    # 4. Rebuild the network
    def rebuild_model(self, nn_model, content_image, style_image,
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

            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)

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
    def get_optimizer(self, input_img):
        """Uses LBFGS as proposed by Gatys himself because it gives best results"""
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    # train model

    # To work with optimizer like LBFGS  you need to use something called "closure"
    # http://sagecal.sourceforge.net/pytorch/index.html <- info here

    # Basically you need to put all the training code in function  defined inside the loop
    # and then pass the function as input to step() method of LBFGS optimizer.
    # Gradients are zeroed with zero_grad() inside this function, same for loss' backward() method
    # closure returns computed loss

    # LOOP START
    def train_model(self, num_steps=1, style_weight=100000, content_weight=1):
        # DEFINE THE CLOSURE FUNCTIONS START
        def closure():
            # Inside closure function
            # correct the values of updated input image to range from 0 to 1 with clamp_()
            with torch.no_grad():
                self._input_batch.clamp_(0, 1)

            # Zero the gradients from last iteration and
            # forward the image through network

            self._optimizer.zero_grad()
            self.model(self._input_batch)

            # Compute the style and content stores
            # based on values computed in style/content layers during forward propagation

            style_loss = 0
            for layer in self.style_layers:
                style_loss = style_loss + layer.loss

            content_loss = 0
            for layer in self.content_layers:
                content_loss = content_loss + layer.loss

            # We need to multiply the scores by weights
            # as described in the paper https://arxiv.org/pdf/1508.06576.pdf,
            # formula nr. 7

            weighed_style_loss = self._style_layer_weight * self._style_weight * style_loss
            weighed_content_loss = self._content_weight * content_loss

            # Compute total loss and propagate it backwards

            loss = weighed_style_loss + weighed_content_loss
            loss.backward()

            # Print training info every X epochs

            print("Epoch = " + str(i) + "\t Style_loss = " + str(weighed_style_loss.item())
                  + "\t Content_loss = " + str(weighed_content_loss.item()) + "\t Loss = " + str(loss.item()))

            # return computed total score value

            return loss

        for i in range(num_steps):
            # Optimizer step
            self._optimizer.step(closure)

        # LOOP END

        # Clamp the image values to a range from 0 to 1

        self._input_image.data.clamp_(0, 1)

        # return image

        return self._input_batch[0]