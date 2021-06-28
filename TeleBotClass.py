# IMPORTS

from PIL import Image
import numpy as np

import torch
import torch.optim as optimizer
import torch.nn.functional as f
from torchvision import models

from BotFunctions import load_image, im_convert, get_features, gram_matrix
#from VGGClass import VggModel


# CLASS

class StyleTransferModel:

    def __init__(self):

        # Set hyper parameters and initialize model

        self.init_model()

        self.set_hyper()

    def init_model(self):

        # Initialize model

        #self.vgg = VggModel()
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        self.vgg = models.vgg19()
        self.vgg.load_state_dict(state_dict)

        for param in self.vgg.parameters():
            param.requires_grad = False

        # Some trick
        for i, layer in enumerate(self.vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                self.vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        #self.vgg.load_state_dict(torch.load('vgg_telebot.pth', map_location=torch.device('cpu')))

        self.vgg.eval()

    def get_model(self):

        # Return model

        return {'model': self.vgg}

    def set_hyper(self):

        # Set hyper parameters

        self.gram_weights = {'conv1_1': 0.75,
                             'conv2_1': 0.6,
                             'conv3_1': 0.5,
                             'conv4_1': 0.4,
                             'conv5_1': 0.3}

        self.content_weight = 1e3
        self.style_weight = 1e3

    def get_hyper(self):

        # Return hyper parameters

        return {'gram weights': self.gram_weights,
                'content weight': self.content_weight,
                'style weight': self.style_weight}

    def get_images_and_features(self, content_name, style_name):

        # Load images, get features, get gram matrix, create target image

        content, shape = load_image(content_name, get_shape=True)
        style = load_image(style_name, shape=shape)

        content_features = get_features(content, self.vgg, 'content')
        style_features = get_features(style, self.vgg, 'style')

        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

        # It shows more interesting result than torch.randn
        target = content.clone().requires_grad_(True)

        del style_features

        return content, content_features, style, style_grams, target

    def set_optimizer_and_scheduler(self, target, lr=0.12, step_size=470, gamma=0.5):

        # Set optimizer and scheduler

        self.optimizer = optimizer.Adam([target], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)

    def get_optimizer_and_scheduler(self):

        # Return optimizer and scheduler

        return {'optimizer': self.optimizer,
                'scheduler': self.scheduler}

    def get_all_parameters(self):

        # Return all parameters

        return {'model': self.vgg,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
                'loss parameters': {'gram weights': self.gram_weights,
                                    'content weight': self.content_weight,
                                    'style weight': self.style_weight}
                }

    def transfer(self, content_features, style_grams, target, epochs):

        # Studying part

        print('Start transfer')  # Just for developer

        for epoch in range(epochs):

            # Just usual pipeline
            self.optimizer.zero_grad()
            target_features = get_features(target, self.vgg, 'target')

            # Get content loss
            loss = f.mse_loss
            content_loss = loss(target_features['conv4_2'], content_features['conv4_2'])

            # Get style loss
            style_loss = 0
            for layer in self.gram_weights:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                style_gram = style_grams[layer]
                layer_style_loss = self.gram_weights[layer] * loss(target_gram, style_gram)
                style_loss += layer_style_loss / (d * h * w)

            # Optimizer and scheduler step
            total_loss = self.content_weight * content_loss + self.style_weight * style_loss
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 50 == 0:
                print(epoch)  # Just for developer

        print('End transfer')  # Just for developer

        final_img = im_convert(target)

        return final_img

    def save_result(self, final_img, name='result.jpg'):

        # Saving result in result.jpg

        img = np.uint8(final_img*255)

        result = Image.fromarray(img)

        result.save(name)

        return name

    def forward(self, content_name, style_name, epochs=501):

        # MAIN FUNCTION

        content, content_features, style, style_grams, target = self.get_images_and_features(
            content_name, style_name)

        self.set_optimizer_and_scheduler(target)

        final_img = self.transfer(content_features, style_grams, target, epochs)

        return self.save_result(final_img)
