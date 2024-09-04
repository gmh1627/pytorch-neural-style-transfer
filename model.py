# model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:29]
        
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features