import torch
import torch.nn as nn
import math
import numpy as np
import torchvision.models as models

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.last_layer = nn.Linear(self.img_feature_dim, 2)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(1000, self.img_feature_dim)    

    def forward(self, image):
        feat = self.base_model(image)
        feat = torch.flatten(feat, start_dim=1)
        output = self.last_layer(feat)
        return output

