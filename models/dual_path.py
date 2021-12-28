import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from resnest.torch import resnest50
import torchvision

class GazeDual(nn.Module):
    """
    Dual pathway gaze estimation model 
    """
    def __init__(self):
        super(GazeDual,self).__init__()
        self.inplanes_scene = 64
        self.inplanes_face = 64
        self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.compress_conv1 = nn.Conv2d(2*2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        model = resnest50(pretrained=True)
        self.face_net = nn.Sequential(*(list(model.children())[:-2]))
        self.scence_net = nn.Sequential(*(list(model.children())[:-2]))


    def forward(self,scene,head):
        head_feat =  self.face_net(head)
        scene_feat =  self.face_net(scene)
        x =  torch.cat([scene_feat,head_feat],1)
        x = self.compress_conv1(x)
        x = self.compress_bn1(x)
        x = self.relu(x)
        x = self.compress_conv2(x)
        x = self.compress_bn2(x)
        x = self.relu(x)

        x = self.deconv1(x)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        
        return x


 