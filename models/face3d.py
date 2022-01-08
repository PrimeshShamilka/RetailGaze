import torch
import torch.nn as nn
import math
import numpy as np
from models.resnet_modified import resnet50
from resnest.torch import resnest50
import matplotlib.pyplot as plt

class Gaze360Static(nn.Module):
    def __init__(self):
        super(Gaze360Static, self).__init__()
        self.img_feature_dim = 256
        self.base_model = resnet50(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        # self.last_layer = nn.Linear(self.img_feature_dim, 3)

    def forward(self, x_in):
        face = x_in["face"]
        base_out = self.base_model(face)
        base_out = torch.flatten(base_out, start_dim=1)
        # output = self.last_layer(base_out)
        # angular_output = output[:,:2]
        # angular_output[:,0:1] = math.pi*nn.Tanh()(angular_output[:,0:1])
        # angular_output[:,1:2] = (math.pi/2)*nn.Tanh()(angular_output[:,1:2])
        # var = math.pi*nn.Sigmoid()(output[:,2:3])
        # var = var.view(-1,1).expand(var.size(0), 2)
        # return angular_output,var
        return base_out

class PinBallLoss(nn.Module):
    def __init__(self):
        super(PinBallLoss, self).__init__()
        self.q1 = 0.1
        self.q9 = 1-self.q1

    def forward(self, output_o, target_o, var_o):
        q_10 = target_o-(output_o-var_o)
        q_90 = target_o-(output_o+var_o)
        loss_10 = torch.max(self.q1*q_10, (self.q1-1)*q_10)
        loss_90 = torch.max(self.q9*q_90, (self.q9-1)*q_90)
        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10+loss_90

class Face3D(nn.Module):
    def __init__(self):
        super(Face3D, self).__init__()
        self.depth = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.base_model = resnet50(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.last_layer = nn.Linear(self.img_feature_dim, 3)
        self.tanh = nn.Tanh()
        self.backbone = resnest50(pretrained=True)
        self.fc2 = nn.Linear(1000, self.img_feature_dim)

    def forward(self, image, face):
        self.depth.eval()
        with torch.no_grad():
            id = self.depth(image)
            id = torch.nn.functional.interpolate(id.unsqueeze(1),size=image.shape[2:],mode="bicubic",align_corners=False,)
        # out = self.backbone(face)
        # out = self.fc2(out)
        base_out = self.base_model(face)
        base_out = torch.flatten(base_out, start_dim=1)
        output = self.last_layer(base_out)
        return output,id


class Face3D_Bias(nn.Module):
    def __init__(self):
        super(Face3D_Bias, self).__init__()
        self.depth = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.base_model = resnet50(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.last_layer = nn.Linear(self.img_feature_dim, 4)
        self.tanh = nn.Tanh()
        self.backbone = resnest50(pretrained=True)
        self.fc2 = nn.Linear(1000, self.img_feature_dim)

    def forward(self, image, face):
        self.depth.eval()
        with torch.no_grad():
            id = self.depth(image)
            id = torch.nn.functional.interpolate(id.unsqueeze(1),size=image.shape[2:],mode="bicubic",align_corners=False,)
        base_out = self.base_model(face)
        base_out = torch.flatten(base_out, start_dim=1)
        output = self.last_layer(base_out)
        bias = math.pi*nn.Sigmoid()(output[:, 3:])
        bias = bias.view(-1, 1).expand(bias.size(0), 3)
        output = output[:, :3]
        return output, bias, id

class FaceDepth(nn.Module):
    def __init__(self):
        super(FaceDepth, self).__init__()
        self.model_type = "DPT_Hybrid"
        self.depth = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.base_model = resnet50(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.last_layer = nn.Linear(self.img_feature_dim, 1)
        # self.fc1 = nn.Linear(self.img_feature_dim, 128)
        # self.fc2 = nn.Linear(128, 56)
        # self.last_layer = nn.Linear(56, 1)
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image, face):
        self.depth.eval()
        with torch.no_grad():
            # image = self.transform(image)
            id = self.depth(image)
            id = torch.nn.functional.interpolate(id.unsqueeze(1),size=image.shape[2:],mode="bicubic",align_corners=False,)
        base_out = self.base_model(face)
        base_out = torch.flatten(base_out, start_dim=1)
        # base_out = self.relu(self.fc1(base_out))
        # base_out = self.relu(self.fc2(base_out))
        # output = self.sigmoid(self.last_layer(base_out))
        output = self.last_layer(base_out)
        return output,id
