import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
import math
import numpy as np
from resnest.torch import resnest50
import torchvision

class HypoAttention(nn.Module):
    def __init__(self):
        super(HypoAttention,self).__init__()
        self.inplanes_scene = 64
        self.inplanes_face = 64
        self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.conv1_scene = nn.Conv2d(5, 3, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.bn1_scene = nn.BatchNorm2d(3)

        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)
        self.compress_conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*7*7, 669)
        self.fc2 = nn.Linear(669, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 169)

        self.smax = nn.LogSoftmax(dim=1)
        self.nolog_smax = nn.Softmax(dim=1)

        self.fc_0_0 = nn.Linear(169, 25)
        self.fc_0_m1 = nn.Linear(169, 25)
        self.fc_0_1 = nn.Linear(169, 25)
        self.fc_m1_0 = nn.Linear(169, 25)
        self.fc_1_0 = nn.Linear(169, 25)
        self.attn = nn.Linear(1296, 1*7*7)
        count=0
        # Initialize weights
        for m in self.modules():
            count+=1
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        model = torchvision.models.vgg16(pretrained=True)
        self.face_net = nn.Sequential(*(list(model.children())[:-2]))
        self.scence_net = nn.Sequential(*(list(model.children())[:-2]))
        for param in self.face_net.parameters():
            param.requires_grad = False
        for param in self.scence_net.parameters():
            param.requires_grad = False

    def forward(self,image,face,head_channel,object_channel):
        face_feat =  self.face_net(face)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head_channel))).view(-1, 784)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 512)
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2)
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        im = torch.cat((image, head_channel,object_channel), dim=1)
   
        im = self.conv1_scene(im)
        im = self.bn1_scene(im)
        im = self.relu(im)
        scene_feat =  self.scence_net(im)
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat)

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
        encoding = self.compress_conv2(scene_face_feat)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv3(encoding)
        encoding = self.compress_bn3(encoding)
        encoding = self.relu(encoding)
        encoding = encoding.view(-1, 256 * 7 * 7)

        fc = self.relu(self.fc1(encoding))
        fc =  self.relu(self.fc2(fc))
        fc =  self.relu(self.fc3(fc))
        output =  self.sigmoid(self.fc4(fc))
        out_0_0 = self.smax(self.fc_0_0(output))
        out_1_0 = self.smax(self.fc_1_0(output))
        out_m1_0 = self.smax(self.fc_m1_0(output))
        out_0_m1 = self.smax(self.fc_0_m1(output))
        out_0_1 = self.smax(self.fc_0_1(output))
        return [out_0_0, out_1_0, out_m1_0, out_0_m1, out_0_1]


    def raw_hm(self,image,face,head_channel,object_channel):
        face_feat =  self.face_net(face)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head_channel))).view(-1, 784)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 512)
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2)
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        im = torch.cat((image, head_channel,object_channel), dim=1)
   
        im = self.conv1_scene(im)
        im = self.bn1_scene(im)
        im = self.relu(im)
        scene_feat =  self.scence_net(im)
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat)

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
        encoding = self.compress_conv2(scene_face_feat)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv3(encoding)
        encoding = self.compress_bn3(encoding)
        encoding = self.relu(encoding)
        encoding = encoding.view(-1, 256 * 7 * 7)

        fc = self.relu(self.fc1(encoding))
        fc =  self.relu(self.fc2(fc))
        fc =  self.relu(self.fc3(fc))
        output =  self.sigmoid(self.fc4(fc))
        hm = torch.zeros(output.size(0), 15, 15).cuda()
        count_hm = torch.zeros(output.size(0), 15, 15).cuda()

        f_0_0 = self.nolog_smax(self.fc_0_0(output)).view(-1, 5, 5)
        f_1_0 = self.nolog_smax(self.fc_1_0(output)).view(-1, 5, 5)
        f_m1_0 = self.nolog_smax(self.fc_m1_0(output)).view(-1, 5, 5)
        f_0_m1 = self.nolog_smax(self.fc_0_m1(output)).view(-1, 5, 5)
        f_0_1 = self.nolog_smax(self.fc_0_1(output)).view(-1, 5, 5)

        f_cell = []
        f_cell.extend([f_0_m1, f_0_1, f_m1_0, f_1_0, f_0_0])

        v_x = [0, 1, -1, 0, 0]
        v_y = [0, 0, 0, -1, 1]

        for k in range(5):
            dx, dy = v_x[k], v_y[k]
            f = f_cell[k]
            for x in range(5):
                for y in range(5):

                    i_x = 3*x - dx
                    i_x = max(i_x, 0)
                    if x == 0:
                        i_x = 0

                    i_y = 3*y - dy
                    i_y = max(i_y, 0)
                    if y == 0:
                        i_y = 0

                    f_x = 3*x + 2 - dx
                    f_x = min(14, f_x)
                    if x == 4:
                        f_x = 14

                    f_y = 3*y + 2 - dy
                    f_y = min(14, f_y)
                    if y == 4:
                        f_y = 14

                    a = f[:, x, y].contiguous()
                    a = a.view(output.size(0), 1, 1)

                    hm[:, i_x: f_x+1, i_y: f_y+1] =  hm[:, i_x: f_x+1, i_y: f_y+1] + a
                    count_hm[:, i_x: f_x+1, i_y: f_y+1] = count_hm[:, i_x: f_x+1, i_y: f_y+1] + 1

        hm_base = hm.div(count_hm)
        raw_hm = hm_base
        hm_base = hm_base.unsqueeze(1)

        hm_base = F.interpolate(input = hm_base, size = (227, 227), mode='bicubic', align_corners=False)

        hm_base = hm_base.squeeze(1)

        return hm_base.view(-1, 227 * 227), raw_hm