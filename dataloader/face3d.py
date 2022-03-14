import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
# from models.gazenet import GazeNet

import time
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging

from scipy import signal
import matplotlib.pyplot as plt
from utils import get_paste_kernel, kernel_map


import pickle
from skimage import io
from dataloader import chong_imutils

import pandas as pd
np.random.seed(1)

def _get_transform(input_resolution):
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def _get_transform2():
    transform_list = []
    transform_list.append(transforms.Resize((448, 448)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def boxes2centers(normalized_boxes):
    center_x = (normalized_boxes[:,0] + normalized_boxes[:,2]) / 2
    center_y = (normalized_boxes[:,1] + normalized_boxes[:,3]) / 2
    center_x = np.expand_dims(center_x, axis=1)
    center_y = np.expand_dims(center_y, axis=1)
    normalized_centers = np.hstack((center_x, center_y))
    return normalized_centers

class GooDataset(Dataset):
    def __init__(self, root_dir, mat_file, training='train', include_path=False, input_size=224, output_size=64, imshow = False, use_gtbox=True):
        assert (training in set(['train', 'test', 'test_prediction']))
        self.root_dir = root_dir
        self.mat_file = mat_file
        self.training = training
        self.include_path = include_path
        self.input_size = input_size
        self.output_size = output_size
        self.imshow = imshow
        self.transform = _get_transform(input_size)
        self.transform2 = _get_transform2()
        self.use_gtbox= use_gtbox

        with open(mat_file, 'rb') as f:
            self.data = pickle.load(f)
            self.image_num = len(self.data)

        print("Number of Images:", self.image_num)
        logging.info('%s contains %d images' % (self.mat_file, self.image_num))

    def create_mask(self, seg_idx, width=640, height=480):
        seg_idx = seg_idx.astype(np.int64)
        seg_mask = np.zeros((height,width)).astype(np.uint8)
        for i in range(seg_idx.shape[0]):
            seg_mask[seg_idx[i,1],seg_idx[i,0]] = 255
        return seg_mask
    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):

        gaze_inside = True
        data = self.data[idx]
        image_path = data['filename']
        image_path = os.path.join(self.root_dir, image_path)
        #print(image_path)

        eye = [float(data['hx'])/640, float(data['hy'])/480]
        gaze = [float(data['gaze_cx'])/640, float(data['gaze_cy'])/480]
        eyess = np.array([eye[0],eye[1]]).astype(np.float)
        gaze_x, gaze_y = gaze

        image_path = image_path.replace('\\', '/')
        img = Image.open(image_path)
        img = img.convert('RGB')
        width, height = img.size
        #Get bounding boxes and class labels as well as gt index for gazed object
        gt_bboxes, gt_labels = np.zeros(1), np.zeros(1)
        gt_labels = np.expand_dims(gt_labels, axis=0)
        gaze_idx = np.copy(data['gazeIdx']).astype(np.int64) #index of gazed object
        gaze_class = np.copy(data['gaze_item']).astype(np.int64) #class of gazed object
        if self.use_gtbox:
            gt_bboxes = np.copy(data['ann']['bboxes']) / [640, 480, 640, 480]
            gt_labels = np.copy(data['ann']['labels'])

            gtbox = gt_bboxes[gaze_idx]
        
        x_min, y_min, x_max, y_max = gt_bboxes[-1] * [width,height,width,height]
        centers = (boxes2centers(gt_bboxes)*[224,224]).astype(int)
        location_channel = np.zeros((224,224), dtype=np.float32)
        for cen in centers:
            location_channel[cen[1],cen[0]] = 1
        head = centers[-1,:]
        gt_label = centers[gaze_idx,:]    
        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        
        
        # heatmap = get_paste_kernel((224 // 4, 224 // 4), gaze, kernel_map, (224 // 4, 224 // 4))
        # seg = data['seg']
        # seg_mask = self.create_mask(np.array(seg).astype(np.int64))
        # seg_mask = cv2.resize(seg_mask, (224//4, 224//4))
        # seg_mask = seg_mask.astype(np.float64)/255.0
        # heatmap = 0.5 * seg_mask + (1 - 0.5) * heatmap
        object_channel = chong_imutils.get_object_box_channel(gt_bboxes[:-1],width,height,resolution=self.input_size).unsqueeze(0)
        head_channel = chong_imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)
        head_box = gt_bboxes[-1]

        if self.imshow:
            img.save("img_aug.jpg")
            face.save('face_aug.jpg')

        if self.transform is not None:
            img = self.transform(img)
            face = self.transform2(face)
        if self.training == 'test':
            return img, face, location_channel,object_channel,head_channel ,head,gt_label, head_box, image_path, gtbox
        elif self.training == 'test_prediction':
            return img, face, head, gt_label, centers, gaze_idx, gt_bboxes, gt_labels
        else:
            return img, face, location_channel, object_channel,head_channel ,head,gt_label, head_box, gtbox
    


class GazeDataset(Dataset):
    def __init__(self, root_dir, mat_file, training='train', input_size=224, output_size=64,  include_path=False, imshow = False):
        assert (training in set(['train', 'test']))
        self.root_dir = root_dir
        self.mat_file = mat_file
        self.training = training
        self.include_path = include_path

        if self.training == "test":
            anns = loadmat(self.mat_file)
            self.bboxes = anns[self.training + '_bbox']
            self.gazes = anns[self.training + '_gaze']
            self.paths = anns[self.training + '_path']
            self.eyes = anns[self.training + '_eyes']
            self.meta = anns[self.training + '_meta']
            self.image_num = self.paths.shape[0]

            logging.info('%s contains %d images' % (self.mat_file, self.image_num))
        else:
            csv_path = mat_file.split(".mat")[0]+".txt"
            #print('csv path', csv_path)
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y',  'meta']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            # df = df[df['inout'] != -1]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
            #print('df', df.head())
            df.reset_index(inplace=True)
            self.y_train = df[['eye_x', 'eye_y', 'gaze_x',
                               'gaze_y']]
            self.X_train = df['path']
            self.image_num = len(df)

        self.input_size = input_size
        self.output_size = output_size
        self.imshow = imshow
        logging.info('%s contains %d images' % (self.mat_file, self.image_num))
        self.transform = _get_transform(input_size)
        print("Number of Images:", self.image_num)

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):

        if self.training == "test":
            # gaze_inside = True # always consider test samples as inside
            image_path = self.paths[idx][0][0]
            image_path = os.path.join(self.root_dir, image_path)
            eye = self.eyes[0, idx][0]
            # todo: process gaze differently for training or testing
            gaze = self.gazes[0, idx].mean(axis=0)
            gaze = gaze.tolist()
            eye = eye.tolist()
            # print('gaze', type(gaze), gaze)
            gaze_x, gaze_y = gaze
            image_path = image_path.replace('\\', '/')
            # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            eye_x, eye_y = eye
            # gaze_x, gaze_y = gaze
            gaze_inside = True # bool(inout)
        else:
            image_path = self.X_train.iloc[idx]
            eye_x, eye_y, gaze_x, gaze_y = self.y_train.iloc[idx]
            gaze_inside = True # bool(inout)
            head_point = np.array([eye_x, eye_y])
            gt_point = np.array([gaze_x, gaze_y])

        image_path = os.path.join(self.root_dir, image_path)
        img = Image.open(image_path)
        img = img.convert('RGB')
        width, height = img.size

        # expand face bbox a bit
        k = 0.1
        x_min = (eye_x - 0.15) * width
        y_min = (eye_y - 0.15) * height
        x_max = (eye_x + 0.15) * width
        y_max = (eye_y + 0.15) * height
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max < 0:
            x_max = 0
        if y_max < 0:
            y_max = 0
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        if self.imshow:
            img.save("origin_img.jpg")

        if self.training == 'test':
            imsize = torch.IntTensor([width, height])
        # else:
        #     ## data augmentation
        #
        #     # Jitter (expansion-only) bounding box size
        #     if np.random.random_sample() <= 0.5:
        #         k = np.random.random_sample() * 0.2
        #         x_min -= k * abs(x_max - x_min)
        #         y_min -= k * abs(y_max - y_min)
        #         x_max += k * abs(x_max - x_min)
        #         y_max += k * abs(y_max - y_min)
        #
        #     # Random Crop
        #     if np.random.random_sample() <= 0.5:
        #         # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
        #         crop_x_min = np.min([gaze_x * width, x_min, x_max])
        #         crop_y_min = np.min([gaze_y * height, y_min, y_max])
        #         crop_x_max = np.max([gaze_x * width, x_min, x_max])
        #         crop_y_max = np.max([gaze_y * height, y_min, y_max])
        #
        #         # Randomly select a random top left corner
        #         if crop_x_min >= 0:
        #             crop_x_min = np.random.uniform(0, crop_x_min)
        #         if crop_y_min >= 0:
        #             crop_y_min = np.random.uniform(0, crop_y_min)
        #
        #         # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
        #         crop_width_min = crop_x_max - crop_x_min
        #         crop_height_min = crop_y_max - crop_y_min
        #         crop_width_max = width - crop_x_min
        #         crop_height_max = height - crop_y_min
        #         # Randomly select a width and a height
        #         crop_width = np.random.uniform(crop_width_min, crop_width_max)
        #         crop_height = np.random.uniform(crop_height_min, crop_height_max)
        #
        #         # Crop it
        #         img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
        #
        #         # Record the crop's (x, y) offset
        #         offset_x, offset_y = crop_x_min, crop_y_min
        #
        #         # convert coordinates into the cropped frame
        #         x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
        #         # if gaze_inside:
        #         gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
        #                          (gaze_y * height - offset_y) / float(crop_height)
        #         width, height = crop_width, crop_height
        #
        #     # Random flip
        #     if np.random.random_sample() <= 0.5:
        #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #         x_max_2 = width - x_min
        #         x_min_2 = width - x_max
        #         x_max = x_max_2
        #         x_min = x_min_2
        #         gaze_x = 1 - gaze_x
        #
        #     # Random color change
        #     if np.random.random_sample() <= 0.5:
        #         img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
        #         img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
        #         img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        head_channel = chong_imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # shifted grids
        grid_size = 5
        gaze_label_size = 5
        v_x = [0, 1, -1, 0, 0]
        v_y = [0, 0, 0, -1, 1]

        shifted_grids = np.zeros((grid_size, gaze_label_size, gaze_label_size))
        for i in range(5):

            x_grid = int(np.floor( gaze_label_size * gaze_x + (v_x[i] * (1/ (grid_size * 3.0))) ) )
            y_grid = int(np.floor( gaze_label_size * gaze_y + (v_y[i] * (1/ (grid_size * 3.0))) ) )

            if x_grid < 0:
                x_grid = 0
            elif x_grid > 4:
                x_grid = 4
            if y_grid < 0:
                y_grid = 0
            elif y_grid > 4:
                y_grid = 4

            try:
                shifted_grids[i][y_grid][x_grid] = 1
            except:
                exit()

        shifted_grids = torch.from_numpy(shifted_grids).contiguous()

        shifted_grids = shifted_grids.view(1, 5, 25)


        if self.imshow:
            img.save("img_aug.jpg")
            face.save('face_aug.jpg')

        if self.transform is not None:
            img = self.transform(img)
            face = self.transform(face)

        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output

        if self.training == 'test':  # aggregated heatmap
            gaze_heatmap = chong_imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                         3,
                                                         type='Gaussian')

        else:
            # if gaze_inside:
            gaze_heatmap = chong_imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')
        # return
        if self.imshow:
            fig = plt.figure(111)
            img = 255 - chong_imutils.unnorm(img.numpy()) * 255
            img = np.clip(img, 0, 255)
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.imshow(imresize(gaze_heatmap, (self.input_size, self.input_size)), cmap='jet', alpha=0.3)
            plt.imshow(imresize(1 - head_channel.squeeze(0), (self.input_size, self.input_size)), alpha=0.2)
            plt.savefig('viz_aug.png')

        # intialize object_channel to all ones
        object_channel = torch.ones(1,224,224)

        if self.training == 'test':
            return img, face, head_channel, object_channel, gaze_heatmap, image_path, gaze_inside
        else:
            return img, face, head_point, gt_point


# class RetailGazeDataset(Dataset):
#     def __init__(self, root_dir, mat_file, training='train', include_path=False, input_size=224, output_size=64,
#                  imshow=False, use_gtbox=True):
#         assert (training in set(['train', 'test', 'test_prediction']))
#         self.root_dir = root_dir
#         self.mat_file = mat_file
#         self.training = training
#         self.include_path = include_path
#         self.input_size = input_size
#         self.output_size = output_size
#         self.imshow = imshow
#         self.transform = _get_transform(input_size)
#         self.transform2 = _get_transform2()
#         self.use_gtbox = use_gtbox

#         with open(mat_file, 'rb') as f:
#             self.data = pickle.load(f)
#             self.image_num = len(self.data)

#         print("Number of Images:", self.image_num)
#         logging.info('%s contains %d images' % (self.mat_file, self.image_num))

#     def create_mask(self, seg_idx, width=640, height=480):
#         seg_idx = seg_idx.astype(np.int64)
#         seg_mask = np.zeros((height, width)).astype(np.uint8)
#         for i in range(seg_idx.shape[0]):
#             seg_mask[seg_idx[i, 1], seg_idx[i, 0]] = 255
#         return seg_mask

#     def __len__(self):
#         return self.image_num

#     def __getitem__(self, idx):

#         gaze_inside = True
#         data = self.data[idx]
#         image_path = data['filename']
#         image_path = os.path.join(self.root_dir, image_path)

#         image_path = image_path.replace('\\', '/')
#         img = Image.open(image_path)
#         img = img.convert('RGB')
#         width, height = img.size
#         # Get bounding boxes and class labels as well as gt index for gazed object
#         gt_bboxes, gt_labels = np.zeros(1), np.zeros(1)
#         gt_labels = np.expand_dims(gt_labels, axis=0)
#         hbox = data['ann']['hbox']

#         x_min, y_min, x_max, y_max = hbox
#         # Crop the face
#         face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
#         # Eyes_loc
#         eye_x = int(((x_min+x_max)/2))
#         eye_y = int(((y_min+y_max)/2))
#         eye_x = int((eye_x/640)*224)
#         eye_y = int((eye_y/480)*224)
#         eyes_loc = [eye_x, eye_y]

#         if self.imshow:
#             img.save("img_aug.jpg")
#             face.save('face_aug.jpg')

#         if self.transform is not None:
#             img = self.transform(img)
#             face = self.transform2(face)
#         if self.training == 'test':
#             return img, face, eyes_loc, image_path
#         elif self.training == 'test_prediction':
#             return img, face, gt_bboxes, gt_labels
#         else:
#             return img, face


class RetailGaze(Dataset):
        def __init__(self, root_dir, mat_file, training='train', include_path=False, input_size=224, output_size=64, imshow = False, use_gtbox=False):
            assert (training in set(['train', 'test', 'inference']))
            self.root_dir = root_dir
            self.mat_file = mat_file
            self.training = training
            self.include_path = include_path
            self.input_size = input_size
            self.output_size = output_size
            self.imshow = imshow
            self.transform = _get_transform(input_size)
            self.transform2 = _get_transform2()
            self.use_gtbox= use_gtbox

            with open(mat_file, 'rb') as f:
                self.data = pickle.load(f)
                self.image_num = len(self.data)

            print("Number of Images:", self.image_num)
            # logging.info('%s contains %d images' % (self.mat_file, self.image_num))

        def __len__(self):
            return self.image_num

        def __getitem__(self, idx):
            gaze_inside = True
            data = self.data[idx]
            image_path = data['filename']
            image_path = os.path.join(self.root_dir, image_path)

            # eye = [float(data['hx'])/640, float(data['hy'])/480]
            gaze = [float(data['gaze_cx'])/640, float(data['gaze_cy'])/480]
            # eyess = np.array([eye[0],eye[1]]).astype(np.float)
            gaze_x, gaze_y = gaze

            image_path = image_path.replace('\\', '/')
            img = Image.open(image_path)
            img = img.convert('RGB')
            width, height = img.size
            #Get bounding boxes and class labels as well as gt index for gazed object
            gt_bboxes, gt_labels = np.zeros(1), np.zeros(1)
            gt_labels = np.expand_dims(gt_labels, axis=0)
            # gaze_idx = np.copy(data['gazeIdx']).astype(np.int64) #index of gazed object
            # gaze_class = np.copy(data['gaze_item']).astype(np.int64) #class of gazed object
            if self.use_gtbox:
                gt_bboxes = np.copy(data['ann']['bboxes']) / [640, 480, 640, 480]
                gt_labels = np.copy(data['ann']['labels'])
                # gtbox = gt_bboxes[gaze_idx]
            hbox = np.copy(data['ann']['hbox'])
            x_min, y_min, x_max, y_max = hbox
            face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
            head_x=((x_min+x_max)/2)/640
            head_y=((y_min+y_max)/2)/480
            head = np.array([head_x, head_y])
            # centers = (boxes2centers(gt_bboxes)*[224,224]).astype(int)
            gt_label = np.array([gaze_x, gaze_y])
            head_box = np.array([x_min/640, y_min/480, x_max/640, y_max/480])

            #plot gaze point
            # i = cv2.imread(image_path)
            # i=cv2.resize(i, (448, 448))
            # p=(gt_label*448).astype(np.int)
            # p2=(head*448).astype(np.int)
            # x,y=p
            # x2, y2=p2
            # i = cv2.circle(i, (x,y), radius=0, color=(0, 0, 255), thickness=4)
            # i = cv2.circle(i, (x2,y2), radius=0, color=(0, 0, 255), thickness=8)
            # cv2.imwrite('/home/primesh/Desktop/fig.jpg', i)
            # face.save('/home/primesh/Desktop/face.jpg')

            # segmentation mask layers
            # seg_path='/media/primesh/F4D0EA80D0EA49061/PROJECTS/FYP/Gaze detection/code/scripts/seg_mask'
            # masks=os.listdir(seg_path)
            # for mask in masks[:]:
            #     if not(mask.endswith('.png')):
            #         masks.remove(mask)
            # layers=[]
            # for mask in masks:
            #     layer=cv2.imread(seg_path+'/'+mask, 0)
            #     layers.append(layer)

            if self.imshow:
                img.save("img_aug.jpg")

            if self.transform is not None:
                img = self.transform(img)
                face = self.transform2(face)

            if self.training == 'test':
                return img, face, head, gt_label, head_box, image_path
            elif self.training == 'test_prediction':
                pass
            elif self.training == 'inference':
                # return img, face, head, gt_label, head_box, image_path, layers
                pass
            else:
                return img, face, head, gt_label, head_box, image_path

# class RetailGaze2(Dataset):
#         def __init__(self, root_dir, mat_file, training='train', include_path=False, input_size=224, output_size=64, imshow = False, use_gtbox=False):
#             assert (training in set(['train', 'test']))
#             self.root_dir = root_dir
#             self.mat_file = mat_file
#             self.training = training
#             self.include_path = include_path
#             self.input_size = input_size
#             self.output_size = output_size
#             self.imshow = imshow
#             self.transform = _get_transform(input_size)
#             self.use_gtbox= use_gtbox

#             with open(mat_file, 'rb') as f:
#                 self.data = pickle.load(f)
#                 self.image_num = len(self.data)

#             print("Number of Images:", self.image_num)
#             # logging.info('%s contains %d images' % (self.mat_file, self.image_num))

#         def __len__(self):
#             return self.image_num

#         def __getitem__(self, idx):
#             gaze_inside = True
#             data = self.data[idx]
#             image_path = data['filename']
#             image_path = os.path.join(self.root_dir, image_path)

#             gaze = [float(data['gaze_cx'])/640, float(data['gaze_cy'])/480]
#             # eyess = np.array([eye[0],eye[1]]).astype(np.float)
#             gaze_x, gaze_y = gaze

#             image_path = image_path.replace('\\', '/')
#             img = Image.open(image_path)
#             img = img.convert('RGB')
#             width, height = img.size
#             #Get bounding boxes and class labels as well as gt index for gazed object
#             gt_bboxes, gt_labels = np.zeros(1), np.zeros(1)
#             gt_labels = np.expand_dims(gt_labels, axis=0)
#             width, height = img.size
#             hbox = np.copy(data['ann']['hbox'])
#             x_min, y_min, x_max, y_max = hbox
#             head_x=((x_min+x_max)/2)/640
#             head_y=((y_min+y_max)/2)/480
#             eye = np.array([head_x, head_y])
#             eye_x, eye_y = eye
#             k = 0.1
#             x_min = (eye_x - 0.15) * width
#             y_min = (eye_y - 0.15) * height
#             x_max = (eye_x + 0.15) * width
#             y_max = (eye_y + 0.15) * height
#             if x_min < 0:
#                 x_min = 0
#             if y_min < 0:
#                 y_min = 0
#             if x_max < 0:
#                 x_max = 0
#             if y_max < 0:
#                 y_max = 0
#             if x_min > 1:
#                 x_min = 1
#             if y_min > 1:
#                 y_min = 1
#             if x_max > 1:
#                 x_max = 1
#             if y_max > 1:
#                 y_max = 1
#             x_min -= k * abs(x_max - x_min)
#             y_min -= k * abs(y_max - y_min)
#             x_max += k * abs(x_max - x_min)
#             y_max += k * abs(y_max - y_min)
#             x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
#             if self.use_gtbox:
#                 gt_bboxes = np.copy(data['ann']['bboxes']) / [640, 480, 640, 480]
#                 gt_labels = np.copy(data['ann']['labels'])
#                 # gtbox = gt_bboxes[gaze_idx]
#             face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
#             head_x=((x_min+x_max)/2)/640
#             head_y=((y_min+y_max)/2)/480
#             head = np.array([head_x, head_y])

#             gt_label = np.array([gaze_x, gaze_y])
#             head_box = np.array([x_min/640, y_min/480, x_max/640, y_max/480])

#             #plot gaze point
#             # i = cv2.imread(image_path)
#             # i=cv2.resize(i, (448, 448))
#             # p=(gt_label*448).astype(np.int)
#             # p2=(head*448).astype(np.int)
#             # x,y=p
#             # x2, y2=p2
#             # i = cv2.circle(i, (x,y), radius=0, color=(0, 0, 255), thickness=4)
#             # i = cv2.circle(i, (x2,y2), radius=0, color=(0, 0, 255), thickness=8)
#             # cv2.imwrite('/home/primesh/Desktop/fig.jpg', i)
#             # face.save('/home/primesh/Desktop/face.jpg')

#             if self.imshow:
#                 img.save("img_aug.jpg")

#             if self.transform is not None:
#                 img = self.transform(img)
#                 face = self.transform(face)

#             if self.training == 'test':
#                 return img, face, head, gt_label, head_box, image_path
#             elif self.training == 'test_prediction':
#                 pass
#             else:
#                 return img, face, head, gt_label, head_box, image_path

# class RetailGaze1(Dataset):
#         def __init__(self, root_dir, mat_file, training='train', include_path=False, input_size=224, output_size=64, imshow = False, use_gtbox=False):
#             assert (training in set(['train', 'test']))
#             self.root_dir = root_dir
#             self.mat_file = mat_file
#             self.training = training
#             self.include_path = include_path
#             self.input_size = input_size
#             self.output_size = output_size
#             self.imshow = imshow
#             self.transform = _get_transform(input_size)
#             self.use_gtbox= use_gtbox

#             with open(mat_file, 'rb') as f:
#                 self.data = pickle.load(f)
#                 self.image_num = len(self.data)

#             print("Number of Images:", self.image_num)
#             # logging.info('%s contains %d images' % (self.mat_file, self.image_num))

#         def __len__(self):
#             return self.image_num

#         def __getitem__(self, idx):
#             gaze_inside = True
#             data = self.data[idx]
#             image_path = data['filename']
#             image_path = os.path.join(self.root_dir, image_path)

#             gaze = [float(data['gaze_cx'])/640, float(data['gaze_cy'])/480]
#             # eyess = np.array([eye[0],eye[1]]).astype(np.float)
#             gaze_x, gaze_y = gaze

#             image_path = image_path.replace('\\', '/')
#             img = Image.open(image_path)
#             img = img.convert('RGB')
#             width, height = img.size
#             #Get bounding boxes and class labels as well as gt index for gazed object
#             gt_bboxes, gt_labels = np.zeros(1), np.zeros(1)
#             gt_labels = np.expand_dims(gt_labels, axis=0)
#             width, height = img.size
#             hbox = np.copy(data['ann']['hbox'])
#             x_min, y_min, x_max, y_max = hbox
#             head_x=((x_min+x_max)/2)/640
#             head_y=((y_min+y_max)/2)/480
#             eye = np.array([head_x, head_y])
#             eye_x, eye_y = eye
#             k = 0.1
#             x_min = (eye_x - 0.15) * width
#             y_min = (eye_y - 0.15) * height
#             x_max = (eye_x + 0.15) * width
#             y_max = (eye_y + 0.15) * height
#             if x_min < 0:
#                 x_min = 0
#             if y_min < 0:
#                 y_min = 0
#             if x_max < 0:
#                 x_max = 0
#             if y_max < 0:
#                 y_max = 0
#             if x_min > 1:
#                 x_min = 1
#             if y_min > 1:
#                 y_min = 1
#             if x_max > 1:
#                 x_max = 1
#             if y_max > 1:
#                 y_max = 1
#             x_min -= k * abs(x_max - x_min)
#             y_min -= k * abs(y_max - y_min)
#             x_max += k * abs(x_max - x_min)
#             y_max += k * abs(y_max - y_min)
#             x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
#             if self.use_gtbox:
#                 gt_bboxes = np.copy(data['ann']['bboxes']) / [640, 480, 640, 480]
#                 gt_labels = np.copy(data['ann']['labels'])
#                 # gtbox = gt_bboxes[gaze_idx]
#             face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
#             head_x=((x_min+x_max)/2)/640
#             head_y=((y_min+y_max)/2)/480
#             head = np.array([head_x, head_y])

#             gt_label = np.array([gaze_x, gaze_y])
#             head_box = np.array([x_min/640, y_min/480, x_max/640, y_max/480])

#             if self.imshow:
#                 img.save("img_aug.jpg")

#             if self.transform is not None:
#                 img = self.transform(img)
#                 face = self.transform(face)

#             if self.training == 'test':
#                 return img, face, head, gt_label, head_box, image_path
#             elif self.training == 'test_prediction':
#                 pass
#             else:
#                 return img, face, head, gt_label, head_box, image_path


# class RetailGaze2(Dataset):
#         def __init__(self, root_dir, mat_file, training='train', include_path=False, input_size=224, output_size=64, imshow = False, use_gtbox=False):
#             assert (training in set(['train', 'test']))
#             self.root_dir = root_dir
#             self.mat_file = mat_file
#             self.training = training
#             self.include_path = include_path
#             self.input_size = input_size
#             self.output_size = output_size
#             self.imshow = imshow
#             self.transform = _get_transform(input_size)
#             self.use_gtbox= use_gtbox

#             with open(mat_file, 'rb') as f:
#                 self.data = pickle.load(f)
#                 self.image_num = len(self.data)

#             print("Number of Images:", self.image_num)
#             # logging.info('%s contains %d images' % (self.mat_file, self.image_num))

#         def __len__(self):
#             return self.image_num

#         def __getitem__(self, idx):
#             gaze_inside = True
#             data = self.data[idx]
#             image_path = data['filename']
#             image_path = os.path.join(self.root_dir, image_path)

#             gaze = np.array([float(data['gaze_cx'])/640, float(data['gaze_cy'])/480])
#             # eyess = np.array([eye[0],eye[1]]).astype(np.float)
#             gaze_x, gaze_y = gaze

#             image_path = image_path.replace('\\', '/')
#             img = Image.open(image_path)
#             img = img.convert('RGB')
#             width, height = img.size
#             #Get bounding boxes and class labels as well as gt index for gazed object
#             gt_bboxes, gt_labels = np.zeros(1), np.zeros(1)
#             gt_labels = np.expand_dims(gt_labels, axis=0)
#             width, height = img.size
#             hbox = np.copy(data['ann']['hbox'])
#             x_min, y_min, x_max, y_max = hbox
#             head_x=((x_min+x_max)/2)/640
#             head_y=((y_min+y_max)/2)/480
#             eye = np.array([head_x, head_y])
#             eye_x, eye_y = eye
#             k = 0.1
#             x_min = (eye_x - 0.15) * width
#             y_min = (eye_y - 0.15) * height
#             x_max = (eye_x + 0.15) * width
#             y_max = (eye_y + 0.15) * height
#             if x_min < 0:
#                 x_min = 0
#             if y_min < 0:
#                 y_min = 0
#             if x_max < 0:
#                 x_max = 0
#             if y_max < 0:
#                 y_max = 0
#             x_min -= k * abs(x_max - x_min)
#             y_min -= k * abs(y_max - y_min)
#             x_max += k * abs(x_max - x_min)
#             y_max += k * abs(y_max - y_min)
#             x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

#             mask_path = image_path.split('/')[:-1]
#             mask_path = '/'.join(mask_path) + "/combined.png"
#             mask = cv2.imread(mask_path,0)
#             mask = cv2.resize(mask, (224,224), interpolation = cv2.INTER_AREA)
#             mask_tensor = image_to_tensor(mask)
#             object_channel = mask_tensor/255.0

#             head_box = np.array([x_min/640, y_min/480, x_max/640, y_max/480])
#             #segmentation layers
#             seg_path_temp = image_path.split('/')[:-1]
#             seg_path = '/'.join(seg_path_temp) + "/masks"
#             seg_folder = '/'.join(seg_path_temp[-2:]) + "/masks"
#             masks=os.listdir(seg_path)
#             for segmask in masks[:]:
#                 if not(segmask.endswith('.png')):
#                     segmask.remove(segmask)
#             layers=[]
#             for segmask in masks:
#                 layer=cv2.imread(seg_path+'/'+segmask, 0)
#                 layers.append(layer)

#             head_channel = chong_imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
#                                                         resolution=self.input_size, coordconv=False).unsqueeze(0)

#             # Crop the face
#             face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
#             grid_size = 5
#             gaze_label_size = 5
#             v_x = [0, 1, -1, 0, 0]
#             v_y = [0, 0, 0, -1, 1]


#             shifted_grids = np.zeros((grid_size, gaze_label_size, gaze_label_size))
#             for i in range(5):

#                 x_grid = int(np.floor( gaze_label_size * gaze_x + (v_x[i] * (1/ (grid_size * 3.0))) ) )
#                 y_grid = int(np.floor( gaze_label_size * gaze_y + (v_y[i] * (1/ (grid_size * 3.0))) ) )

#                 if x_grid < 0:
#                     x_grid = 0
#                 elif x_grid > 4:
#                     x_grid = 4
#                 if y_grid < 0:
#                     y_grid = 0
#                 elif y_grid > 4:
#                     y_grid = 4

#                 try:
#                     shifted_grids[i][y_grid][x_grid] = 1
#                 except:
#                     exit()

#             shifted_grids = torch.from_numpy(shifted_grids).contiguous()

#             shifted_grids = shifted_grids.view(1, 5, 25)
#             gaze_final = np.ones(100)
#             gaze_final *= -1
#             gaze_final[0] = gaze_x
#             gaze_final[1] = gaze_y
#             eyes_loc_size = 13
#             eyes_loc = np.zeros((eyes_loc_size, eyes_loc_size))
#             eyes_loc[int(np.floor(eyes_loc_size * eye_y))][int(np.floor(eyes_loc_size * eye_x))] = 1

#             eyes_loc = torch.from_numpy(eyes_loc).contiguous()
#             if self.imshow:
#                 img.save("img_aug.jpg")

#             if self.transform is not None:
#                 img = self.transform(img)
#                 face = self.transform(face)

#             if self.training == 'test':
#                 return img, face, head_channel, head_box, object_channel,gaze_final,eye,gt_bboxes,gt_labels, gaze, image_path, layers, seg_folder, masks
#             else:
#                 return img, face, head_channel, object_channel,eyes_loc, image_path, gaze_inside , shifted_grids, gaze_final


# class RetailGaze(Dataset):
#         def __init__(self, root_dir, mat_file, training='train', include_path=False, input_size=224, output_size=64, imshow = False, use_gtbox=False):
#             assert (training in set(['train', 'test']))
#             self.root_dir = root_dir
#             self.mat_file = mat_file
#             self.training = training
#             self.include_path = include_path
#             self.input_size = input_size
#             self.output_size = output_size
#             self.imshow = imshow
#             self.transform = _get_transform(input_size)
#             self.use_gtbox= use_gtbox

#             with open(mat_file, 'rb') as f:
#                 self.data = pickle.load(f)
#                 self.image_num = len(self.data)

#             print("Number of Images:", self.image_num)
#             # logging.info('%s contains %d images' % (self.mat_file, self.image_num))

#         def __len__(self):
#             return self.image_num

#         def __getitem__(self, idx):
#             gaze_inside = True
#             data = self.data[idx]
#             image_path = data['filename']
#             image_path = os.path.join(self.root_dir, image_path)

#             gaze = [float(data['gaze_cx'])/640, float(data['gaze_cy'])/480]
#             # eyess = np.array([eye[0],eye[1]]).astype(np.float)
#             gaze_x, gaze_y = gaze

#             image_path = image_path.replace('\\', '/')
#             img = Image.open(image_path)
#             img = img.convert('RGB')
#             width, height = img.size
#             #Get bounding boxes and class labels as well as gt index for gazed object
#             gt_bboxes, gt_labels = np.zeros(1), np.zeros(1)
#             gt_labels = np.expand_dims(gt_labels, axis=0)
#             width, height = img.size
#             hbox = np.copy(data['ann']['hbox'])
#             x_min, y_min, x_max, y_max = hbox
#             head_x=((x_min+x_max)/2)/640
#             head_y=((y_min+y_max)/2)/480
#             eye = np.array([head_x, head_y])
#             eye_x, eye_y = eye
#             k = 0.1
#             x_min = (eye_x - 0.15) * width
#             y_min = (eye_y - 0.15) * height
#             x_max = (eye_x + 0.15) * width
#             y_max = (eye_y + 0.15) * height
#             if x_min < 0:
#                 x_min = 0
#             if y_min < 0:
#                 y_min = 0
#             if x_max < 0:
#                 x_max = 0
#             if y_max < 0:
#                 y_max = 0
#             if x_min > 1:
#                 x_min = 1
#             if y_min > 1:
#                 y_min = 1
#             if x_max > 1:
#                 x_max = 1
#             if y_max > 1:
#                 y_max = 1
#             x_min -= k * abs(x_max - x_min)
#             y_min -= k * abs(y_max - y_min)
#             x_max += k * abs(x_max - x_min)
#             y_max += k * abs(y_max - y_min)
#             x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
#             if self.use_gtbox:
#                 gt_bboxes = np.copy(data['ann']['bboxes']) / [640, 480, 640, 480]
#                 gt_labels = np.copy(data['ann']['labels'])
#                 # gtbox = gt_bboxes[gaze_idx]
#             face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
#             head_x=((x_min+x_max)/2)/640
#             head_y=((y_min+y_max)/2)/480
#             head = np.array([head_x, head_y])

#             gt_label = np.array([gaze_x, gaze_y])
#             head_box = np.array([x_min/640, y_min/480, x_max/640, y_max/480])

#             if self.imshow:
#                 img.save("img_aug.jpg")

#             if self.transform is not None:
#                 img = self.transform(img)
#                 face = self.transform(face)

#             if self.training == 'test':
#                 return img, face, head, gt_label, head_box, image_path
#             elif self.training == 'test_prediction':
#                 pass
#             else:
#                 return img, face, head, gt_label, head_box, image_path

