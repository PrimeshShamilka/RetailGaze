import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import pickle
import numpy as np 
from PIL import Image 
import os

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
        assert (training in set(['train', 'test']))
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
        
        centers = (boxes2centers(gt_bboxes)*[224,224]).astype(int)
        gt_label = centers[gaze_idx,:]    
        
        if self.imshow:
            img.save("img_aug.jpg")

        if self.transform is not None:
            img = self.transform(img)

        if self.training == 'test':
            pass
        elif self.training == 'test_prediction':
            pass
        else:
            return img, gt_label, image_path