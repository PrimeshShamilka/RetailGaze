
import time
import torch
import torch.optim as optim
import numpy as np
from early_stopping_pytorch.pytorchtools import EarlyStopping
from tqdm import tqdm
import torch.nn as nn
import warnings
import csv

class GazeOptimizer():
    def __init__(self, net, initial_lr, weight_decay=1e-6):

        self.INIT_LR = initial_lr
        self.WEIGHT_DECAY = weight_decay
        self.optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)

    def getOptimizer(self, epoch, decay_epoch=15):

        if epoch < decay_epoch:
            lr = self.INIT_LR
        else:
            lr = self.INIT_LR / 10

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = self.WEIGHT_DECAY

        return self.optimizer

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


def get_bb_binary(box):
    xmin, ymin, xmax, ymax = box
    b = np.zeros((224, 224), dtype='float32')
    for j in range(ymin, ymax):
        for k in range(xmin, xmax):
            b[j][k] = 1
    return b

    
def train_face3d(model,train_data_loader,validation_data_loader, criterion, optimizer, logger, writer ,num_epochs=5,patience=10):
    since = time.time()
    n_total_steps = len(train_data_loader)
    n_total_steps_val = len(validation_data_loader)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(num_epochs):

        model.train()  # Set model to training mode

        running_loss = []
        running_loss2 = []
        validation_loss = []
        for i, (img, face, head, gt_label, head_box, image_path) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            gt_label = gt_label
            head = head
            optimizer.zero_grad()
            gaze,depth = model(image,face)
            depth =  depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                hbox = head_box[i].cpu().detach().numpy()*224
                hbox = hbox.astype(int)
                gt = (gt_label[i] - head[i])/224
                label[i,0] = gt[0]
                label[i,1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(hbox))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary==1)
                gt_depth = depth[i][0][int(gt_label[i][0]*224)-1][int(gt_label[i][1]*224)-1]
                label[i, 2] = (gt_depth - head_depth)
                # label[i,2] = (depth[i,:,gt_label[i,0],gt_label[i,1]] - depth[i,:,head[i,0],head[i,1]])
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            running_loss2.append(loss.item())
            if i % 10 == 9:
                logger.info('%s'%(str(np.mean(running_loss))))
                # writer.add_scalar('training_loss',np.mean(running_loss),epoch*n_total_steps+i)
                running_loss = []

        with open('training_loss.cvs', 'a') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch * n_total_steps, str(np.mean(running_loss2))])
        running_loss2 = []

         # Validation
        model.eval()
        for i, (img, face, head, gt_label, head_box, image_path) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            gt_label = gt_label
            head = head
            optimizer.zero_grad()
            gaze,depth = model(image,face)
            depth =  depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                hbox = head_box[i].cpu().detach().numpy()*224
                hbox = hbox.astype(int)
                gt = (gt_label[i] - head[i])/224
                label[i,0] = gt[0]
                label[i,1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(hbox))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary==1)
                gt_depth = depth[i][0][int(gt_label[i][0]*224)-1][int(gt_label[i][1]*224)-1]
                label[i, 2] = (gt_depth - head_depth)
                # label[i,2] = (depth[i,:,gt_label[i,0],gt_label[i,1]] - depth[i,:,head[i,0],head[i,1]])
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)
            validation_loss.append(loss.item())
        val_loss = np.mean(validation_loss)

        logger.info('%s'%(str(val_loss)))
        writer.add_scalar('validation_loss',val_loss,epoch)
        with open ('validation_loss.cvs', 'a') as f:
            writer_csv2 = csv.writer(f)
            writer_csv2.writerow([epoch*n_total_steps, str(val_loss)])
        validation_loss = []

        early_stopping(val_loss, model, optimizer, epoch, logger)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model


def test_face3d(model, test_data_loader, logger, test_depth=True, save_output=False):
    model.eval()
    angle_error = []
    l2=[]
    with torch.no_grad():
        for img, face, head, gt_label, head_box, image_path in test_data_loader: 
            image =  img.cuda()
            face = face.cuda()
            gaze,depth = model(image,face)
            max_depth = torch.max(depth)
            depth = depth / max_depth
            depth =  depth.cpu()
            gaze =  gaze.cpu().data.numpy()
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                hbox = head_box[i].cpu().detach().numpy()*224
                hbox = hbox.astype(int)
                gt = (gt_label[i] - head[i])/224
                label[i,0] = gt[0]
                label[i,1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(hbox))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary == 1)
                gt_depth = depth[i][0][int(gt_label[i][0]*224)-1][int(gt_label[i][1]*224)-1]
                label[i, 2] = (gt_depth - head_depth)
            for i in range(img.shape[0]):
                if test_depth == True:
                    ae = np.dot(gaze[i,:],label[i,:])/(np.sqrt(np.dot(label[i,:],label[i,:])*np.dot(gaze[i,:],gaze[i,:]))+np.finfo(np.float32).eps)
                else:
                    ae = np.dot(gaze[i,:2],label[i,:2])/(np.sqrt(np.dot(label[i,:2],label[i,:2])*np.dot(gaze[i,:2], gaze[i,:2]))+np.finfo(np.float32).eps)
                ae = np.arccos(np.maximum(np.minimum(ae,1.0),-1.0)) * 180 / np.pi
                angle_error.append(ae)

                # L2 dist
                euclid_dist = np.sqrt(np.power((label[i, 0] - gaze[i, 0]), 2) + np.power((label[i, 1] - gaze[i, 1]), 2))
                l2.append(euclid_dist)

        angle_error = np.mean(np.array(angle_error),axis=0)
        l2_dist = np.mean(np.array(l2), axis=0)
    print(angle_error, l2_dist)
