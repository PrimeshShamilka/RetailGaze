import time
import torch
import torch.optim as optim
import numpy as np
from early_stopping_pytorch.pytorchtools import EarlyStopping
from tqdm import tqdm
import torch.nn as nn
import warnings
import csv

# warnings.filterwarnings('error')

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

def get_bb_binary(box):
    xmin, ymin, xmax, ymax = box
    b = np.zeros((224, 224), dtype='float32')
    for j in range(ymin, ymax):
        for k in range(xmin, xmax):
            b[j][k] = 1
    return b

def train_base_model(model,train_data_loader,validation_data_loader, criterion, optimizer, logger, writer ,num_epochs=5,patience=10):
    since = time.time()
    n_total_steps = len(train_data_loader)
    n_total_steps_val = len(validation_data_loader)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(num_epochs):

        model.train()  # Set model to training mode

        running_loss = []
        running_loss2 = []
        validation_loss = []

        for i, (img, gt_label, img_path) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            optimizer.zero_grad()
            pred = model(image)
            loss = criterion(pred, gt_label)
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
        for i, (img, gt_label, img_path) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            optimizer.zero_grad()
            pred = model(image)
            loss = criterion(pred, gt_label)
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


def test_base_model(model, test_data_loader, logger, test_depth=True, save_output=False):
    model.eval()
    angle_error = []
    with torch.no_grad():
        for img, gt_label, img_path in test_data_loader:
            image =  img.cuda()
            pred = model(image)
            label = gt_label
            gaze = pred
            for i in range(img.shape[0]):
                if test_depth == True:
                    ae = np.dot(gaze[i,:],label[i,:])/(np.sqrt(np.dot(label[i,:],label[i,:])*np.dot(gaze[i,:],gaze[i,:]))+np.finfo(np.float32).eps)
                else:
                    ae = np.dot(gaze[i,:2],label[i,:2])/(np.sqrt(np.dot(label[i,:2],label[i,:2])*np.dot(gaze[i,:2], gaze[i,:2]))+np.finfo(np.float32).eps)
                ae = np.arccos(np.maximum(np.minimum(ae,1.0),-1.0)) * 180 / np.pi
                angle_error.append(ae)
        angle_error = np.mean(np.array(angle_error),axis=0)
    print(angle_error)