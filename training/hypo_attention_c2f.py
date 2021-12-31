import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import time
import models.__init__ as init
import utils
import sys
from sklearn.metrics import roc_auc_score
import numpy as np

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from early_stopping_pytorch.pytorchtools import EarlyStopping
import csv

def euclid_dist(output, target):
    output = output

    predy = ((output / 227.0) / 227.0)
    predx = ((output % 227.0) / 227.0)

    x_list = []
    y_list = []
    for j in range(100):
        ground_x = target[2 * j]
        ground_y = target[2 * j + 1]

        if ground_x == -1 or ground_y == -1:
            break

        x_list.append(ground_x)
        y_list.append(ground_y)

    x_truth = np.mean(x_list)
    y_truth = np.mean(y_list)

    total = np.sqrt(np.power((x_truth - predx), 2) + np.power((y_truth - predy), 2))
    return total, np.array([predx, predy])


def calc_ang_err(output, target, eyes):
    total = 0

    output = output.cpu()
    target = target

    predy = ((output / 227.0) / 227.0)
    predx = ((output % 227.0) / 227.0)
    pred_point = np.array([predx, predy])

    eye_point = eyes

    x_list = []
    y_list = []
    for j in range(100):
        ground_x = target[2 * j]
        ground_y = target[2 * j + 1]

        if ground_x == -1 or ground_y == -1:
            break

        x_list.append(ground_x)
        y_list.append(ground_y)

    x_truth = np.mean(x_list)
    y_truth = np.mean(y_list)

    gt_point = np.stack([x_truth, y_truth])

    pred_dir = pred_point - eye_point
    gt_dir = gt_point - eye_point

    norm_pred = (pred_dir[0] ** 2 + pred_dir[1] ** 2) ** 0.5
    norm_gt = (gt_dir[0] ** 2 + gt_dir[1] ** 2) ** 0.5

    cos_sim = (pred_dir[0] * gt_dir[0] + pred_dir[1] * gt_dir[1]) / \
              (norm_gt * norm_pred + 1e-6)
    cos_sim = np.maximum(np.minimum(cos_sim, 1.0), -1.0)
    ang_error = np.arccos(cos_sim) * 180 / np.pi

    return ang_error

# Loss function used is cross entropy
criterion = nn.NLLLoss().cuda()


def train(model, train_data_loader, criterion, optimizer, logger, writer, num_epochs=5, ):
    since = time.time()
    n_total_steps = len(train_data_loader)
    for epoch in range(num_epochs + 1):

        mse_loss = nn.MSELoss(reduce=False)  # not reducing in order to ignore outside cases
        loss_amp_factor = 10000  # multiplied to the loss to prevent underflow

        optimizer.zero_grad()
        model.train()  # Set model to training mode

        running_loss = []
        # print("Training in progress ...")


        # Iterate over data.
        for i, (img, face, head_channel, object_channel, eyes_loc, gaze_heatmap, image_path, gaze_inside, shifted_targets,
                gaze_final) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            image = img.cuda()
            head_channel = head_channel.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()
            shifted_targets = shifted_targets.cuda().squeeze()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            outputs, gaze_heatmap_pred = model(image, face, head_channel, object_channel)
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)

            # classification loss
            total_loss = criterion(outputs[0], shifted_targets[:, 0, :].max(1)[1])
            for j in range(1, len(outputs)):
                total_loss += criterion(outputs[j], shifted_targets[:, j, :].max(1)[1])

            total_loss = total_loss / (len(outputs) * 1.0)

            # regression loss
            # l2 loss computed only for inside case
            l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmap.cuda()) * loss_amp_factor
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.mean(l2_loss, dim=1)
            gaze_inside = gaze_inside.cuda().to(torch.float)
            l2_loss = torch.mul(l2_loss, gaze_inside)  # zero out loss when it's out-of-frame gaze case
            l2_loss = torch.sum(l2_loss) / torch.sum(gaze_inside)

            print(gaze_inside, total_loss, l2_loss)
            total_loss += l2_loss

            total_loss.backward()
            optimizer.step()

            inputs_size = image.size(0)

            running_loss.append(total_loss.item())
            if i % 50 == 49:
                logger.info('%s' % (str(np.mean(running_loss))))
                writer.add_scalar('training_loss', np.mean(running_loss), epoch * n_total_steps + i)
                running_loss = []
        # for name, weight in model.named_parameters():
        #     writer.add_histogram(name, weight, epoch)
        #     writer.add_histogram(f'{name}.grad', weight.grad, epoch)
    return model

# GOO dataset
def train_with_early_stopping(model, train_data_loader, valid_data_loader, criterion, optimizer, logger, writer, num_epochs=5, patience=5):

    # initialize the early_stopping object
    # patience = 5
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    since = time.time()
    n_total_steps = len(train_data_loader)
    for epoch in range(num_epochs + 1):

        mse_loss = nn.MSELoss(reduce=False)  # not reducing in order to ignore outside cases
        loss_amp_factor = 10  # multiplied to the loss to prevent underflow

        optimizer.zero_grad()
        model.train()  # Set model to training mode

        running_loss = []
        running_loss2 = []
        valid_losses = []
        avg_valid_losses = []
        # print("Training in progress ...")

        # Iterate over data.
        for i, (img, face, head_channel, object_channel, eyes_loc, gaze_heatmap, image_path, gaze_inside, shifted_targets,
                gaze_final) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            image = img.cuda()
            head_channel = head_channel.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()
            shifted_targets = shifted_targets.cuda().squeeze()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            outputs, gaze_heatmap_pred = model(image, face, head_channel, object_channel)
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)

            # classification loss
            total_loss = criterion(outputs[0], shifted_targets[:, 0, :].max(1)[1])
            for j in range(1, len(outputs)):
                total_loss += criterion(outputs[j], shifted_targets[:, j, :].max(1)[1])

            total_loss = total_loss / (len(outputs) * 1.0)

            # regression loss
            # l2 loss computed only for inside case
            l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmap.cuda()) * loss_amp_factor
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.mean(l2_loss, dim=1)
            gaze_inside = gaze_inside.cuda().to(torch.float)
            l2_loss = torch.mul(l2_loss, gaze_inside)  # zero out loss when it's out-of-frame gaze case
            l2_loss = torch.sum(l2_loss) / torch.sum(gaze_inside)

            # print(gaze_inside, total_loss, l2_loss)
            total_loss += l2_loss

            total_loss.backward()
            optimizer.step()

            inputs_size = image.size(0)

            running_loss.append(total_loss.item())
            running_loss2.append(total_loss.item())
            if i % 50 == 49:
                logger.info('%s' % (str(np.mean(running_loss))))
                writer.add_scalar('training_loss', np.mean(running_loss), epoch * n_total_steps + i)
                running_loss = []

        with open('training_loss.cvs', 'a') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch * n_total_steps, str(np.mean(running_loss2))])
        running_loss2 = []

        # validate the model
        model.eval()
        for i, (img, face, head_channel, object_channel, eyes_loc, gaze_heatmap, image_path, gaze_inside, shifted_targets,
                gaze_final) in tqdm(enumerate(valid_data_loader), total=len(valid_data_loader)):

            image = img.cuda()
            head_channel = head_channel.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()
            shifted_targets = shifted_targets.cuda().squeeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            outputs, gaze_heatmap_pred = model(image, face, head_channel, object_channel)
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)

            # classification loss
            total_loss = criterion(outputs[0], shifted_targets[:, 0, :].max(1)[1])
            for j in range(1, len(outputs)):
                total_loss += criterion(outputs[j], shifted_targets[:, j, :].max(1)[1])

            total_loss = total_loss / (len(outputs) * 1.0)

            # regression loss
            # l2 loss computed only for inside case
            l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmap.cuda()) * loss_amp_factor
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.mean(l2_loss, dim=1)
            gaze_inside = gaze_inside.cuda().to(torch.float)
            l2_loss = torch.mul(l2_loss, gaze_inside)  # zero out loss when it's out-of-frame gaze case
            l2_loss = torch.sum(l2_loss) / torch.sum(gaze_inside)

            # print(gaze_inside, total_loss, l2_loss)
            total_loss += l2_loss

            valid_losses.append(total_loss.item())

        valid_loss = np.average(valid_losses)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(num_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' + f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        valid_losses = []

        writer.add_scalar('validation_loss', valid_loss, epoch * n_total_steps)
        with open ('validation_loss.cvs', 'a') as f:
            writer_csv2 = csv.writer(f)
            writer_csv2.writerow([epoch*n_total_steps, str(valid_loss)])

        # early stopping detector
        early_stopping(valid_loss, model, optimizer, epoch, logger)
        if early_stopping.early_stop:
            print("Early stopping")
            break


        # for name, weight in model.named_parameters():
        #     writer.add_histogram(name, weight, epoch)
        #     writer.add_histogram(f'{name}.grad', weight.grad, epoch)

    return model

# Gaze dataset
def train_gazefollow_with_early_stopping(model, train_data_loader, valid_data_loader, criterion, optimizer, logger, writer, num_epochs=5, ):

    # initialize the early_stopping object
    patience = 5
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    since = time.time()
    n_total_steps = len(train_data_loader)
    for epoch in range(num_epochs + 1):

        mse_loss = nn.MSELoss(reduce=False)  # not reducing in order to ignore outside cases
        loss_amp_factor = 10  # multiplied to the loss to prevent underflow

        optimizer.zero_grad()
        model.train()  # Set model to training mode

        running_loss = []
        valid_losses = []
        avg_valid_losses = []
        # print("Training in progress ...")

        # Iterate over data.
        for i, (img, face, head_channel, object_channel, gaze_heatmap, image_path, gaze_inside, shifted_targets) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            image = img.cuda()
            head_channel = head_channel.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()
            shifted_targets = shifted_targets.cuda().squeeze()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            outputs, gaze_heatmap_pred = model(image, face, head_channel, object_channel)
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)

            # classification loss
            total_loss = criterion(outputs[0], shifted_targets[:, 0, :].max(1)[1])
            for j in range(1, len(outputs)):
                total_loss += criterion(outputs[j], shifted_targets[:, j, :].max(1)[1])

            total_loss = total_loss / (len(outputs) * 1.0)

            # regression loss
            # l2 loss computed only for inside case
            l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmap.cuda()) * loss_amp_factor
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.mean(l2_loss, dim=1)
            gaze_inside = gaze_inside.cuda().to(torch.float)
            l2_loss = torch.mul(l2_loss, gaze_inside)  # zero out loss when it's out-of-frame gaze case
            l2_loss = torch.sum(l2_loss) / torch.sum(gaze_inside)

            # print(gaze_inside, total_loss, l2_loss)
            total_loss += l2_loss

            total_loss.backward()
            optimizer.step()

            inputs_size = image.size(0)

            running_loss.append(total_loss.item())
            if i % 50 == 49:
                logger.info('%s' % (str(np.mean(running_loss))))
                writer.add_scalar('training_loss', np.mean(running_loss), epoch * n_total_steps + i)
                running_loss = []


        # validate the model
        model.eval()
        for i, (img, face, head_channel, object_channel, gaze_heatmap, image_path, gaze_inside, shifted_targets) in tqdm(enumerate(valid_data_loader), total=len(valid_data_loader)):
            image = img.cuda()
            head_channel = head_channel.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()
            shifted_targets = shifted_targets.cuda().squeeze()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            outputs, gaze_heatmap_pred = model(image, face, head_channel, object_channel)
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)

            # classification loss
            total_loss = criterion(outputs[0], shifted_targets[:, 0, :].max(1)[1])
            for j in range(1, len(outputs)):
                total_loss += criterion(outputs[j], shifted_targets[:, j, :].max(1)[1])

            total_loss = total_loss / (len(outputs) * 1.0)

            # regression loss
            # l2 loss computed only for inside case
            l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmap.cuda()) * loss_amp_factor
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.mean(l2_loss, dim=1)
            gaze_inside = gaze_inside.cuda().to(torch.float)
            l2_loss = torch.mul(l2_loss, gaze_inside)  # zero out loss when it's out-of-frame gaze case
            l2_loss = torch.sum(l2_loss) / torch.sum(gaze_inside)

            # print(gaze_inside, total_loss, l2_loss)
            total_loss += l2_loss

            valid_losses.append(total_loss.item())

        valid_loss = np.average(valid_losses)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(num_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' + f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        valid_losses = []

        writer.add_scalar('validation_loss', valid_loss, epoch * n_total_steps)

        # early stopping detector
        early_stopping(valid_loss, model, optimizer, epoch, logger)
        if early_stopping.early_stop:
            print("Early stopping")
            break


        # for name, weight in model.named_parameters():
        #     writer.add_histogram(name, weight, epoch)
        #     writer.add_histogram(f'{name}.grad', weight.grad, epoch)

    return model


class GazeOptimizer():

    def __init__(self, net, initial_lr, weight_decay=1e-6):

        self.INIT_LR = initial_lr
        self.WEIGHT_DECAY = weight_decay
        self.optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
        # self.optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.8)

    def getOptimizer(self, epoch, decay_epoch=15):

        if epoch < decay_epoch:
            lr = self.INIT_LR
        else:
            lr = self.INIT_LR / 10

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = self.WEIGHT_DECAY

        return self.optimizer


def cal_auc(target, pred_heatmap):
    x_list = []
    y_list = []
    for j in range(100):
        ground_x = target[2 * j]
        ground_y = target[2 * j + 1]

        if ground_x == -1 or ground_y == -1:
            break

        x_list.append(ground_x)
        y_list.append(ground_y)

    x_truth = np.mean(x_list)
    y_truth = np.mean(y_list)

    gt_point = np.stack([x_truth, y_truth])
    # score = cal_auc_per_point(gt_point, pred_heatmap)

    return cal_auc_per_point(gt_point, pred_heatmap)


def cal_auc_per_point(gt_point, pred_heatmap):
    '''
        Input: gt_point shape: (2,)
                pred_heatmap shape: (1,X,Y) or (X,Y)
        Returns: auc_score (1,) (float32)
    '''

    pred_heatmap = np.squeeze(pred_heatmap.cpu().numpy())
    pred_heatmap = cv2.resize(pred_heatmap, (5, 5))
    gt_heatmap = np.zeros((5, 5))

    x, y = list(map(int, gt_point * 5))
    gt_heatmap[y, x] = 1.0

    # score = roc_auc_score(gt_heatmap.reshape([-1]).astype(np.int32), pred_heatmap.reshape([-1]))

    return pred_heatmap, gt_heatmap


def boxes2centers(normalized_boxes):
    center_x = (normalized_boxes[:, 0] + normalized_boxes[:, 2]) / 2
    center_y = (normalized_boxes[:, 1] + normalized_boxes[:, 3]) / 2
    center_x = np.expand_dims(center_x, axis=1)
    center_y = np.expand_dims(center_y, axis=1)
    normalized_centers = np.hstack((center_x, center_y))
    return normalized_centers


def select_nearest_bbox(gazepoint, gt_bboxes):
    centers = boxes2centers(gt_bboxes)
    diff = centers - gazepoint
    l2dist = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    min_idx = np.argsort(l2dist)[:5]

    nearest_box = {
        'box': gt_bboxes[min_idx],
        'index': min_idx
    }
    return nearest_box

def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]) # left
    yA = max(boxA[1], boxB[1]) # top
    xB = min(boxA[2], boxB[2]) # right
    yB = min(boxA[3], boxB[3]) # down
    if xB < xA or yB < yA:
        return 0.0
    interArea = (xB - xA) * (yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = round(interArea / float(boxAArea + boxBArea - interArea), 2)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_bb_binary(gt_bboxes):
    bbox_l = []
    for i in range(gt_bboxes.shape[0]):
        bbox = (gt_bboxes[i][0] * 224).astype(int)
        xmin, ymin, xmax, ymax = bbox
        b = np.zeros((224, 224), dtype='float32')
        # assert xmin < xmax
        # assert ymin < ymax
        for j in range(ymin, ymax):
            for k in range(xmin, xmax):
                b[j][k] = 1
        bbox_l.append(b)
    return bbox_l

def test(model, test_data_loader, logger, save_output=False):
    model.eval()
    total_error = []
    all_gt_heat = []
    all_pred_heat = []

    percent_dists = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    PA_count = np.zeros((len(percent_dists)))

    all_gazepoints = []
    all_gtmap = []
    all_predmap = []

    with torch.no_grad():
        for i, (img, face, head_channel, object_channel, gaze_final, eye, gaze_idx, gt_bboxes,
                gt_labels) in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            image = img.cuda()
            head_channel = head_channel.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()

            outputs, raw_hm = model.raw_hm(image, face, head_channel, object_channel)

            pred_labels = outputs.max(1)[1]  # max function returns both values and indices. so max()[0] is values, max()[1] is indices
            inputs_size = image.size(0)

            for i in range(inputs_size):
                distval, f_point = euclid_dist(pred_labels.data.cpu()[i], gaze_final[i])
                ang_error = calc_ang_err(pred_labels.data.cpu()[i], gaze_final[i], eye.cpu()[i])
                # auc_score = cal_auc(ground_labels[i], raw_hm[i, :, :])
                predmap, gtmap = cal_auc(gaze_final[i], raw_hm[i, :, :])

                all_gazepoints.append(f_point)
                all_predmap.append(predmap)
                all_gtmap.append(gtmap)

                PA_count[np.array(percent_dists) > distval.item()] += 1

                total_error.append([distval, ang_error])

        l2, ang = np.mean(np.array(total_error), axis=0)

        all_gazepoints = np.vstack(all_gazepoints)
        all_predmap = np.stack(all_predmap).reshape([-1])
        all_gtmap = np.stack(all_gtmap).reshape([-1])
        auc = roc_auc_score(all_gtmap, all_predmap)

    if save_output:
        np.savez('predictions.npz', gazepoints=all_gazepoints)

    proxAcc = PA_count / len(test_data_loader.dataset)
    logger.info('proximate accuracy: %s' % str(proxAcc))
    logger.info('average error: %s' % str([auc, l2, ang]))

    return [auc, l2, ang]

# test with gaze object prediction
def test_gop(model, test_data_loader, logger, save_output=False):
    model.eval()
    total_error = []
    all_gt_heat = []
    all_pred_heat = []

    percent_dists = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    PA_count = np.zeros((len(percent_dists)))

    all_gazepoints = []
    all_gtmap = []
    all_predmap = []
    all_auc = []
    all_auc2 = []
    all_label = []
    all_iou = []
    all_overlap = []

    with torch.no_grad():
        count=0
        for img, face, head_channel, object_channel, gaze_final, eye, gaze_idx, gt_bboxes, gt_labels in test_data_loader:
            image = img.cuda()
            head_channel = head_channel.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()
            gt_bboxes = np.array(list(gt_bboxes))
            gt_labels = np.array(list(gt_labels))
            gaze_idx = np.array(gaze_idx)
            outputs, raw_hm = model.raw_hm(image, face, head_channel, object_channel)
            _, hm = model(image, face, head_channel, object_channel)
            # overlay output on image
            # hm = outputs.view(-1, 227, 227)
            # hm = hm.squeeze().detach().cpu().numpy()
            # hm = np.resize(hm, (224,224))
            # i = img.squeeze().detach().cpu().numpy().transpose(1,2,0)
            # plt.imshow(i)
            # plt.imshow(hm, 'jet', interpolation='none', alpha=0.5)
            # count+=1
            # plt.savefig('heatmap/overlay' + str(count) + '.png')

            final_output = raw_hm.cpu().data.numpy()
            pred_labels = outputs.max(1)[1]  # max function returns both values and indices. so max()[0] is values, max()[1] is indices
            inputs_size = image.size(0)
            for i in range(inputs_size):
                distval, f_point = euclid_dist(pred_labels.data.cpu()[i], gaze_final[i])
                ang_error = calc_ang_err(pred_labels[i], gaze_final[i], eye[i])
                # auc_score = cal_auc(ground_labels[i], raw_hm[i, :, :])
                predmap, gtmap = cal_auc(gaze_final[i], raw_hm[i, :, :])
                # select 5 nearest boxes
                bbox_data = select_nearest_bbox(f_point, gt_bboxes[:-1, i, :])
                nearest_bbox = bbox_data['index']
                min_id = 0
                min_dist = np.NINF
                # box distance
                for k, b in enumerate(bbox_data['box']):
                    b = b * [640, 480, 640, 480]
                    b = b.astype(int)
                    contour = np.array([[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]])
                    dist = cv2.pointPolygonTest(contour, (f_point[0] * 640, f_point[1] * 480), True)
                    if min_dist < dist:
                        min_dist = dist
                        min_id = k
                # bbox IOU using gtbox
                max_id = -1
                max_iou = 0
                for k, b in enumerate(bbox_data['box']):
                    b = b * [640, 480, 640, 480]
                    b = b.astype(int)
                    gt_box = gt_bboxes[gaze_idx[i], i] * [640, 480, 640, 480]
                    gt_box = gt_box.astype(int)
                    iou = bb_iou(b, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_id = k
                # bbox IOU using heatmap
                nearest_box_binary = get_bb_binary(gt_bboxes)
                heatmap = outputs.view(-1, 227, 227).squeeze().detach().cpu().numpy()
                heatmap = np.resize(heatmap, (224,224)).clip(min=0)
                heatmap2 = hm.cpu().detach().numpy()*255
                heatmap2 = heatmap2.squeeze()
                heatmap2 = cv2.resize(heatmap2, (224,224))
                max_overlap = 0
                max_overlap_id = -1
                for k, b in enumerate(nearest_box_binary):
                    overlap = np.sum(np.multiply(b, heatmap2))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_overlap_id = k

                # nearest box by center
                if (gaze_idx[i] == nearest_bbox[0]):
                    all_auc.append(1)
                else:
                    all_auc.append(0)
                # nearest box by distance to box
                if (gaze_idx[i] == nearest_bbox[min_id]):
                    all_auc2.append(1)
                else:
                    all_auc2.append(0)
                # nearest box by label
                gt_label = gt_labels[gaze_idx[i], i]
                if (gt_label == gt_labels[nearest_bbox[0], i]):
                    all_label.append(1)
                else:
                    all_label.append(0)
                # nearest box by box_iou
                if max_id == -1:
                    all_iou.append(0)
                elif (gaze_idx[i] == nearest_bbox[max_id]):
                    all_iou.append(1)
                else:
                    all_iou.append(0)
                # nearest box by heatmap overlap
                if max_overlap_id == -1:
                    all_overlap.append(0)
                elif gaze_idx[i] == max_overlap_id:
                    all_overlap.append(1)
                else:
                    all_overlap.append(0)
                all_gazepoints.append(f_point)
                all_predmap.append(predmap)
                all_gtmap.append(gtmap)
                PA_count[np.array(percent_dists) > distval.item()] += 1
                total_error.append([distval, ang_error])

        l2, ang = np.mean(np.array(total_error), axis=0)

        all_gazepoints = np.vstack(all_gazepoints)
        all_predmap = np.stack(all_predmap).reshape([-1])
        all_gtmap = np.stack(all_gtmap).reshape([-1])
        auc = roc_auc_score(all_gtmap, all_predmap)
        box_auc = (sum(all_auc) / len(all_auc)) * 100
        box_auc2 = (sum(all_auc2) / len(all_auc2)) * 100
        label_auc = (sum(all_label) / len(all_label)) * 100
        iou_auc = (sum(all_iou) / len(all_iou)) * 100
        overlap_auc = (sum(all_overlap) / len(all_overlap)) * 100

    if save_output:
        np.savez('predictions.npz', gazepoints=all_gazepoints)

    proxAcc = PA_count / len(test_data_loader.dataset)
    logger.info('proximate accuracy: %s' % str(proxAcc))
    logger.info('average error: %s' % str([auc, l2, ang, box_auc, box_auc2, label_auc, iou_auc, overlap_auc]))

    return [auc, l2, ang]