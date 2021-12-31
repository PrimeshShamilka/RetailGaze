import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils_logging import setup_logger
from models.base import BaseModel
from dataloader.base import GooDataset
from training.base import train_base_model, GazeOptimizer

logger = setup_logger(name='first_logger',
                      log_dir ='./logs/',
                      log_file='train_chong_gooreal.log',
                      log_format = '%(asctime)s %(levelname)s %(message)s',
                      verbose=True)

batch_size=4
workers=12
images_dir = '/media/primesh/F4D0EA80D0EA49061/PROJECTS/FYP/Gaze detection/Datasets/gooreal/finalrealdatasetImgsV2'
pickle_path = '/media/primesh/F4D0EA80D0EA49061/PROJECTS/FYP/Gaze detection/Datasets/gooreal/oneshotrealhumansNew2.pickle'
test_images_dir = '/media/primesh/F4D0EA80D0EA49061/PROJECTS/FYP/Gaze detection/Datasets/gooreal/finalrealdatasetImgsV2'
test_pickle_path = '/media/primesh/F4D0EA80D0EA49061/PROJECTS/FYP/Gaze detection/Datasets/gooreal/testrealhumansNew2.pickle'
val_images_dir = '/media/primesh/F4D0EA80D0EA49061/PROJECTS/FYP/Gaze detection/Datasets/gooreal/finalrealdatasetImgsV2'
val_pickle_path = '/media/primesh/F4D0EA80D0EA49061/PROJECTS/FYP/Gaze detection/Datasets/gooreal/valrealhumansNew2.pickle'


print ('Train')
train_set = GooDataset(images_dir, pickle_path, 'train')
train_data_loader = DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=16)
print ('Val')
val_set = GooDataset(val_images_dir, val_pickle_path, 'train')
val_data_loader = DataLoader(dataset=val_set,
                                           batch_size=4,
                                           shuffle=True,
                                           num_workers=16)
print ('Test')
test_set = GooDataset(test_images_dir, test_pickle_path, 'test')
test_data_loader = DataLoader(test_set, batch_size=batch_size//2,
                            shuffle=False, num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BaseModel().cuda()
start_epoch = 0
max_epoch = 5
learning_rate = 1e-4

# Initializes Optimizer
gaze_opt = GazeOptimizer(model, learning_rate)
optimizer = gaze_opt.getOptimizer(start_epoch)
# Loss criteria
# criterion = nn.NLLLoss().cuda()
criterion = nn.MSELoss()


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/base_model')

train_base_model(model, train_data_loader, val_data_loader, criterion, optimizer, logger, writer, num_epochs=50, patience=10)