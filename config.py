# Configurations for pretraining Mask R-CNN
import torch

# Dataset paths
# Specify the root directory where your dataset is located
dataset_root = "coco2017/train2017/"
# Specify the path to your annotation file in COCO JSON format
annotations_file = "coco2017/annotations/instances_train2017.json"

# Model settings
backbone = 'maskrcnn_resnet50_fpn_v2'
pretrained_backbone = True
num_classes = 91  # Number of classes including background

# Training hyperparameters
step_size = 3
batch_size = 4
learning_rate = 0.008
momentum = 0.9
num_epochs = 1
gamma = 0.1

# Data loader settings
shuffle = True
num_workers = 1

# Loss function
# Define your desired loss function here
# loss_fn = ...
 
# Other configurations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = 'pretrained_mask_rcnn.pth'