# train by CocoDetection
import torch
import torchvision
from torch.utils.data import DataLoader
# from torchvision.datasets import CocoDetection
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.transforms import transforms

import config
from MyCocoDetection import CocoDetection
from tarin_utils import showImageAndTarget

# Define the path to your dataset annotations and images
ann_file = 'coco2017/annotations/instances_train2017.json'
image_root = 'coco2017/train2017/'

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])


def custom_collate_fn(batch):
    images, targets = zip(*batch)
    return images, targets


# Create an instance of your CocoDataset
train_dataset = CocoDetection(root=image_root, annFile=ann_file, transform=transform)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    collate_fn=custom_collate_fn
)
# Test
showImageAndTarget(train_dataset)

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# preprocess = weights.transforms()
num_classes = 91
model = maskrcnn_resnet50_fpn_v2(
    weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    num_classes=91
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
# Modify the model for pretraining
model.train()

# Define the loss function and optimizer
params = [p for p in model.parameters() if p.requires_grad]
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params, lr=0.08, momentum=0.9)

# Training
num_epochs = 10

for epoch in range(num_epochs):
    for images, targets in train_dataloader:
        images = list(image.to(config.device) for image in images)
        targets = [{k: v.to(config.device) for k, v in t.items()} for t in targets]
        # Forward pass, Compute the loss
        outputs = model(images, targets)
        loss_dict = outputs['losses']
        total_loss = sum(loss for loss in loss_dict.values())

        # Backward pass, Update the model parameters
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # loss information
        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {total_loss}")

    # save model
    torch.save(model.state_dict(), '/path/to/save/model.pt')