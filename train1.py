from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torchvision
from MyCocoDetection import CocoDetection
import config
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def custom_collate_fn(batch):
    filtered_batch = [(image, target) for image, target in batch if target]
    images, targets = zip(*filtered_batch)
    return images, targets


weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
preprocess = weights.transforms()
# Define the path to your dataset annotations and images
ann_file = 'coco2017/annotations/instances_train2017.json'
image_root = 'coco2017/train2017/'

# Define the transformation to apply to the images
transform = transforms.Compose([
    # transforms.Resize((480, 640)),
    transforms.ToTensor()
])

# Create an instance of your CocoDataset
train_dataset = CocoDetection(root=image_root, annFile=ann_file, transform=transform)
# Create a DataLoader with the custom collate function
train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    collate_fn=custom_collate_fn
)

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
preprocess = weights.transforms()
num_classes = 91
model = maskrcnn_resnet50_fpn_v2(
    weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    num_classes=num_classes
)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
# Modify the model for pretraining
model.train()

images, labels = next(iter(train_dataloader))
image = images[0]
# Convert the image tensor to a numpy array
image_numpy = image.numpy()

# Ensure the image has the correct shape
if image.ndim == 3:
    image_numpy = image_numpy.transpose(1, 2, 0)  # Convert from CxHxW to HxWxC

# Show the image
plt.imshow(image_numpy)
plt.show()

# Define the loss function and optimizer
params = [p for p in model.parameters() if p.requires_grad]
criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.SGD(params, lr=0.08, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.step_size, config.gamma)
# Training loop
for epoch in range(config.num_epochs):
    print("Epoch:", epoch)
    running_loss = 0.0
    # Iterate over the training dataset
    for images, targets in train_dataloader:
        images = list(image.to(config.device) for image in images)
        targets = [{k: v.to(config.device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images, targets)
        # Compute loss
        # loss = outputs['loss_mask']
        total_loss = sum(loss for loss in outputs.values())
        # Backward pass
        total_loss.backward()

        # Update the model parameters
        optimizer.step()

        # Clear the gradients
        optimizer.zero_grad()
        
        # Update the running loss
        running_loss += total_loss.item()

    # Adjust learning rate
    lr_scheduler.step()

    # Compute the average loss for the epoch
    epoch_loss = running_loss / len(train_dataloader)

    # Print the loss for the epoch
    print(f"Epoch [{epoch + 1}/{config.num_epochs}], Loss: {epoch_loss:.4f}")