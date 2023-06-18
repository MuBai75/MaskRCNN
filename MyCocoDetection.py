import torch
import numpy as np
import os
from torchvision.datasets import CocoDetection
from PIL import Image
from pycocotools.coco import COCO


class CocoDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        super(CocoDetection, self).__init__(root=root, annFile=annFile,
                                            transform=transform, target_transform=target_transform)
        self.annFile = annFile
        self.root = root
        self.transform = transform
        self.coco = COCO(self.annFile)
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def is_valid_bbox(self, bbox):
        # Check if the bbox has valid coordinates
        x1, y1, x2, y2 = bbox
        if x1 < x2 and y1 < y2 and x2 > 0 and y2 > 0:
            return True
        return False

    def get_target(self, img_id: int, annotations: list, width: int, height: int):
        targets = {}
        boxes = []
        masks = []
        labels = []
        # Iterate over annotations
        for ann in annotations:
            # Check if the annotation has a valid mask
            if 'segmentation' in ann and ann['segmentation'] is not None:
                bbox = ann['bbox']
                # Perform validation on the bbox coordinates
                if self.is_valid_bbox(bbox):
                    boxes.append(bbox)
                    mask = self.coco.annToMask(ann)
                    masks.append(mask)
                    labels.append(ann['category_id'])

        # Check if there are valid boxes and masks
        if len(boxes) > 0 and len(masks) > 0:
            targets['boxes'] = torch.tensor(boxes)
            targets['masks'] = torch.tensor(np.array(masks), dtype=torch.uint8)
            targets['image_id'] = torch.tensor(img_id)
            targets['labels'] = torch.tensor(labels)

        return targets

    def __getitem__(self, index):
        # Load image
        coco = self.coco
        img_id = self.image_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        width, height = image.size
        target = self.get_target(img_id, annotations, width, height)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    """ area = torch.tensor([obj["area"] for obj in annotations])
    iscrowd = torch.tensor([obj["iscrowd"] for obj in annotations]) """
