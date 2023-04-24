import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pytorch_lightning import Trainer
import timm
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import json
from pycocotools.mask import decode as mask_decode
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    logger.info(f"Annotations data keys are {data.keys()}")
    annotations = []
    for item in data:
        ann = {
            "bbox": item["bbox"],
            "category_id": item["category_id"],
            "iscrowd": item["iscrowd"],
            "area": item["area"],
            "masks": mask_decode(item["segmentation"])
        }
        annotations.append(ann)
    return annotations


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "images", "*.jpg")))
        self.json_paths = sorted(glob.glob(os.path.join(root_dir, "annotations", "*.json")))
        logger.info(os.path.join(root_dir, "images"))
        logger.info(f"Number of images are {len(self.image_paths)}")
        logger.info(f"Number of annotations are {len(self.json_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        json_path = self.json_paths[idx]

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Load the annotations
        annotations = load_annotations(json_path)

        # Apply the transforms if provided
        if self.transform:
            image = self.transform(image)
        logger.info(f"Annotations are {annotations}")

        # Create the target dictionary
        target = {
            "boxes": torch.tensor([ann["bbox"] for ann in annotations], dtype=torch.float32),
            "labels": torch.tensor([ann["category_id"] for ann in annotations], dtype=torch.int64),
            "masks": torch.tensor([ann["masks"] for ann in annotations], dtype=torch.uint8),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor([ann["area"] for ann in annotations], dtype=torch.float32),
            "iscrowd": torch.tensor([ann["iscrowd"] for ann in annotations], dtype=torch.int64)
        }

        return image, target


class COCODataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=4, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Define the dataset transforms
        transforms = ToTensor()

        # Load the custom dataset
        self.train_dataset = CustomDataset(os.path.join(self.data_dir, "train"), transform=transforms)
        self.val_dataset = CustomDataset(os.path.join(self.data_dir, "val"), transform=transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def get_instance_segmentation_model(num_classes, backbone_name, dim):
    # Load the backbone from timm
    backbone = timm.create_model(backbone_name, pretrained=True, features_only=True, num_classes=dim)
    # Get the number of output channels from the backbone
    backbone_out_channels = backbone.feature_info.channels()[-1]

    # Create an anchor generator for the RPN
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                                                        aspect_ratios=((0.5, 1.0, 2.0),))

    # Create the Region Proposal Network (RPN) head
    rpn_head = torchvision.models.detection.rpn.RPNHead(backbone_out_channels, anchor_generator.num_anchors_per_location()[0])

    # Create the RPN
    positive_fraction = 0.5
    pre_nms_top_n = {"training": 2000, "testing": 1000}
    post_nms_top_n = {"training": 2000, "testing": 1000}
    nms_thresh = 0.7

    rpn = torchvision.models.detection.rpn.RegionProposalNetwork(
        anchor_generator, rpn_head, 0.7, 2000, 1000, positive_fraction, pre_nms_top_n, post_nms_top_n, nms_thresh)

    # Create the box head for the Mask R-CNN model
    box_head = torchvision.models.detection.roi_heads.TwoMLPHead(backbone_out_channels * 7 * 7, 1024)

    # Create the box predictor for the Mask R-CNN model
    box_predictor = FastRCNNPredictor(1024, num_classes)

    # Create the mask head for the Mask R-CNN model
    mask_head = torchvision.models.detection.roi_heads.MaskRCNNHeads(backbone_out_channels, 1024, num_classes)

    # Create the mask predictor for the Mask R-CNN model
    mask_predictor = MaskRCNNPredictor(1024, 256, num_classes)

    # Assemble the Mask R-CNN model using the custom backbone and heads
    model = MaskRCNN(backbone, rpn, box_head, box_predictor, mask_head, mask_predictor)

    return model



class InstanceSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes, backbone_name, dim, learning_rate=0.001):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = get_instance_segmentation_model(num_classes, backbone_name, dim)

    def forward(self, x, targets=None):
        if self.training or targets is not None:
            return self.model(x, targets)
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # Set the target's device to be the same as the model's device
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Run the model on the images and targets
        loss_dict = self.model(images, targets)

        # Calculate the total loss by summing individual losses
        total_loss = sum(loss for loss in loss_dict.values())

        # Log the training losses
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for key, value in loss_dict.items():
            self.log(f"train_{key}", value, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # Set the target's device to be the same as the model's device
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Run the model on the images and targets
        loss_dict = self.model(images, targets)

        # Calculate the total loss by summing individual losses
        total_loss = sum(loss for loss in loss_dict.values())

        # Log the validation losses
        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for key, value in loss_dict.items():
            self.log(f"val_{key}", value, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__=="__main__":
    # Set the number of classes, backbone, and dimension
    num_classes = 2  # 1 class (person) + 1 background class
    backbone_name = "resnet50"  # You can use any other backbone supported by timm
    dim = 2048  # Feature dimension, you can set it according to the backbone used

    # Initialize the Lightning module
    lightning_module = InstanceSegmentationModel(num_classes, backbone_name, dim)

    # Set the path to the COCO dataset
    coco_data_dir = "/path/to/coco/dataset"

    # Initialize the data module
    data_module = COCODataModule(coco_data_dir)

    # Initialize the trainer
    trainer = Trainer(gpus=1, max_epochs=10, progress_bar_refresh_rate=20)

    # Start the training
    trainer.fit(lightning_module, data_module)


    # Assuming the lightning_module is an instance of InstanceSegmentationModel
    #lightning_module.eval()  # Set the model to evaluation mode
    #with torch.no_grad():
    #    images = ...  # Load your images as a batch of tensors
    #    predictions = lightning_module(images)
