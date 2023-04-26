import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from pytorch_lightning import Trainer
import timm
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import json
from pycocotools.mask import decode as mask_decode
import logging
import numpy as np
import cv2
from collections import OrderedDict
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as cocomask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_mask_list(mask_list_points,height,width):
    """
    Decode a list of masks to a list of binary masks
    :param mask_list: List of masks
    :return: List of binary masks
    """
    binary_mask = np.zeros((height,width),dtype=np.uint8)
    for points in mask_list_points:
        # Reshape the flattened list into a list of 2D points
        np_points = np.array(points).reshape(-1, 2).astype(np.int32)
        contour = [np_points]  
        binary_mask = cv2.drawContours(binary_mask, contour, -1, 1, -1) 
    binary_mask=torch.tensor(binary_mask)
    return binary_mask


def load_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    #logger.info(f"Annotations data keys are {data.keys()}")
    annotations = []
    for item in data["annotations"]:
        ann = {
            "bbox": item["bbox"],
            "category_id": item["category_id"],
            "iscrowd": item["iscrowd"],
            "area": item["area"],
            "masks": decode_mask_list(item["segmentation"],data["images"][0]['height'],data["images"][0]['width'])
        }
        annotations.append(ann)
    return annotations


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "images", "*.jpg")))
        self.json_paths = sorted(glob.glob(os.path.join(root_dir, "annotations", "*.json")))

        print(f"Images length {len(self.image_paths)}")
        print(f"Labels length {len(self.json_paths)}")
        
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

        # Create the target dictionary
        target = {
            "boxes": torch.tensor([[x, y, x + w, y + h] for x, y, w, h in [ann["bbox"] for ann in annotations]], dtype=torch.float32),
            "labels": torch.tensor([ann["category_id"] for ann in annotations], dtype=torch.int64),
            "masks": torch.stack([ann["masks"] for ann in annotations], dim=0),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor([ann["area"] for ann in annotations], dtype=torch.float32),
            "iscrowd": torch.tensor([ann["iscrowd"] for ann in annotations], dtype=torch.int64)
        }
        
        return image, target


class COCODataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=2, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Define the dataset transforms
        transforms = ToTensor()

        # Load the custom dataset
        self.train_dataset = CustomDataset(os.path.join(self.data_dir, "train"), transform=transforms)
        self.val_dataset = CustomDataset(os.path.join(self.data_dir, "valid"), transform=transforms)
    
    
    def custom_collate_fn(self, batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        #print(f"The images length are {len(images)}")
        #print(f"The targets length are {len(targets)}")
        #print(f"Image shape is {images[0].shape}")
        #print(f"Image min and max are  {images[0].min(),images[0].max()}")
        #print(f"Target keys are {targets[0].keys()}") 
        #print(f"Target masks min and max are  {targets[0]['masks'].min(),targets[0]['masks'].max()}") 
        return images, targets

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,collate_fn=self.custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,collate_fn=self.custom_collate_fn)

# To return dict of features instead of list
class CustomTimmModel(torch.nn.Module):
    def __init__(self, backbone_name, pretrained=True, features_only=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=features_only)
        self.out_channels = self.backbone.feature_info.channels()[-1]

    def forward(self, x):
        # Call the forward() method of the timm model
        feature_maps = self.backbone(x)
        
        return feature_maps[-1]

def get_instance_segmentation_model(backbone_name,num_classes):
    # Load the backbone from timm
    backbone = CustomTimmModel(backbone_name, pretrained=True, features_only=True)

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=14,sampling_ratio=2)


    # put the pieces together inside a MaskRCNN model
    model = MaskRCNN(backbone,
     num_classes=2,
     rpn_anchor_generator=anchor_generator,
     box_roi_pool=roi_pooler,
     mask_roi_pool=mask_roi_pooler)

    return model


class InstanceSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes, backbone_name, learning_rate=0.001):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = get_instance_segmentation_model(num_classes, backbone_name)
        self.all_preds = []
        self.all_targets = []
        # For coco evaluation
        self.img_idx = 0
        self.ann_id = 0

    def forward(self, x, targets=None):
        if self.training and targets is not None:
            return self.model(x, targets)
        else:
            return self.model(x)

    def training_step(self, batch,batch_idx):
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

    def on_validation_epoch_start(self):
        self.all_preds = []
        self.all_targets = []
        self.img_idx = 0
        
    def convert_predictions_to_coco_format(self, preds):
        coco_preds = []
        ann_id=0
        for pred_idx, pred in enumerate(preds):
            img_id = self.img_idx + pred_idx
            for box, label, score, mask in zip(pred["boxes"], pred["labels"], pred["scores"], pred["masks"]):
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1
                bbox = [x1, y1, w, h]

                rle_mask = rle_mask = cocomask.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))

                coco_pred = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": label.item(),
                    "bbox": bbox,
                    "score": score.item(),
                    "segmentation": rle_mask,
                }
                coco_preds.append(coco_pred)
        return coco_preds
    
    def convert_targets_to_coco_format(self, targets):
        coco_targets = []
        ann_id=0
        for img_idx, target in enumerate(targets):
            img_id = self.img_idx + img_idx
            for box, label, mask, area, iscrowd in zip(target["boxes"], target["labels"], target["masks"], target["area"], target["iscrowd"]):
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1
                bbox = [x1, y1, w, h]

                rle_mask = cocomask.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))

                coco_target = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": label.item(),
                    "bbox": bbox,
                    "area": area.item(),
                    "segmentation": rle_mask,
                    "iscrowd": iscrowd.item()
                }
                coco_targets.append(coco_target)
                ann_id += 1
        return coco_targets

    def validation_step(self, batch,batch_idx):
        #pass
        images, targets = batch
    
        # Run the model on the images and targets
        preds = self.model(images, targets)
        
        # Convert everything to CPU
        images=[img.detach().cpu() for img in images]
        targets = [{k: v.to("cpu") for k, v in t.items()} for t in targets]
        preds = [{k: v.to("cpu") for k, v in t.items()} for t in preds]

        # Convert predictions and targets to the COCO format
        coco_preds = self.convert_predictions_to_coco_format(preds)
        coco_targets = self.convert_targets_to_coco_format(targets)

        # Add predictions and ground truth to lists
        self.all_preds.append(coco_preds)
        self.all_targets.append(coco_targets)
        
        self.img_idx += len(images)

    def on_validation_epoch_end(self):
        # Combine all predictions and ground truth from the validation set
        combined_preds = np.concatenate(self.all_preds, axis=0)
        combined_targets = np.concatenate(self.all_targets, axis=0)

        # Evaluate the predictions using COCO API
        box_ap, mask_ap = self.evaluate_predictions(combined_preds, combined_targets)

        # Log the results
        self.log("val_box_mAP", box_ap, prog_bar=True, logger=True)
        self.log("val_mask_mAP", mask_ap, prog_bar=True, logger=True)
    
    def evaluate_predictions(self, preds, targets):
        # Create a COCO object for the ground truth
        coco_gt = COCO()
        coco_gt.dataset = {"images": [{"id": idx} for idx in range(len(targets))], "annotations": targets}
        coco_gt.createIndex()

        # Create a COCO object for the predictions
        coco_preds = COCO()
        coco_preds.dataset = {"images": [{"id": idx} for idx in range(len(preds))], "annotations": preds}
        coco_preds.createIndex()

        # Calculate box mAP
        box_eval = COCOeval(coco_gt, coco_preds, iouType='bbox')
        box_eval.evaluate()
        box_eval.accumulate()
        box_eval.summarize()
        box_ap = box_eval.stats[0]

        # Calculate mask mAP
        mask_eval = COCOeval(coco_gt, coco_preds, iouType='segm')
        mask_eval.evaluate()
        mask_eval.accumulate()
        mask_eval.summarize()
        mask_ap = mask_eval.stats[0]

        return box_ap, mask_ap
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__=="__main__":
    # Set the number of classes, backbone, and dimension
    num_classes = 2  # 1 class (person) + 1 background class
    backbone_name = "resnet18"  # You can use any other backbone supported by timm

    lightning_module = InstanceSegmentationModel(backbone_name,num_classes)
    
    #lightning_module.eval()
    #x = [torch.rand(3, 300, 400)]
    #predictions = lightning_module(x)
    #print(predictions[0]["masks"].shape)
    #exit()

    # Set the path to the COCO dataset
    coco_data_dir = "/home/asad/Downloads/Balloons.v15i.coco-segmentation/"

    # Initialize the data module
    data_module = COCODataModule(coco_data_dir)

    # Initialize the trainer
    trainer = Trainer(accelerator="gpu", max_epochs=10)

    # Start the training
    trainer.fit(lightning_module, data_module)


    # Assuming the lightning_module is an instance of InstanceSegmentationModel
    #lightning_module.eval()  # Set the model to evaluation mode
    #with torch.no_grad():
    #    images = ...  # Load your images as a batch of tensors
    #    predictions = lightning_module(images)
