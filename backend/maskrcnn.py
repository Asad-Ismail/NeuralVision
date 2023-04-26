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
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


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
        # using just last feature map as input not building FPN can change in future
        feature_maps = self.backbone(x)
        return feature_maps[-1]
    
    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

def get_instance_segmentation_model(backbone_name,num_classes):
    # Load the backbone from timm
    backbone = CustomTimmModel(backbone_name, pretrained=True, features_only=True)
    # Freeze the backbone
    backbone.freeze()

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
        self.categories = [{"id": 1, "name": "Ballon"}]


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
                mask[mask>0.5]=1
                mask=mask.squeeze(0)
                h,w=mask.shape
                rle_mask = cocomask.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))
                area = float(mask.sum().item())
                coco_pred = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": label.item(),
                    "bbox": bbox,
                    "score": score.item(),
                    "segmentation": rle_mask,
                    "area": area,
                    "height": h,
                    "width": w
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
                h,w=mask.shape
                rle_mask = cocomask.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))

                coco_target = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": label.item(),
                    "bbox": bbox,
                    "area": area.item(),
                    "segmentation": rle_mask,
                    "iscrowd": iscrowd.item(),
                    "height": h,
                    "width": w
                }
                coco_targets.append(coco_target)
                ann_id += 1
        return coco_targets

    def validation_step(self, batch,batch_idx):
        images, targets = batch
    
        # Run the model on the images and targets
        with torch.no_grad():
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
        coco_gt.dataset = {"images": [{"id": idx, "height": targets[idx]["height"], "width": targets[idx]["width"]} for idx in range(len(targets))], "annotations": targets,"categories": self.categories}
        coco_gt.createIndex()

        # Create a COCO object for the predictions
        coco_preds = COCO()
        #coco_preds.dataset = {"images": [{"id": idx, "height": preds[idx]["height"], "width": preds[idx]["width"]} for idx in range(len(preds))], "annotations": preds, "categories": self.categories}
        coco_preds.dataset = {"images": [{"id": idx, "height": targets[idx]["height"], "width": targets[idx]["width"]} for idx in range(len(targets))], "annotations": preds, "categories": self.categories}
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


def visualize_predictions(image,preds,score_threshold=0.5):
    img = to_pil_image(image.cpu())
    boxes = preds['boxes'].cpu().numpy()
    labels = preds['labels'].cpu().numpy()
    masks = preds['masks'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()

    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Draw bounding boxes and class labels
    for box, label, mask,score in zip(boxes, labels, masks,scores):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box.astype(np.int32)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, str(label), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Create a mask overlay
        mask_overlay = np.zeros_like(img_cv, dtype=np.uint8)
        mask[mask>0.5]=1
        mask_overlay[:, :, 1] = mask * 255
        # Combine the original image and the mask overlay
        alpha = 0.5
        img_cv = cv2.addWeighted(img_cv, 1, mask_overlay, alpha, 0)
    # Display the image
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def main(perform_inference=False):
    # Set the number of classes, backbone, and dimension
    num_classes = 2  # 1 class (person) + 1 background class
    backbone_name = "resnet18"  # You can use any other backbone supported by timm

    lightning_module = InstanceSegmentationModel(backbone_name, num_classes,learning_rate=1e-4)
    # Set the path to the COCO dataset
    coco_data_dir = "/home/asad/Downloads/Balloons.v15i.coco-segmentation/"

    # Initialize the data module
    data_module = COCODataModule(coco_data_dir)
    # Create a ModelCheckpoint callback to save the best model and the last model weights
    checkpoint_callback = ModelCheckpoint(
            dirpath="model_checkpoints",
            filename="best_model_and_last_weights-{epoch:02d}-{train_loss:.2f}",
            save_top_k=1,
            verbose=True,
            monitor="train_loss",
            mode="min",
            save_last=True,
        )
    # Initialize the trainer
    trainer = Trainer(accelerator="gpu", max_epochs=60, callbacks=[checkpoint_callback])

    if not perform_inference:
        # Start the training
        trainer.fit(lightning_module, data_module)
    else:
        # Initialize data module
        data_module.setup()
        # Load the best model weights
        lightning_module.load_state_dict(torch.load("model_checkpoints/last.ckpt")["state_dict"])
        lightning_module.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for the model
            for i,data in enumerate(data_module.val_dataloader()):
                if i<1:
                    continue
                # Get the first batch of data
                images, targets = data
                # Perform inference on the validation dataset
                preds = lightning_module(images)
                # Extract single image and targets from the batch
                image=images[0]
                pred=preds[0]
                #print(image.shape)
                # Visualize the results
                visualize_predictions(image,pred)  # Assuming you have a function named visualize_predictions
                # Break from the loop if you have enough data to visualize the results. Otherwise, keep going.
                if len(preds) > 0:
                    break

if __name__ == "__main__":
    main(perform_inference=False)
