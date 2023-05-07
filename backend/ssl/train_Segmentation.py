#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import torch
import json
import random
from scipy.spatial import distance

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    DatasetMapper,
    build_detection_test_loader,
)
from detectron2.engine import (
    DefaultPredictor,
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    COCOEvaluator,
    PascalVOCDetectionEvaluator,
    inference_on_dataset,
    DatasetEvaluator,
    inference_context,
)
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import Res5ROIHeads, ROI_HEADS_REGISTRY
from detectron2.structures import BoxMode
from detectron2.utils import logger, comm
from detectron2.utils.analysis import flop_count_operators
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine.hooks import HookBase
import detectron2.data.detection_utils as mapperutils
import detectron2.data.transforms as mapperT
from detectron2.utils.visualizer import ColorMode
import argparse

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

from detectron2.utils.logger import setup_logger
setup_logger()


# Add arguments
parser = argparse.ArgumentParser(description="Training and validation for instance segmentation")
parser.add_argument("--mode", type=str,default="train",choices=["train", "inference"],help="Mode can be train and val")
parser.add_argument("--val_dataset", type=str, help="Validation dataset")
parser.add_argument("--train_dataset", type=str, help="Training dataset")
parser.add_argument("--val_dataset", type=str, help="Validation dataset")
args = parser.parse_args()


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if "coco" in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert "voc" in dataset_name
            return PascalVOCDetectionEvaluator(dataset_name)


def setup():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.freeze()
    return cfg



cfg=setup()


model = Trainer.build_model(cfg)


cfg.OUTPUT_DIR,cfg.MODEL.WEIGHTS



model_weights="detection/best_converted.pth"
out_dir="."



checkpointer = DetectionCheckpointer(model)
checkpointer.load(model_weights)


def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
    
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")



def vis_dataset():  
    import random
    dataset_dicts = get_balloon_dicts("balloon/train")
    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        v = Visualizer(im[:, :, ::-1], metadata=balloon_metadata, scale=0.8)
        v = v.draw_dataset_dict(d)
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()


class LossEvalHook(HookBase):
    """Hook Class for Calculating Evaluation

    Args:
        HookBase ([type]): Hooks for training intermediate output
    """

    def __init__(self, cfg, model, data_loader):
        self.cfg=cfg
        self._period = self.cfg.TEST.EVAL_PERIOD
        self._data_loader = data_loader
        self.predictor = DefaultPredictor(cfg)
        self.predictor.model=model
        self.calculate_flops=True
        #self.cls_meta = cfg.DATASETS.METADATAINFO
        assert id(self.predictor.model)==id(model)

    def _do_loss_eval(self):
        losses = []
        with torch.no_grad():
            for idx, inputs in enumerate(self._data_loader):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                loss_batch = self._get_loss(inputs)
                losses.append(loss_batch)
            mean_loss = np.mean(losses)
            logger.logging.info(f"Mean Validation loss {mean_loss}")
            self.trainer.storage.put_scalar("Validation_Loss", mean_loss)
            comm.synchronize()
        return losses
    
    
    def vis_images(self,num_images=20):
        # set model to eval
        self.predictor.model.eval()
        log_counter = 0
        for idx, inputs in enumerate(self._data_loader):
            img=inputs[0]["image"].permute(1,2,0).detach().to("cpu").numpy()
            img=img.copy()
            outputs = self.predictor(img) 
            v = Visualizer(img[:, :, ::-1],
                            metadata=None, 
                            scale=0.5, 
                            instance_mode=ColorMode.IMAGE_BW   
                )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            log_counter += 1
            #out.get_image()
            if idx>=num_images:
                break
        #set back to train
        self.predictor.model.train()
        

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self.predictor.model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
    
    def get_flops(self):
        # Take one batch
        inputs= next(iter(self._data_loader))
        flops=flop_count_operators(self.predictor.model,inputs)
        return flops
    
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if self.calculate_flops:
            flps=self.get_flops()
            total=0
            for k,v in flps.items():
                total+=v
            # total has flops
            self.calculate_flops=False
            
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
            logger.logging.info(f"Validation Bbox AP {self.trainer.storage.history('bbox/AP').latest()}")
            logger.logging.info(f"Validation Segmentation AP {self.trainer.storage.history('segm/AP').latest()}")
            self.vis_images()



class LossTrainHook(HookBase):
    """Hook Class for Calculating Evaluation

    Args:
        HookBase ([type]): Hooks for training intermediate output
    """

    def __init__(self,period):
        self._period=period

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            train_loss=self.trainer.storage.history("total_loss").median(20)
            #if self.wab:
            #    self.wab.log({"Train Loss": train_loss})


class InstanceTrainer(DefaultTrainer):
    """Create a trainer for veg datasets

    Args:
        DefaultTrainer ([type]): Default trainer from detectron2

    Returns:
        [type]: [description]
    """
    def __init__(self,cfg):
        super().__init__(cfg)
        
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name,output_dir=cfg.OUTPUT_DIR)

    def build_hooks(self):
        hooks = super().build_hooks()
        # Build mapper Augmentations by passing the test mode but pass the test mode=False in mapper to get the labels finally replace the augm list with the 
        #  test one a workaround to not change the mapper class 
        augs=mapperutils.build_augmentation(self.cfg, is_train=False)
        augsList=mapperT.AugmentationList(augs)
        val_mapper=DatasetMapper(self.cfg, True)
        val_mapper.augmentations=augsList
          
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg,
                self.model,
                build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], val_mapper),
            ),
        )
        hooks.insert(
            -1,
            LossTrainHook(self.cfg.TRAINLOG)
        )
        return hooks


# Config
def get_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ("balloon_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_weights
    # cfg["INPUT"]["MASK_FORMAT"]='bitmask'
    cfg["INPUT"]["RANDOM_FLIP"] = "horizontal"
    cfg["INPUT"]["ROTATE"] = [-2.0, 2.0]
    cfg["INPUT"]["LIGHT_SCALE"] = 1.1
    cfg["INPUT"]["Brightness_SCALE"] = [0.9, 1.1]
    cfg["INPUT"]["Contrast_SCALE"] = [0.9, 1.1]
    cfg["INPUT"]["Saturation_SCALE"] = [0.9, 1.1]
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 1e-4  # pick a good LR
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.SOLVER.MAX_ITER = 10000  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.TEST.EVAL_PERIOD=1000
    cfg.OUTPUT_DIR=out_dir
    os.makedirs(cfg.OUTPUT_DIR ,exist_ok=True)
    # New configs
    cfg.set_new_allowed(True)
    cfg.TRAINLOG=100
    cfg_dict=yaml.load(cfg.dump())
    # save all the configs for prediction
    with open(os.path.join(cfg.OUTPUT_DIR,"pred_config.yaml"), 'w') as file:
        file.write(cfg.dump())


# Traning
def train(cfg):
    trainer = InstanceTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()




# Inference
def inference(cfg):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.DATASETS.TEST = ('balloon_val', )
    predictor = DefaultPredictor(cfg)
    dataset_dicts = get_balloon_dicts("balloon/val")
    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata, 
                       scale=0.8, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()
    
if __name__=="__main__":
    cvf=get_config() 
    if args.mode=="train:
        train(cfg)
    else:
        inference(cfg)
    