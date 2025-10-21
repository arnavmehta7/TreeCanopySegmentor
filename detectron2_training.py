#!/usr/bin/env python3
"""
Detectron2 Training and Fine-tuning for Tree Canopy Segmentation
Integrates with W&B for experiment tracking and exports results in the required format.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

# Detectron2 imports
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import get_detection_dataset_dicts
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetMapper
from detectron2.checkpoint import DetectionCheckpointer

# W&B imports
import wandb
import weave

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TreeCanopyDataset(Dataset):
    """Custom dataset class for tree canopy segmentation"""
    
    def __init__(self, dataset_dicts, mapper=None):
        self.dataset_dicts = dataset_dicts
        self.mapper = mapper
    
    def __len__(self):
        return len(self.dataset_dicts)
    
    def __getitem__(self, idx):
        if self.mapper is None:
            return self.dataset_dicts[idx]
        else:
            return self.mapper(self.dataset_dicts[idx])

class Detectron2Trainer:
    """Main trainer class for Detectron2 models"""
    
    def __init__(self, wandb_project: str = "tree-canopy-detectron2", 
                 wandb_entity: str = "paapi", api_key: str = None):
        """
        Initialize the Detectron2 trainer
        
        Args:
            wandb_project: W&B project name
            wandb_entity: W&B entity/team name
            api_key: W&B API key
        """
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        
        # Initialize Weave
        weave.init(f"{wandb_entity}/{wandb_project}")
        
        # Set up logging
        setup_logger()
        
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        logger.info(f"Initialized Detectron2 Trainer for project: {wandb_project}")

    def register_datasets(self, data_yaml: str):
        """Register datasets with Detectron2"""
        try:
            # Load data configuration
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Define dataset paths
            data_root = Path(data_config['path'])
            train_dir = data_root / data_config['train']
            val_dir = data_root / data_config['val']
            
            # Register training dataset
            def get_tree_canopy_dicts(img_dir):
                dataset_dicts = []
                for idx, img_path in enumerate(Path(img_dir).glob("*.tif")):
                    record = {}
                    
                    # Load image
                    image = cv2.imread(str(img_path))
                    height, width = image.shape[:2]
                    
                    record["file_name"] = str(img_path)
                    record["image_id"] = idx
                    record["height"] = height
                    record["width"] = width
                    
                    # Load annotations (assuming YOLO format)
                    ann_path = img_path.with_suffix('.txt')
                    if ann_path.exists():
                        objs = []
                        with open(ann_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 7:  # class_id + 6 coords (3 points)
                                    class_id = int(parts[0])
                                    coords = [float(x) for x in parts[1:]]
                                    
                                    # Convert normalized coords to absolute
                                    points = []
                                    for i in range(0, len(coords), 2):
                                        x = coords[i] * width
                                        y = coords[i + 1] * height
                                        points.append([x, y])
                                    
                                    if len(points) >= 3:  # At least 3 points for polygon
                                        obj = {
                                            "bbox": [min(p[0] for p in points), min(p[1] for p in points),
                                                   max(p[0] for p in points), max(p[1] for p in points)],
                                            "bbox_mode": BoxMode.XYXY_ABS,
                                            "category_id": class_id,
                                            "segmentation": [points]
                                        }
                                        objs.append(obj)
                        
                        record["annotations"] = objs
                    else:
                        record["annotations"] = []
                    
                    dataset_dicts.append(record)
                return dataset_dicts
            
            # Register datasets
            DatasetCatalog.register("tree_canopy_train", lambda: get_tree_canopy_dicts(train_dir))
            DatasetCatalog.register("tree_canopy_val", lambda: get_tree_canopy_dicts(val_dir))
            
            # Register metadata
            MetadataCatalog.get("tree_canopy_train").set(thing_classes=["tree_canopy"])
            MetadataCatalog.get("tree_canopy_val").set(thing_classes=["tree_canopy"])
            
            logger.info("Datasets registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering datasets: {str(e)}")
            return False

    def create_config(self, model_name: str, num_classes: int = 1, 
                     ims_per_batch: int = 2, base_lr: float = 0.00025,
                     max_iter: int = 1000, checkpoint_period: int = 500) -> Any:
        """Create Detectron2 configuration"""
        
        cfg = get_cfg()
        
        # Model configuration based on model_name
        if "mask_rcnn" in model_name.lower():
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        elif "cascade" in model_name.lower():
            cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
        elif "retinanet" in model_name.lower():
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
        else:
            # Default to Mask R-CNN
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        # Dataset configuration
        cfg.DATASETS.TRAIN = ("tree_canopy_train",)
        cfg.DATASETS.TEST = ("tree_canopy_val",)
        cfg.DATALOADER.NUM_WORKERS = 2
        
        # Model configuration
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        # Training configuration
        cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
        cfg.SOLVER.BASE_LR = base_lr
        cfg.SOLVER.MAX_ITER = max_iter
        cfg.SOLVER.STEPS = (int(max_iter * 0.8), int(max_iter * 0.9))
        cfg.SOLVER.GAMMA = 0.1
        cfg.SOLVER.WARMUP_ITERS = 100
        cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
        cfg.SOLVER.WEIGHT_DECAY = 0.0001
        cfg.SOLVER.MOMENTUM = 0.9
        
        # Checkpointing
        cfg.OUTPUT_DIR = f"outputs/{model_name}_{int(time.time())}"
        cfg.CHECKPOINT_PERIOD = checkpoint_period
        
        # Data augmentation
        cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
        cfg.INPUT.MAX_SIZE_TRAIN = 1333
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333
        
        # Augmentation
        cfg.INPUT.CROP.ENABLED = True
        cfg.INPUT.CROP.TYPE = "relative_range"
        cfg.INPUT.CROP.SIZE = [0.8, 0.8]
        
        return cfg

    @weave.op()
    def train_model(self, model_name: str, data_yaml: str, 
                   num_classes: int = 1, ims_per_batch: int = 2,
                   base_lr: float = 0.00025, max_iter: int = 1000,
                   checkpoint_period: int = 500) -> Dict[str, Any]:
        """
        Train a Detectron2 model
        
        Args:
            model_name: Name of the model to train
            data_yaml: Path to data configuration
            num_classes: Number of classes
            ims_per_batch: Images per batch
            base_lr: Base learning rate
            max_iter: Maximum iterations
            checkpoint_period: Checkpoint period
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting training for {model_name}")
        
        # Initialize W&B run
        run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=f"{model_name}_{int(time.time())}",
            config={
                'model_name': model_name,
                'data_yaml': data_yaml,
                'num_classes': num_classes,
                'ims_per_batch': ims_per_batch,
                'base_lr': base_lr,
                'max_iter': max_iter,
                'checkpoint_period': checkpoint_period
            },
            tags=["detectron2", "segmentation", "tree-canopy", model_name]
        )
        
        try:
            # Register datasets
            if not self.register_datasets(data_yaml):
                raise Exception("Failed to register datasets")
            
            # Create configuration
            cfg = self.create_config(
                model_name=model_name,
                num_classes=num_classes,
                ims_per_batch=ims_per_batch,
                base_lr=base_lr,
                max_iter=max_iter,
                checkpoint_period=checkpoint_period
            )
            
            # Create trainer
            trainer = DefaultTrainer(cfg)
            
            # Add W&B logging hook
            trainer.register_hooks([WandBLoggingHook(cfg)])
            
            # Start training
            start_time = time.time()
            trainer.resume_or_load(resume=False)
            trainer.train()
            training_time = time.time() - start_time
            
            # Evaluate model
            evaluator = COCOEvaluator("tree_canopy_val", output_dir=cfg.OUTPUT_DIR)
            val_loader = build_detection_test_loader(cfg, "tree_canopy_val")
            results = inference_on_dataset(trainer.model, val_loader, evaluator)
            
            # Extract metrics
            metrics = {
                'model_name': model_name,
                'training_time': training_time,
                'max_iter': max_iter,
                'base_lr': base_lr,
                'ims_per_batch': ims_per_batch,
                'bbox_mAP': results.get('bbox/AP', 0.0),
                'bbox_mAP50': results.get('bbox/AP50', 0.0),
                'bbox_mAP75': results.get('bbox/AP75', 0.0),
                'segm_mAP': results.get('segm/AP', 0.0),
                'segm_mAP50': results.get('segm/AP50', 0.0),
                'segm_mAP75': results.get('segm/AP75', 0.0),
                'output_dir': cfg.OUTPUT_DIR
            }
            
            # Log metrics
            wandb.log(metrics)
            
            # Save model artifact
            model_path = Path(cfg.OUTPUT_DIR) / "model_final.pth"
            if model_path.exists():
                artifact = wandb.Artifact(f"detectron2-model-{model_name}", type="model")
                artifact.add_file(str(model_path))
                wandb.log_artifact(artifact)
            
            logger.info(f"Completed training for {model_name}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            wandb.log({"error": str(e)})
            return {"model_name": model_name, "error": str(e)}
        finally:
            wandb.finish()

    def predict_and_export(self, model_path: str, test_images_dir: str, 
                          output_dir: str, confidence_threshold: float = 0.5) -> str:
        """
        Run inference and export results in the required format
        
        Args:
            model_path: Path to trained model
            test_images_dir: Directory containing test images
            output_dir: Directory to save results
            confidence_threshold: Confidence threshold for predictions
            
        Returns:
            Path to exported submission file
        """
        logger.info("Starting prediction and export...")
        
        # Load configuration
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        
        # Create predictor
        predictor = DefaultPredictor(cfg)
        
        # Get test images
        test_images = list(Path(test_images_dir).glob("*.tif"))
        
        # Process each image
        results = []
        for img_path in test_images:
            # Load image
            image = cv2.imread(str(img_path))
            height, width = image.shape[:2]
            
            # Get predictions
            outputs = predictor(image)
            
            # Process predictions
            instances = outputs["instances"]
            annotations = []
            
            for i in range(len(instances)):
                # Get bounding box and confidence
                bbox = instances.pred_boxes[i].tensor[0].cpu().numpy()
                confidence = instances.scores[i].cpu().numpy()
                
                # Get segmentation mask
                if hasattr(instances, 'pred_masks'):
                    mask = instances.pred_masks[i].cpu().numpy()
                    # Convert mask to polygon
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Get largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        # Simplify contour
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        # Convert to normalized coordinates
                        segmentation = []
                        for point in approx:
                            x, y = point[0]
                            segmentation.extend([x / width, y / height])
                        
                        annotations.append({
                            "class": 0,  # tree_canopy class
                            "confidence_level": float(confidence),
                            "segmentation": segmentation
                        })
            
            # Create image record
            image_record = {
                "file_name": img_path.name,
                "width": width,
                "height": height,
                "cm_resolution": img_path.name[:2],  # Extract from filename
                "scene_type": "Unknown",
                "annotations": annotations
            }
            
            results.append(image_record)
        
        # Create submission format
        submission = {"images": results}
        
        # Save submission
        output_path = Path(output_dir) / "submission.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(submission, f, indent=2)
        
        logger.info(f"Exported submission to {output_path}")
        return str(output_path)

class WandBLoggingHook:
    """Custom hook for W&B logging during training"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.step = 0
    
    def __call__(self, trainer):
        if hasattr(trainer, 'storage') and trainer.storage:
            metrics = {}
            for k, v in trainer.storage.latest().items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
            
            if metrics:
                wandb.log(metrics, step=self.step)
        
        self.step += 1

def main():
    """Main function to run Detectron2 training"""
    # Set up API key
    os.environ['WANDB_API_KEY'] = 'f21e18e97e313644edf2723adf692971cec13175'
    
    # Initialize trainer
    trainer = Detectron2Trainer(
        wandb_project="tree-canopy-detectron2",
        wandb_entity="paapi"
    )
    
    # Data configuration
    data_yaml = "/workspace/configurations/model_data-seg.yaml"
    
    # Models to train
    models = [
        "mask_rcnn_R_50_FPN_3x",
        "mask_rcnn_R_101_FPN_3x",
        "cascade_mask_rcnn_R_50_FPN_3x",
        "cascade_mask_rcnn_R_101_FPN_3x"
    ]
    
    # Training configurations
    configs = [
        {"ims_per_batch": 2, "base_lr": 0.00025, "max_iter": 1000},
        {"ims_per_batch": 4, "base_lr": 0.0005, "max_iter": 2000},
        {"ims_per_batch": 2, "base_lr": 0.0001, "max_iter": 1500},
        {"ims_per_batch": 1, "base_lr": 0.00025, "max_iter": 3000}
    ]
    
    # Train models
    results = {}
    for i, model in enumerate(models):
        config = configs[i % len(configs)]
        logger.info(f"Training {model} with config {config}")
        
        result = trainer.train_model(
            model_name=model,
            data_yaml=data_yaml,
            **config
        )
        
        results[model] = result
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1].get('segm_mAP50', 0))
    logger.info(f"Best model: {best_model[0]} with mAP50: {best_model[1].get('segm_mAP50', 0)}")
    
    # Export results for best model
    model_path = f"outputs/{best_model[0]}_*/model_final.pth"
    test_images_dir = "/workspace/data/processed/images/predict"
    output_dir = "/workspace/exports"
    
    submission_path = trainer.predict_and_export(
        model_path=model_path,
        test_images_dir=test_images_dir,
        output_dir=output_dir
    )
    
    logger.info(f"Training completed! Best model: {best_model[0]}")
    logger.info(f"Submission exported to: {submission_path}")

if __name__ == "__main__":
    main()