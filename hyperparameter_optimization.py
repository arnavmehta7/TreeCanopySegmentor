#!/usr/bin/env python3
"""
Hyperparameter Optimization for YOLO Tree Canopy Segmentation
Uses Optuna for efficient hyperparameter search with W&B integration.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import optuna
from optuna.integration import WeightsAndBiasesCallback
import torch
from ultralytics import YOLO

# W&B imports
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOHyperparameterOptimizer:
    """Hyperparameter optimization for YOLO models using Optuna"""
    
    def __init__(self, wandb_project: str = "tree-canopy-hyperopt", 
                 wandb_entity: str = "paapi", api_key: str = None):
        """
        Initialize the hyperparameter optimizer
        
        Args:
            wandb_project: W&B project name
            wandb_entity: W&B entity/team name
            api_key: W&B API key
        """
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        logger.info(f"Initialized YOLO Hyperparameter Optimizer for project: {wandb_project}")

    def objective(self, trial, model_name: str, data_yaml: str, 
                  base_epochs: int = 50, base_batch_size: int = 8) -> float:
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            model_name: YOLO model name (e.g., 'yolov8s-seg.pt')
            data_yaml: Path to data configuration YAML
            base_epochs: Base number of epochs for optimization
            base_batch_size: Base batch size for optimization
            
        Returns:
            Validation mAP50 score to maximize
        """
        try:
            # Sample hyperparameters
            lr0 = trial.suggest_float('lr0', 1e-5, 1e-1, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            momentum = trial.suggest_float('momentum', 0.6, 0.98)
            warmup_epochs = trial.suggest_int('warmup_epochs', 0, 10)
            warmup_momentum = trial.suggest_float('warmup_momentum', 0.5, 0.95)
            warmup_bias_lr = trial.suggest_float('warmup_bias_lr', 0.01, 0.5)
            
            # Loss function weights
            box_loss_weight = trial.suggest_float('box_loss_weight', 0.5, 10.0)
            cls_loss_weight = trial.suggest_float('cls_loss_weight', 0.5, 10.0)
            dfl_loss_weight = trial.suggest_float('dfl_loss_weight', 0.5, 10.0)
            
            # Augmentation parameters
            hsv_h = trial.suggest_float('hsv_h', 0.0, 0.1)
            hsv_s = trial.suggest_float('hsv_s', 0.0, 1.0)
            hsv_v = trial.suggest_float('hsv_v', 0.0, 1.0)
            degrees = trial.suggest_float('degrees', 0.0, 45.0)
            translate = trial.suggest_float('translate', 0.0, 0.5)
            scale = trial.suggest_float('scale', 0.0, 1.0)
            shear = trial.suggest_float('shear', 0.0, 10.0)
            perspective = trial.suggest_float('perspective', 0.0, 0.001)
            flipud = trial.suggest_float('flipud', 0.0, 1.0)
            fliplr = trial.suggest_float('fliplr', 0.0, 1.0)
            mosaic = trial.suggest_float('mosaic', 0.0, 1.0)
            mixup = trial.suggest_float('mixup', 0.0, 0.3)
            copy_paste = trial.suggest_float('copy_paste', 0.0, 0.3)
            
            # Model architecture parameters
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            
            # Input size
            imgsz = trial.suggest_categorical('imgsz', [416, 512, 640, 832, 1024])
            
            # Batch size (adjust based on input size)
            if imgsz <= 512:
                batch_size = trial.suggest_categorical('batch_size', [8, 12, 16, 20, 24])
            elif imgsz <= 640:
                batch_size = trial.suggest_categorical('batch_size', [6, 8, 12, 16, 20])
            elif imgsz <= 832:
                batch_size = trial.suggest_categorical('batch_size', [4, 6, 8, 12, 16])
            else:
                batch_size = trial.suggest_categorical('batch_size', [2, 4, 6, 8, 12])
            
            # Optimizer choice
            optimizer = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW', 'RMSprop'])
            
            # Learning rate scheduler
            cos_lr = trial.suggest_categorical('cos_lr', [True, False])
            
            # Initialize W&B run for this trial
            run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=f"trial_{trial.number}_{model_name}",
                config={
                    'trial_number': trial.number,
                    'model_name': model_name,
                    'lr0': lr0,
                    'weight_decay': weight_decay,
                    'momentum': momentum,
                    'warmup_epochs': warmup_epochs,
                    'warmup_momentum': warmup_momentum,
                    'warmup_bias_lr': warmup_bias_lr,
                    'box_loss_weight': box_loss_weight,
                    'cls_loss_weight': cls_loss_weight,
                    'dfl_loss_weight': dfl_loss_weight,
                    'hsv_h': hsv_h,
                    'hsv_s': hsv_s,
                    'hsv_v': hsv_v,
                    'degrees': degrees,
                    'translate': translate,
                    'scale': scale,
                    'shear': shear,
                    'perspective': perspective,
                    'flipud': flipud,
                    'fliplr': fliplr,
                    'mosaic': mosaic,
                    'mixup': mixup,
                    'copy_paste': copy_paste,
                    'dropout': dropout,
                    'imgsz': imgsz,
                    'batch_size': batch_size,
                    'optimizer': optimizer,
                    'cos_lr': cos_lr
                },
                tags=["hyperopt", "yolo", "segmentation", "tree-canopy", f"trial_{trial.number}"]
            )
            
            # Load model
            model = YOLO(model_name)
            
            # Training arguments
            train_args = {
                'data': data_yaml,
                'epochs': base_epochs,
                'imgsz': imgsz,
                'batch': batch_size,
                'device': '0' if torch.cuda.is_available() else 'cpu',
                'workers': 8,
                'project': f"runs/hyperopt/{model_name}",
                'name': f"trial_{trial.number}",
                'patience': 20,
                'save_period': -1,
                'cache': 'ram',
                'deterministic': True,
                'seed': 42,
                'amp': True,
                'fraction': 1.0,
                'val': True,
                'plots': False,
                'save_json': True,
                'conf': 0.001,
                'iou': 0.6,
                'max_det': 300,
                'optimizer': optimizer,
                'lr0': lr0,
                'weight_decay': weight_decay,
                'momentum': momentum,
                'warmup_epochs': warmup_epochs,
                'warmup_momentum': warmup_momentum,
                'warmup_bias_lr': warmup_bias_lr,
                'box': box_loss_weight,
                'cls': cls_loss_weight,
                'dfl': dfl_loss_weight,
                'dropout': dropout,
                'cos_lr': cos_lr,
                'hsv_h': hsv_h,
                'hsv_s': hsv_s,
                'hsv_v': hsv_v,
                'degrees': degrees,
                'translate': translate,
                'scale': scale,
                'shear': shear,
                'perspective': perspective,
                'flipud': flipud,
                'fliplr': fliplr,
                'mosaic': mosaic,
                'mixup': mixup,
                'copy_paste': copy_paste
            }
            
            # Train model
            results = model.train(**train_args)
            
            # Get validation results
            val_results = model.val()
            
            # Extract mAP50 score
            mAP50 = val_results.box.map50 if hasattr(val_results, 'box') else 0.0
            
            # Log metrics
            wandb.log({
                'mAP50': mAP50,
                'mAP50-95': val_results.box.map if hasattr(val_results, 'box') else 0.0,
                'seg_mAP50': val_results.seg.map50 if hasattr(val_results, 'seg') else 0.0,
                'seg_mAP50-95': val_results.seg.map if hasattr(val_results, 'seg') else 0.0,
                'precision': val_results.box.mp if hasattr(val_results, 'box') else 0.0,
                'recall': val_results.box.mr if hasattr(val_results, 'box') else 0.0,
                'f1': val_results.box.f1 if hasattr(val_results, 'box') else 0.0
            })
            
            # Report intermediate result for pruning
            trial.report(mAP50, step=base_epochs)
            
            # Check if trial should be pruned
            if trial.should_prune():
                wandb.finish()
                raise optuna.TrialPruned()
            
            wandb.finish()
            return mAP50
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            wandb.log({"error": str(e)})
            wandb.finish()
            return 0.0

    def optimize(self, model_name: str, data_yaml: str, n_trials: int = 100, 
                 timeout: int = None, base_epochs: int = 50) -> optuna.Study:
        """
        Run hyperparameter optimization
        
        Args:
            model_name: YOLO model name
            data_yaml: Path to data configuration YAML
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            base_epochs: Number of epochs per trial
            
        Returns:
            Optuna study object with optimization results
        """
        logger.info(f"Starting hyperparameter optimization for {model_name}")
        logger.info(f"Running {n_trials} trials with {base_epochs} epochs each")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=5
            )
        )
        
        # Create W&B callback
        wandb_callback = WeightsAndBiasesCallback(
            project=self.wandb_project,
            entity=self.wandb_entity,
            as_multirun=True
        )
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, model_name, data_yaml, base_epochs),
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[wandb_callback]
        )
        
        # Log best parameters
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best mAP50: {study.best_value}")
        
        # Save study
        study_path = f"hyperopt_study_{model_name.replace('.pt', '')}.pkl"
        optuna.study.save_study(study, study_path)
        logger.info(f"Study saved to {study_path}")
        
        return study

    def create_optimized_config(self, study: optuna.Study, 
                              model_name: str, output_path: str = None) -> Dict[str, Any]:
        """
        Create optimized configuration file from study results
        
        Args:
            study: Optuna study object
            model_name: YOLO model name
            output_path: Path to save configuration file
            
        Returns:
            Dictionary containing optimized configuration
        """
        best_params = study.best_params
        
        config = {
            'task': 'segment',
            'mode': 'train',
            'model': model_name,
            'data': '/workspace/configurations/model_data-seg.yaml',
            'epochs': 200,  # Use more epochs for final training
            'time': None,
            'patience': 50,
            'batch': best_params.get('batch_size', 8),
            'imgsz': best_params.get('imgsz', 640),
            'save': True,
            'save_period': -1,
            'cache': 'ram',
            'device': '0',
            'workers': 8,
            'project': f'runs/segment/{model_name.replace(".pt", "")}_optimized',
            'name': f'train_{model_name.replace(".pt", "")}_optimized',
            'exist_ok': False,
            'pretrained': True,
            'optimizer': best_params.get('optimizer', 'SGD'),
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': best_params.get('cos_lr', False),
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': best_params.get('dropout', 0.0),
            'val': True,
            'split': 'val',
            'save_json': True,
            'conf': 0.001,
            'iou': 0.6,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'plots': True,
            'source': None,
            'vid_stride': 1,
            'stream_buffer': False,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'embed': None,
            'show': False,
            'save_frames': False,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'show_boxes': True,
            'line_width': None,
            'format': 'torchscript',
            'keras': False,
            'optimize': False,
            'int8': False,
            'dynamic': False,
            'simplify': True,
            'opset': None,
            'workspace': None,
            'nms': False,
            'tracker': 'botsort.yaml',
            'save_dir': f'runs/segment/{model_name.replace(".pt", "")}_optimized/train_{model_name.replace(".pt", "")}_optimized',
            # Optimized hyperparameters
            'lr0': best_params.get('lr0', 0.01),
            'lrf': 0.01,
            'momentum': best_params.get('momentum', 0.937),
            'weight_decay': best_params.get('weight_decay', 0.0005),
            'warmup_epochs': best_params.get('warmup_epochs', 3.0),
            'warmup_momentum': best_params.get('warmup_momentum', 0.8),
            'warmup_bias_lr': best_params.get('warmup_bias_lr', 0.1),
            'box': best_params.get('box_loss_weight', 7.5),
            'cls': best_params.get('cls_loss_weight', 0.5),
            'dfl': best_params.get('dfl_loss_weight', 1.5),
            'pose': 12.0,
            'kobj': 1.0,
            'nbs': 32,
            'hsv_h': best_params.get('hsv_h', 0.015),
            'hsv_s': best_params.get('hsv_s', 0.7),
            'hsv_v': best_params.get('hsv_v', 0.4),
            'degrees': best_params.get('degrees', 0.0),
            'translate': best_params.get('translate', 0.1),
            'scale': best_params.get('scale', 0.5),
            'shear': best_params.get('shear', 0.0),
            'perspective': best_params.get('perspective', 0.0),
            'flipud': best_params.get('flipud', 0.0),
            'fliplr': best_params.get('fliplr', 0.5),
            'bgr': 0.0,
            'mosaic': best_params.get('mosaic', 1.0),
            'mixup': best_params.get('mixup', 0.0),
            'cutmix': 0.0,
            'copy_paste': best_params.get('copy_paste', 0.0),
            'copy_paste_mode': 'flip',
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'cfg': None
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Optimized configuration saved to {output_path}")
        
        return config

def main():
    """Main function to run hyperparameter optimization"""
    # Set up API key
    os.environ['WANDB_API_KEY'] = 'f21e18e97e313644edf2723adf692971cec13175'
    
    # Initialize optimizer
    optimizer = YOLOHyperparameterOptimizer(
        wandb_project="tree-canopy-hyperopt",
        wandb_entity="paapi"
    )
    
    # Data configuration
    data_yaml = "/workspace/configurations/model_data-seg.yaml"
    
    # Models to optimize
    models_to_optimize = [
        "yolov8s-seg.pt",
        "yolov8m-seg.pt", 
        "yolo11s-seg.pt",
        "yolo11m-seg.pt"
    ]
    
    # Run optimization for each model
    for model_name in models_to_optimize:
        logger.info(f"Starting optimization for {model_name}")
        
        # Run optimization
        study = optimizer.optimize(
            model_name=model_name,
            data_yaml=data_yaml,
            n_trials=50,  # Adjust based on available compute
            base_epochs=30  # Adjust based on available time
        )
        
        # Create optimized configuration
        config = optimizer.create_optimized_config(
            study=study,
            model_name=model_name,
            output_path=f"optimized_config_{model_name.replace('.pt', '')}.yaml"
        )
        
        logger.info(f"Completed optimization for {model_name}")
        logger.info(f"Best mAP50: {study.best_value}")
        logger.info(f"Best parameters: {study.best_params}")

if __name__ == "__main__":
    main()