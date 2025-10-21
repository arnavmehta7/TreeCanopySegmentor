#!/usr/bin/env python3
"""
Comprehensive YOLO Training Script for Tree Canopy Segmentation
Integrates W&B, hyperparameter optimization, and model comparison.
"""

import os
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from ultralytics import YOLO

# W&B imports
import wandb
import weave

# Import our custom modules
from yolo_benchmark_wandb import YOLOBenchmarker, ModelConfig, TrainingConfig
from hyperparameter_optimization import YOLOHyperparameterOptimizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOTrainer:
    """Main training class that orchestrates the entire process"""
    
    def __init__(self, config_path: str = None, wandb_project: str = "tree-canopy-segmentation",
                 wandb_entity: str = "paapi", api_key: str = None):
        """
        Initialize the YOLO trainer
        
        Args:
            config_path: Path to configuration file
            wandb_project: W&B project name
            wandb_entity: W&B entity/team name
            api_key: W&B API key
        """
        self.config_path = config_path
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        
        # Initialize components
        self.benchmarker = YOLOBenchmarker(wandb_project, wandb_entity, api_key)
        self.optimizer = YOLOHyperparameterOptimizer(wandb_project, wandb_entity, api_key)
        
        # Load configuration
        self.config = self._load_config()
        
        logger.info(f"Initialized YOLO Trainer for project: {wandb_project}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        else:
            config = self._create_default_config()
            logger.info("Using default configuration")
        
        return config

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'data_yaml': '/workspace/configurations/model_data-seg.yaml',
            'models': [
                'yolov8n-seg.pt',
                'yolov8s-seg.pt',
                'yolov8m-seg.pt',
                'yolov8l-seg.pt',
                'yolo11n-seg.pt',
                'yolo11s-seg.pt',
                'yolo11m-seg.pt',
                'yolo11l-seg.pt'
            ],
            'input_sizes': [640, 832, 1024],
            'batch_sizes': [8, 12, 16],
            'epochs': 100,
            'hyperopt_trials': 30,
            'hyperopt_epochs': 50,
            'device': '0' if torch.cuda.is_available() else 'cpu',
            'workers': 8,
            'patience': 50,
            'save_period': -1,
            'cache': 'ram',
            'deterministic': True,
            'seed': 42,
            'amp': True,
            'fraction': 1.0,
            'val': True,
            'plots': True,
            'save_json': True,
            'conf': 0.001,
            'iou': 0.6,
            'max_det': 300
        }

    def run_quick_benchmark(self) -> Dict[str, Any]:
        """Run a quick benchmark with default configurations"""
        logger.info("Running quick benchmark...")
        
        # Create quick configurations
        quick_configs = [
            ModelConfig(
                name="YOLOv8s-seg-quick",
                model_path="yolov8s-seg.pt",
                input_size=640,
                batch_size=12,
                epochs=50,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation={
                    'hsv_h': 0.015,
                    'hsv_s': 0.7,
                    'hsv_v': 0.4,
                    'degrees': 0.0,
                    'translate': 0.1,
                    'scale': 0.5,
                    'shear': 0.0,
                    'perspective': 0.0,
                    'flipud': 0.0,
                    'fliplr': 0.5,
                    'mosaic': 1.0,
                    'mixup': 0.0,
                    'copy_paste': 0.0
                }
            ),
            ModelConfig(
                name="YOLOv11s-seg-quick",
                model_path="yolo11s-seg.pt",
                input_size=640,
                batch_size=12,
                epochs=50,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation={
                    'hsv_h': 0.015,
                    'hsv_s': 0.7,
                    'hsv_v': 0.4,
                    'degrees': 0.0,
                    'translate': 0.1,
                    'scale': 0.5,
                    'shear': 0.0,
                    'perspective': 0.0,
                    'flipud': 0.0,
                    'fliplr': 0.5,
                    'mosaic': 1.0,
                    'mixup': 0.0,
                    'copy_paste': 0.0
                }
            )
        ]
        
        training_config = TrainingConfig(
            data_yaml=self.config['data_yaml'],
            project_name="tree-canopy-segmentation",
            device=self.config['device'],
            workers=self.config['workers'],
            patience=self.config['patience'],
            save_period=self.config['save_period'],
            cache=self.config['cache'],
            deterministic=self.config['deterministic'],
            seed=self.config['seed'],
            amp=self.config['amp'],
            fraction=self.config['fraction'],
            val=self.config['val'],
            plots=self.config['plots'],
            save_json=self.config['save_json'],
            conf=self.config['conf'],
            iou=self.config['iou'],
            max_det=self.config['max_det']
        )
        
        return self.benchmarker.run_benchmark(
            data_yaml=self.config['data_yaml'],
            configs=quick_configs,
            training_config=training_config
        )

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run full benchmark with all model configurations"""
        logger.info("Running full benchmark...")
        
        # Get all configurations
        configs = self.benchmarker.create_model_configs()
        
        # Filter based on config
        if 'models' in self.config:
            configs = [c for c in configs if any(model in c.model_path for model in self.config['models'])]
        
        training_config = TrainingConfig(
            data_yaml=self.config['data_yaml'],
            project_name="tree-canopy-segmentation",
            device=self.config['device'],
            workers=self.config['workers'],
            patience=self.config['patience'],
            save_period=self.config['save_period'],
            cache=self.config['cache'],
            deterministic=self.config['deterministic'],
            seed=self.config['seed'],
            amp=self.config['amp'],
            fraction=self.config['fraction'],
            val=self.config['val'],
            plots=self.config['plots'],
            save_json=self.config['save_json'],
            conf=self.config['conf'],
            iou=self.config['iou'],
            max_det=self.config['max_det']
        )
        
        return self.benchmarker.run_benchmark(
            data_yaml=self.config['data_yaml'],
            configs=configs,
            training_config=training_config
        )

    def run_hyperparameter_optimization(self, model_name: str) -> Dict[str, Any]:
        """Run hyperparameter optimization for a specific model"""
        logger.info(f"Running hyperparameter optimization for {model_name}")
        
        study = self.optimizer.optimize(
            model_name=model_name,
            data_yaml=self.config['data_yaml'],
            n_trials=self.config.get('hyperopt_trials', 30),
            base_epochs=self.config.get('hyperopt_epochs', 50)
        )
        
        # Create optimized configuration
        config = self.optimizer.create_optimized_config(
            study=study,
            model_name=model_name,
            output_path=f"optimized_config_{model_name.replace('.pt', '')}.yaml"
        )
        
        return {
            'study': study,
            'config': config,
            'best_params': study.best_params,
            'best_value': study.best_value
        }

    def train_optimized_model(self, model_name: str, optimized_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a model with optimized hyperparameters"""
        logger.info(f"Training optimized model: {model_name}")
        
        # Initialize W&B run
        run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=f"optimized_{model_name}_{int(time.time())}",
            config=optimized_config,
            tags=["optimized", "yolo", "segmentation", "tree-canopy"]
        )
        
        try:
            # Load model
            model = YOLO(model_name)
            
            # Train with optimized parameters
            results = model.train(**optimized_config)
            
            # Get validation results
            val_results = model.val()
            
            # Extract metrics
            metrics = {
                'model_name': model_name,
                'mAP50': val_results.box.map50 if hasattr(val_results, 'box') else 0,
                'mAP50-95': val_results.box.map if hasattr(val_results, 'box') else 0,
                'seg_mAP50': val_results.seg.map50 if hasattr(val_results, 'seg') else 0,
                'seg_mAP50-95': val_results.seg.map if hasattr(val_results, 'seg') else 0,
                'precision': val_results.box.mp if hasattr(val_results, 'box') else 0,
                'recall': val_results.box.mr if hasattr(val_results, 'box') else 0,
                'f1': val_results.box.f1 if hasattr(val_results, 'box') else 0
            }
            
            # Log metrics
            wandb.log(metrics)
            
            # Save model artifact
            if hasattr(results, 'save_dir'):
                model_path = Path(results.save_dir) / 'weights' / 'best.pt'
                if model_path.exists():
                    artifact = wandb.Artifact(f"optimized-model-{model_name}", type="model")
                    artifact.add_file(str(model_path))
                    wandb.log_artifact(artifact)
            
            logger.info(f"Completed training optimized model: {model_name}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training optimized model {model_name}: {str(e)}")
            wandb.log({"error": str(e)})
            return {"model_name": model_name, "error": str(e)}
        finally:
            wandb.finish()

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        logger.info("Starting complete training pipeline...")
        
        pipeline_results = {}
        
        # Step 1: Quick benchmark
        logger.info("Step 1: Running quick benchmark...")
        quick_results = self.run_quick_benchmark()
        pipeline_results['quick_benchmark'] = quick_results
        
        # Step 2: Hyperparameter optimization for best models
        logger.info("Step 2: Running hyperparameter optimization...")
        best_models = ['yolov8s-seg.pt', 'yolo11s-seg.pt']  # Based on quick results
        
        optimization_results = {}
        for model_name in best_models:
            opt_result = self.run_hyperparameter_optimization(model_name)
            optimization_results[model_name] = opt_result
        
        pipeline_results['hyperparameter_optimization'] = optimization_results
        
        # Step 3: Train optimized models
        logger.info("Step 3: Training optimized models...")
        optimized_results = {}
        for model_name, opt_result in optimization_results.items():
            optimized_result = self.train_optimized_model(model_name, opt_result['config'])
            optimized_results[model_name] = optimized_result
        
        pipeline_results['optimized_training'] = optimized_results
        
        # Step 4: Full benchmark with all models
        logger.info("Step 4: Running full benchmark...")
        full_results = self.run_full_benchmark()
        pipeline_results['full_benchmark'] = full_results
        
        # Save all results
        with open('complete_pipeline_results.json', 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        logger.info("Complete pipeline finished!")
        return pipeline_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLO Training Script for Tree Canopy Segmentation')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['quick', 'full', 'hyperopt', 'complete'], 
                       default='complete', help='Training mode')
    parser.add_argument('--model', type=str, help='Specific model to train (for hyperopt mode)')
    parser.add_argument('--wandb-project', type=str, default='tree-canopy-segmentation',
                       help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default='paapi',
                       help='W&B entity name')
    parser.add_argument('--api-key', type=str, help='W&B API key')
    
    args = parser.parse_args()
    
    # Set up API key
    if args.api_key:
        os.environ['WANDB_API_KEY'] = args.api_key
    else:
        os.environ['WANDB_API_KEY'] = 'f21e18e97e313644edf2723adf692971cec13175'
    
    # Initialize trainer
    trainer = YOLOTrainer(
        config_path=args.config,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        api_key=args.api_key
    )
    
    # Run based on mode
    if args.mode == 'quick':
        results = trainer.run_quick_benchmark()
    elif args.mode == 'full':
        results = trainer.run_full_benchmark()
    elif args.mode == 'hyperopt':
        if not args.model:
            logger.error("Model name required for hyperopt mode")
            sys.exit(1)
        results = trainer.run_hyperparameter_optimization(args.model)
    elif args.mode == 'complete':
        results = trainer.run_complete_pipeline()
    
    logger.info("Training completed successfully!")
    logger.info(f"Results: {results}")

if __name__ == "__main__":
    main()