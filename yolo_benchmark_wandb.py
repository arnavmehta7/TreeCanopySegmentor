#!/usr/bin/env python3
"""
YOLO Model Benchmarking and Training Script for Tree Canopy Segmentation
Integrates with Weights & Biases for experiment tracking and model comparison.
"""

import os
import yaml
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
from PIL import Image

# W&B and Weave imports
import wandb
import weave
from wandb.integration.ultralytics import add_wandb_callback

# YOLO imports
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.data import build_dataloader
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, SegmentMetrics
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import de_parallel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for YOLO model variants"""
    name: str
    model_path: str
    input_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    momentum: float
    optimizer: str
    augmentation: Dict[str, Any]
    dropout: float = 0.0
    freeze_layers: Optional[List[int]] = None

@dataclass
class TrainingConfig:
    """Training configuration"""
    data_yaml: str
    project_name: str
    device: str = '0'
    workers: int = 8
    patience: int = 50
    save_period: int = -1
    cache: str = 'ram'
    deterministic: bool = True
    seed: int = 42
    amp: bool = True
    fraction: float = 1.0
    val: bool = True
    plots: bool = True
    save_json: bool = True
    conf: float = 0.001
    iou: float = 0.6
    max_det: int = 300

class YOLOBenchmarker:
    """Main class for YOLO model benchmarking and training"""
    
    def __init__(self, wandb_project: str = "tree-canopy-segmentation", 
                 wandb_entity: str = "paapi", api_key: str = None):
        """
        Initialize the YOLO benchmarker with W&B integration
        
        Args:
            wandb_project: W&B project name
            wandb_entity: W&B entity/team name
            api_key: W&B API key
        """
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.results = {}
        self.best_models = {}
        
        # Set up W&B
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        
        # Initialize Weave
        weave.init(f"{wandb_entity}/{wandb_project}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        logger.info(f"Initialized YOLO Benchmarker for project: {wandb_project}")

    def create_model_configs(self) -> List[ModelConfig]:
        """Create different YOLO model configurations for benchmarking"""
        
        base_augmentation = {
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
        
        aggressive_augmentation = {
            'hsv_h': 0.025,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.2,
            'scale': 0.9,
            'shear': 2.0,
            'perspective': 0.0001,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1
        }
        
        configs = [
            # YOLOv8 Models
            ModelConfig(
                name="YOLOv8n-seg",
                model_path="yolov8n-seg.pt",
                input_size=640,
                batch_size=16,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=base_augmentation
            ),
            ModelConfig(
                name="YOLOv8s-seg",
                model_path="yolov8s-seg.pt",
                input_size=640,
                batch_size=12,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=base_augmentation
            ),
            ModelConfig(
                name="YOLOv8m-seg",
                model_path="yolov8m-seg.pt",
                input_size=640,
                batch_size=8,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=base_augmentation
            ),
            ModelConfig(
                name="YOLOv8l-seg",
                model_path="yolov8l-seg.pt",
                input_size=640,
                batch_size=6,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=base_augmentation
            ),
            ModelConfig(
                name="YOLOv8x-seg",
                model_path="yolov8x-seg.pt",
                input_size=640,
                batch_size=4,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=base_augmentation
            ),
            
            # YOLOv11 Models
            ModelConfig(
                name="YOLOv11n-seg",
                model_path="yolo11n-seg.pt",
                input_size=640,
                batch_size=16,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=base_augmentation
            ),
            ModelConfig(
                name="YOLOv11s-seg",
                model_path="yolo11s-seg.pt",
                input_size=640,
                batch_size=12,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=base_augmentation
            ),
            ModelConfig(
                name="YOLOv11m-seg",
                model_path="yolo11m-seg.pt",
                input_size=640,
                batch_size=8,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=base_augmentation
            ),
            ModelConfig(
                name="YOLOv11l-seg",
                model_path="yolo11l-seg.pt",
                input_size=640,
                batch_size=6,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=base_augmentation
            ),
            ModelConfig(
                name="YOLOv11x-seg",
                model_path="yolo11x-seg.pt",
                input_size=640,
                batch_size=4,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=base_augmentation
            ),
            
            # High Resolution Variants
            ModelConfig(
                name="YOLOv8s-seg-1024",
                model_path="yolov8s-seg.pt",
                input_size=1024,
                batch_size=6,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=base_augmentation
            ),
            ModelConfig(
                name="YOLOv11s-seg-1024",
                model_path="yolo11s-seg.pt",
                input_size=1024,
                batch_size=6,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=base_augmentation
            ),
            
            # Aggressive Augmentation Variants
            ModelConfig(
                name="YOLOv8s-seg-aug",
                model_path="yolov8s-seg.pt",
                input_size=640,
                batch_size=12,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=aggressive_augmentation
            ),
            ModelConfig(
                name="YOLOv11s-seg-aug",
                model_path="yolo11s-seg.pt",
                input_size=640,
                batch_size=12,
                epochs=100,
                learning_rate=0.01,
                weight_decay=0.0005,
                momentum=0.937,
                optimizer="SGD",
                augmentation=aggressive_augmentation
            ),
            
            # AdamW Optimizer Variants
            ModelConfig(
                name="YOLOv8s-seg-adamw",
                model_path="yolov8s-seg.pt",
                input_size=640,
                batch_size=12,
                epochs=100,
                learning_rate=0.001,
                weight_decay=0.01,
                momentum=0.937,
                optimizer="AdamW",
                augmentation=base_augmentation
            ),
            ModelConfig(
                name="YOLOv11s-seg-adamw",
                model_path="yolo11s-seg.pt",
                input_size=640,
                batch_size=12,
                epochs=100,
                learning_rate=0.001,
                weight_decay=0.01,
                momentum=0.937,
                optimizer="AdamW",
                augmentation=base_augmentation
            ),
        ]
        
        return configs

    @weave.op()
    def train_model(self, config: ModelConfig, training_config: TrainingConfig) -> Dict[str, Any]:
        """
        Train a single YOLO model with the given configuration
        
        Args:
            config: Model configuration
            training_config: Training configuration
            
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info(f"Starting training for {config.name}")
        
        # Initialize W&B run
        run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=f"{config.name}_{int(time.time())}",
            config={
                **asdict(config),
                **asdict(training_config)
            },
            tags=["yolo", "segmentation", "tree-canopy", config.name.split('-')[0]]
        )
        
        try:
            # Load model
            model = YOLO(config.model_path)
            
            # Add W&B callback
            add_wandb_callback(model, enable_model_checkpointing=True)
            
            # Training arguments
            train_args = {
                'data': training_config.data_yaml,
                'epochs': config.epochs,
                'imgsz': config.input_size,
                'batch': config.batch_size,
                'device': training_config.device,
                'workers': training_config.workers,
                'project': f"runs/segment/{config.name}",
                'name': f"train_{config.name}_{int(time.time())}",
                'patience': training_config.patience,
                'save_period': training_config.save_period,
                'cache': training_config.cache,
                'deterministic': training_config.deterministic,
                'seed': training_config.seed,
                'amp': training_config.amp,
                'fraction': training_config.fraction,
                'val': training_config.val,
                'plots': training_config.plots,
                'save_json': training_config.save_json,
                'conf': training_config.conf,
                'iou': training_config.iou,
                'max_det': training_config.max_det,
                'optimizer': config.optimizer,
                'lr0': config.learning_rate,
                'weight_decay': config.weight_decay,
                'momentum': config.momentum,
                'dropout': config.dropout,
                **config.augmentation
            }
            
            # Start training
            start_time = time.time()
            results = model.train(**train_args)
            training_time = time.time() - start_time
            
            # Get validation results
            val_results = model.val()
            
            # Extract metrics
            metrics = {
                'model_name': config.name,
                'training_time': training_time,
                'best_epoch': results.results_dict.get('epoch', 0),
                'box_loss': results.results_dict.get('train/box_loss', 0),
                'seg_loss': results.results_dict.get('train/seg_loss', 0),
                'cls_loss': results.results_dict.get('train/cls_loss', 0),
                'dfl_loss': results.results_dict.get('train/dfl_loss', 0),
                'val_box_loss': results.results_dict.get('val/box_loss', 0),
                'val_seg_loss': results.results_dict.get('val/seg_loss', 0),
                'val_cls_loss': results.results_dict.get('val/cls_loss', 0),
                'val_dfl_loss': results.results_dict.get('val/dfl_loss', 0),
                'mAP50': val_results.box.map50 if hasattr(val_results, 'box') else 0,
                'mAP50-95': val_results.box.map if hasattr(val_results, 'box') else 0,
                'seg_mAP50': val_results.seg.map50 if hasattr(val_results, 'seg') else 0,
                'seg_mAP50-95': val_results.seg.map if hasattr(val_results, 'seg') else 0,
                'precision': val_results.box.mp if hasattr(val_results, 'box') else 0,
                'recall': val_results.box.mr if hasattr(val_results, 'box') else 0,
                'f1': val_results.box.f1 if hasattr(val_results, 'box') else 0,
                'model_size_mb': self._get_model_size(model),
                'inference_time_ms': self._benchmark_inference(model, config.input_size)
            }
            
            # Log metrics to W&B
            wandb.log(metrics)
            
            # Save model artifacts
            if hasattr(results, 'save_dir'):
                model_path = Path(results.save_dir) / 'weights' / 'best.pt'
                if model_path.exists():
                    artifact = wandb.Artifact(f"model-{config.name}", type="model")
                    artifact.add_file(str(model_path))
                    wandb.log_artifact(artifact)
            
            # Create visualizations
            self._create_visualizations(results, config.name)
            
            logger.info(f"Completed training for {config.name}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training {config.name}: {str(e)}")
            wandb.log({"error": str(e)})
            return {"model_name": config.name, "error": str(e)}
        finally:
            wandb.finish()

    def _get_model_size(self, model) -> float:
        """Calculate model size in MB"""
        try:
            # Get the model file size
            model_path = model.ckpt_path if hasattr(model, 'ckpt_path') else None
            if model_path and Path(model_path).exists():
                return Path(model_path).stat().st_size / (1024 * 1024)
            return 0.0
        except:
            return 0.0

    def _benchmark_inference(self, model, input_size: int, num_runs: int = 10) -> float:
        """Benchmark inference time in milliseconds"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, input_size, input_size)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    _ = model(dummy_input)
                times.append(time.time() - start_time)
            
            return np.mean(times) * 1000  # Convert to milliseconds
        except:
            return 0.0

    def _create_visualizations(self, results, model_name: str):
        """Create and log visualizations to W&B"""
        try:
            # Training curves
            if hasattr(results, 'results_dict') and results.results_dict:
                # Create training loss plot
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'Training Results - {model_name}', fontsize=16)
                
                # Plot training losses
                epochs = range(1, len(results.results_dict.get('train/box_loss', [])) + 1)
                
                axes[0, 0].plot(epochs, results.results_dict.get('train/box_loss', []), label='Box Loss')
                axes[0, 0].plot(epochs, results.results_dict.get('val/box_loss', []), label='Val Box Loss')
                axes[0, 0].set_title('Box Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
                
                axes[0, 1].plot(epochs, results.results_dict.get('train/seg_loss', []), label='Seg Loss')
                axes[0, 1].plot(epochs, results.results_dict.get('val/seg_loss', []), label='Val Seg Loss')
                axes[0, 1].set_title('Segmentation Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
                
                axes[1, 0].plot(epochs, results.results_dict.get('train/cls_loss', []), label='Cls Loss')
                axes[1, 0].plot(epochs, results.results_dict.get('val/cls_loss', []), label='Val Cls Loss')
                axes[1, 0].set_title('Classification Loss')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
                
                axes[1, 1].plot(epochs, results.results_dict.get('train/dfl_loss', []), label='DFL Loss')
                axes[1, 1].plot(epochs, results.results_dict.get('val/dfl_loss', []), label='Val DFL Loss')
                axes[1, 1].set_title('DFL Loss')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
                
                plt.tight_layout()
                wandb.log({"training_curves": wandb.Image(fig)})
                plt.close(fig)
                
        except Exception as e:
            logger.warning(f"Could not create visualizations for {model_name}: {str(e)}")

    def run_benchmark(self, data_yaml: str, configs: List[ModelConfig] = None, 
                     training_config: TrainingConfig = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across all model configurations
        
        Args:
            data_yaml: Path to data configuration YAML file
            configs: List of model configurations to test
            training_config: Training configuration
            
        Returns:
            Dictionary containing all benchmark results
        """
        if configs is None:
            configs = self.create_model_configs()
        
        if training_config is None:
            training_config = TrainingConfig(
                data_yaml=data_yaml,
                project_name="tree-canopy-segmentation"
            )
        
        logger.info(f"Starting benchmark with {len(configs)} model configurations")
        
        # Initialize W&B project
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=f"benchmark_{int(time.time())}",
            tags=["benchmark", "yolo", "segmentation", "tree-canopy"]
        )
        
        all_results = []
        
        for i, config in enumerate(configs):
            logger.info(f"Training model {i+1}/{len(configs)}: {config.name}")
            
            # Train model
            result = self.train_model(config, training_config)
            all_results.append(result)
            
            # Store results
            self.results[config.name] = result
            
            # Log progress
            wandb.log({
                "benchmark_progress": i + 1,
                "total_models": len(configs)
            })
        
        # Create comparison analysis
        self._create_comparison_analysis(all_results)
        
        # Finish W&B run
        wandb.finish()
        
        logger.info("Benchmark completed successfully")
        return self.results

    def _create_comparison_analysis(self, results: List[Dict[str, Any]]):
        """Create comprehensive comparison analysis and visualizations"""
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Remove error entries
            df = df[~df['model_name'].str.contains('error', na=False)]
            
            if df.empty:
                logger.warning("No valid results to analyze")
                return
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('YOLO Models Comparison for Tree Canopy Segmentation', fontsize=16)
            
            # Model size vs mAP50
            axes[0, 0].scatter(df['model_size_mb'], df['mAP50'], alpha=0.7)
            axes[0, 0].set_xlabel('Model Size (MB)')
            axes[0, 0].set_ylabel('mAP50')
            axes[0, 0].set_title('Model Size vs mAP50')
            axes[0, 0].grid(True)
            
            # Inference time vs mAP50
            axes[0, 1].scatter(df['inference_time_ms'], df['mAP50'], alpha=0.7)
            axes[0, 1].set_xlabel('Inference Time (ms)')
            axes[0, 1].set_ylabel('mAP50')
            axes[0, 1].set_title('Inference Time vs mAP50')
            axes[0, 1].grid(True)
            
            # Training time vs mAP50
            axes[0, 2].scatter(df['training_time'], df['mAP50'], alpha=0.7)
            axes[0, 2].set_xlabel('Training Time (s)')
            axes[0, 2].set_ylabel('mAP50')
            axes[0, 2].set_title('Training Time vs mAP50')
            axes[0, 2].grid(True)
            
            # mAP50 comparison
            df_sorted = df.sort_values('mAP50', ascending=True)
            axes[1, 0].barh(range(len(df_sorted)), df_sorted['mAP50'])
            axes[1, 0].set_yticks(range(len(df_sorted)))
            axes[1, 0].set_yticklabels(df_sorted['model_name'], fontsize=8)
            axes[1, 0].set_xlabel('mAP50')
            axes[1, 0].set_title('mAP50 by Model')
            axes[1, 0].grid(True)
            
            # Segmentation mAP50 comparison
            if 'seg_mAP50' in df.columns:
                df_seg_sorted = df.sort_values('seg_mAP50', ascending=True)
                axes[1, 1].barh(range(len(df_seg_sorted)), df_seg_sorted['seg_mAP50'])
                axes[1, 1].set_yticks(range(len(df_seg_sorted)))
                axes[1, 1].set_yticklabels(df_seg_sorted['model_name'], fontsize=8)
                axes[1, 1].set_xlabel('Segmentation mAP50')
                axes[1, 1].set_title('Segmentation mAP50 by Model')
                axes[1, 1].grid(True)
            
            # F1 Score comparison
            if 'f1' in df.columns:
                df_f1_sorted = df.sort_values('f1', ascending=True)
                axes[1, 2].barh(range(len(df_f1_sorted)), df_f1_sorted['f1'])
                axes[1, 2].set_yticks(range(len(df_f1_sorted)))
                axes[1, 2].set_yticklabels(df_f1_sorted['model_name'], fontsize=8)
                axes[1, 2].set_xlabel('F1 Score')
                axes[1, 2].set_title('F1 Score by Model')
                axes[1, 2].grid(True)
            
            plt.tight_layout()
            wandb.log({"model_comparison": wandb.Image(fig)})
            plt.close(fig)
            
            # Create summary table
            summary_table = wandb.Table(
                columns=["Model", "mAP50", "Seg mAP50", "F1", "Model Size (MB)", "Inference Time (ms)", "Training Time (s)"],
                data=df[['model_name', 'mAP50', 'seg_mAP50', 'f1', 'model_size_mb', 'inference_time_ms', 'training_time']].values.tolist()
            )
            wandb.log({"model_summary": summary_table})
            
            # Find best models
            best_overall = df.loc[df['mAP50'].idxmax()]
            best_seg = df.loc[df['seg_mAP50'].idxmax()] if 'seg_mAP50' in df.columns else None
            best_speed = df.loc[df['inference_time_ms'].idxmin()]
            
            wandb.log({
                "best_overall_model": best_overall['model_name'],
                "best_overall_mAP50": best_overall['mAP50'],
                "best_segmentation_model": best_seg['model_name'] if best_seg is not None else "N/A",
                "best_segmentation_mAP50": best_seg['seg_mAP50'] if best_seg is not None else 0,
                "fastest_model": best_speed['model_name'],
                "fastest_inference_time": best_speed['inference_time_ms']
            })
            
        except Exception as e:
            logger.error(f"Error creating comparison analysis: {str(e)}")

    def save_results(self, output_path: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

def main():
    """Main function to run the benchmark"""
    # Set up API key
    os.environ['WANDB_API_KEY'] = 'f21e18e97e313644edf2723adf692971cec13175'
    
    # Initialize benchmarker
    benchmarker = YOLOBenchmarker(
        wandb_project="tree-canopy-segmentation",
        wandb_entity="paapi"
    )
    
    # Data configuration (update this path as needed)
    data_yaml = "/home/arnavmehta/clg/solafune_tree_canopy/configurations/model_data-seg.yaml"
    
    # Create custom configurations if needed
    custom_configs = benchmarker.create_model_configs()
    
    # Run benchmark
    results = benchmarker.run_benchmark(data_yaml, custom_configs)
    
    # Save results
    benchmarker.save_results()
    
    print("Benchmark completed! Check W&B dashboard for detailed results.")

if __name__ == "__main__":
    main()