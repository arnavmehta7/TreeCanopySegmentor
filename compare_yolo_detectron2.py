#!/usr/bin/env python3
"""
Comprehensive comparison between YOLO and Detectron2 models for Tree Canopy Segmentation
Runs both frameworks and provides detailed comparison metrics.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# W&B imports
import wandb
import weave

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparison:
    """Comprehensive comparison between YOLO and Detectron2 models"""
    
    def __init__(self, wandb_project: str = "tree-canopy-comparison", 
                 wandb_entity: str = "paapi", api_key: str = None):
        """
        Initialize the model comparison
        
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
        
        # Set random seeds
        np.random.seed(42)
        
        logger.info(f"Initialized Model Comparison for project: {wandb_project}")

    def run_yolo_training(self, data_yaml: str) -> Dict[str, Any]:
        """Run YOLO training and return results"""
        logger.info("Running YOLO training...")
        
        try:
            from train_yolo_models import YOLOTrainer
            
            # Initialize YOLO trainer
            yolo_trainer = YOLOTrainer(
                wandb_project="tree-canopy-yolo-comparison",
                wandb_entity=self.wandb_entity
            )
            
            # Run quick benchmark
            results = yolo_trainer.run_quick_benchmark()
            
            logger.info("YOLO training completed")
            return results
            
        except Exception as e:
            logger.error(f"YOLO training failed: {str(e)}")
            return {"error": str(e)}

    def run_detectron2_training(self, data_yaml: str) -> Dict[str, Any]:
        """Run Detectron2 training and return results"""
        logger.info("Running Detectron2 training...")
        
        try:
            from train_detectron2_models import Detectron2Trainer
            
            # Initialize Detectron2 trainer
            detectron2_trainer = Detectron2Trainer(
                wandb_project="tree-canopy-detectron2-comparison",
                wandb_entity=self.wandb_entity
            )
            
            # Run training
            results = detectron2_trainer.run_benchmark(
                data_yaml=data_yaml,
                models=["mask_rcnn_R_50_FPN_3x", "mask_rcnn_R_101_FPN_3x"]
            )
            
            logger.info("Detectron2 training completed")
            return results
            
        except Exception as e:
            logger.error(f"Detectron2 training failed: {str(e)}")
            return {"error": str(e)}

    def compare_results(self, yolo_results: Dict[str, Any], 
                       detectron2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results from both frameworks"""
        logger.info("Comparing results...")
        
        comparison = {
            "yolo_results": yolo_results,
            "detectron2_results": detectron2_results,
            "comparison_metrics": {},
            "recommendations": []
        }
        
        try:
            # Extract metrics for comparison
            yolo_metrics = self._extract_yolo_metrics(yolo_results)
            detectron2_metrics = self._extract_detectron2_metrics(detectron2_results)
            
            # Compare metrics
            comparison["comparison_metrics"] = {
                "yolo_best_mAP50": yolo_metrics.get("best_mAP50", 0),
                "detectron2_best_mAP50": detectron2_metrics.get("best_mAP50", 0),
                "yolo_best_seg_mAP50": yolo_metrics.get("best_seg_mAP50", 0),
                "detectron2_best_seg_mAP50": detectron2_metrics.get("best_seg_mAP50", 0),
                "yolo_training_time": yolo_metrics.get("total_training_time", 0),
                "detectron2_training_time": detectron2_metrics.get("total_training_time", 0),
                "yolo_model_size": yolo_metrics.get("best_model_size", 0),
                "detectron2_model_size": detectron2_metrics.get("best_model_size", 0)
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(comparison["comparison_metrics"])
            comparison["recommendations"] = recommendations
            
            logger.info("Comparison completed")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing results: {str(e)}")
            comparison["error"] = str(e)
            return comparison

    def _extract_yolo_metrics(self, yolo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from YOLO results"""
        metrics = {
            "best_mAP50": 0,
            "best_seg_mAP50": 0,
            "total_training_time": 0,
            "best_model_size": 0,
            "best_model_name": ""
        }
        
        if isinstance(yolo_results, dict) and "error" not in yolo_results:
            # Find best model
            best_model = None
            best_mAP50 = 0
            
            for model_name, result in yolo_results.items():
                if isinstance(result, dict) and "mAP50" in result:
                    if result["mAP50"] > best_mAP50:
                        best_mAP50 = result["mAP50"]
                        best_model = result
            
            if best_model:
                metrics["best_mAP50"] = best_model.get("mAP50", 0)
                metrics["best_seg_mAP50"] = best_model.get("seg_mAP50", 0)
                metrics["total_training_time"] = best_model.get("training_time", 0)
                metrics["best_model_size"] = best_model.get("model_size_mb", 0)
                metrics["best_model_name"] = best_model.get("model_name", "")
        
        return metrics

    def _extract_detectron2_metrics(self, detectron2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from Detectron2 results"""
        metrics = {
            "best_mAP50": 0,
            "best_seg_mAP50": 0,
            "total_training_time": 0,
            "best_model_size": 0,
            "best_model_name": ""
        }
        
        if isinstance(detectron2_results, dict) and "error" not in detectron2_results:
            # Find best model
            best_model = None
            best_mAP50 = 0
            
            for model_name, result in detectron2_results.items():
                if isinstance(result, dict) and "bbox_mAP50" in result:
                    if result["bbox_mAP50"] > best_mAP50:
                        best_mAP50 = result["bbox_mAP50"]
                        best_model = result
            
            if best_model:
                metrics["best_mAP50"] = best_model.get("bbox_mAP50", 0)
                metrics["best_seg_mAP50"] = best_model.get("segm_mAP50", 0)
                metrics["total_training_time"] = best_model.get("training_time", 0)
                metrics["best_model_size"] = 0  # Detectron2 model size not easily available
                metrics["best_model_name"] = best_model.get("model_name", "")
        
        return metrics

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison metrics"""
        recommendations = []
        
        yolo_mAP50 = metrics.get("yolo_best_mAP50", 0)
        detectron2_mAP50 = metrics.get("detectron2_best_mAP50", 0)
        yolo_seg_mAP50 = metrics.get("yolo_best_seg_mAP50", 0)
        detectron2_seg_mAP50 = metrics.get("detectron2_best_seg_mAP50", 0)
        yolo_time = metrics.get("yolo_training_time", 0)
        detectron2_time = metrics.get("detectron2_training_time", 0)
        
        # Accuracy recommendations
        if detectron2_mAP50 > yolo_mAP50:
            recommendations.append(f"Detectron2 shows better overall mAP50 ({detectron2_mAP50:.3f} vs {yolo_mAP50:.3f})")
        else:
            recommendations.append(f"YOLO shows better overall mAP50 ({yolo_mAP50:.3f} vs {detectron2_mAP50:.3f})")
        
        if detectron2_seg_mAP50 > yolo_seg_mAP50:
            recommendations.append(f"Detectron2 shows better segmentation mAP50 ({detectron2_seg_mAP50:.3f} vs {yolo_seg_mAP50:.3f})")
        else:
            recommendations.append(f"YOLO shows better segmentation mAP50 ({yolo_seg_mAP50:.3f} vs {detectron2_seg_mAP50:.3f})")
        
        # Speed recommendations
        if yolo_time < detectron2_time:
            recommendations.append(f"YOLO trains faster ({yolo_time:.1f}s vs {detectron2_time:.1f}s)")
        else:
            recommendations.append(f"Detectron2 trains faster ({detectron2_time:.1f}s vs {yolo_time:.1f}s)")
        
        # Use case recommendations
        if detectron2_seg_mAP50 > yolo_seg_mAP50 * 1.1:  # 10% better
            recommendations.append("For high-accuracy segmentation, use Detectron2")
        elif yolo_mAP50 > detectron2_mAP50 * 1.05:  # 5% better
            recommendations.append("For balanced performance and speed, use YOLO")
        
        if yolo_time < detectron2_time * 0.5:  # 2x faster
            recommendations.append("For quick prototyping and development, use YOLO")
        
        return recommendations

    def create_comparison_visualizations(self, comparison: Dict[str, Any]) -> None:
        """Create comparison visualizations"""
        logger.info("Creating comparison visualizations...")
        
        try:
            metrics = comparison["comparison_metrics"]
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('YOLO vs Detectron2 Comparison for Tree Canopy Segmentation', fontsize=16)
            
            # mAP50 comparison
            frameworks = ['YOLO', 'Detectron2']
            mAP50_scores = [metrics.get('yolo_best_mAP50', 0), metrics.get('detectron2_best_mAP50', 0)]
            
            axes[0, 0].bar(frameworks, mAP50_scores, color=['blue', 'red'], alpha=0.7)
            axes[0, 0].set_title('Overall mAP50 Comparison')
            axes[0, 0].set_ylabel('mAP50')
            axes[0, 0].set_ylim(0, 1)
            for i, v in enumerate(mAP50_scores):
                axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Segmentation mAP50 comparison
            seg_mAP50_scores = [metrics.get('yolo_best_seg_mAP50', 0), metrics.get('detectron2_best_seg_mAP50', 0)]
            
            axes[0, 1].bar(frameworks, seg_mAP50_scores, color=['blue', 'red'], alpha=0.7)
            axes[0, 1].set_title('Segmentation mAP50 Comparison')
            axes[0, 1].set_ylabel('Segmentation mAP50')
            axes[0, 1].set_ylim(0, 1)
            for i, v in enumerate(seg_mAP50_scores):
                axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Training time comparison
            training_times = [metrics.get('yolo_training_time', 0), metrics.get('detectron2_training_time', 0)]
            
            axes[1, 0].bar(frameworks, training_times, color=['blue', 'red'], alpha=0.7)
            axes[1, 0].set_title('Training Time Comparison')
            axes[1, 0].set_ylabel('Training Time (seconds)')
            for i, v in enumerate(training_times):
                axes[1, 0].text(i, v + max(training_times) * 0.01, f'{v:.1f}s', ha='center', va='bottom')
            
            # Model size comparison (if available)
            model_sizes = [metrics.get('yolo_model_size', 0), metrics.get('detectron2_model_size', 0)]
            
            if any(model_sizes):
                axes[1, 1].bar(frameworks, model_sizes, color=['blue', 'red'], alpha=0.7)
                axes[1, 1].set_title('Model Size Comparison')
                axes[1, 1].set_ylabel('Model Size (MB)')
                for i, v in enumerate(model_sizes):
                    if v > 0:
                        axes[1, 1].text(i, v + max(model_sizes) * 0.01, f'{v:.1f}MB', ha='center', va='bottom')
            else:
                axes[1, 1].text(0.5, 0.5, 'Model size data\nnot available', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Model Size Comparison')
            
            plt.tight_layout()
            
            # Save plot
            plt.savefig('yolo_detectron2_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Comparison visualizations created")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")

    def run_complete_comparison(self, data_yaml: str) -> Dict[str, Any]:
        """Run complete comparison between YOLO and Detectron2"""
        logger.info("Starting complete model comparison...")
        
        # Initialize W&B run
        run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=f"comparison_{int(time.time())}",
            tags=["comparison", "yolo", "detectron2", "segmentation", "tree-canopy"]
        )
        
        try:
            # Run YOLO training
            yolo_results = self.run_yolo_training(data_yaml)
            
            # Run Detectron2 training
            detectron2_results = self.run_detectron2_training(data_yaml)
            
            # Compare results
            comparison = self.compare_results(yolo_results, detectron2_results)
            
            # Create visualizations
            self.create_comparison_visualizations(comparison)
            
            # Log to W&B
            wandb.log(comparison["comparison_metrics"])
            
            # Save results
            with open('model_comparison_results.json', 'w') as f:
                json.dump(comparison, f, indent=2)
            
            logger.info("Complete comparison finished")
            return comparison
            
        except Exception as e:
            logger.error(f"Error in complete comparison: {str(e)}")
            wandb.log({"error": str(e)})
            return {"error": str(e)}
        finally:
            wandb.finish()

def main():
    """Main function to run model comparison"""
    parser = argparse.ArgumentParser(description='Compare YOLO and Detectron2 models')
    parser.add_argument('--data-yaml', type=str, default='/workspace/configurations/model_data-seg.yaml',
                       help='Path to data configuration YAML')
    parser.add_argument('--wandb-project', type=str, default='tree-canopy-comparison',
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
    
    # Initialize comparison
    comparison = ModelComparison(
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        api_key=args.api_key
    )
    
    # Run complete comparison
    results = comparison.run_complete_comparison(args.data_yaml)
    
    # Print summary
    if "error" not in results:
        metrics = results["comparison_metrics"]
        recommendations = results["recommendations"]
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(f"YOLO Best mAP50: {metrics.get('yolo_best_mAP50', 0):.3f}")
        print(f"Detectron2 Best mAP50: {metrics.get('detectron2_best_mAP50', 0):.3f}")
        print(f"YOLO Best Seg mAP50: {metrics.get('yolo_best_seg_mAP50', 0):.3f}")
        print(f"Detectron2 Best Seg mAP50: {metrics.get('detectron2_best_seg_mAP50', 0):.3f}")
        print(f"YOLO Training Time: {metrics.get('yolo_training_time', 0):.1f}s")
        print(f"Detectron2 Training Time: {metrics.get('detectron2_training_time', 0):.1f}s")
        
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*60)
    else:
        print(f"Comparison failed: {results['error']}")

if __name__ == "__main__":
    main()