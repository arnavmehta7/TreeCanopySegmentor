#!/usr/bin/env python3
"""
Setup script for Detectron2 training and inference
Handles environment setup, dependency installation, and provides easy entry points.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment and install dependencies"""
    logger.info("Setting up Detectron2 environment...")
    
    # Set environment variables
    os.environ['WANDB_API_KEY'] = 'f21e18e97e313644edf2723adf692971cec13175'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
    
    # Install PyTorch (if not already installed)
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        logger.info("Installing PyTorch...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision'], 
                      check=True, capture_output=True)
    
    # Install Detectron2
    try:
        import detectron2
        logger.info(f"Detectron2 version: {detectron2.__version__}")
    except ImportError:
        logger.info("Installing Detectron2...")
        # Install from source for better compatibility
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'git+https://github.com/facebookresearch/detectron2.git'
        ], check=True, capture_output=True)
    
    # Install other dependencies
    dependencies = [
        'opencv-python',
        'pycocotools',
        'wandb',
        'weave',
        'pyyaml',
        'pandas',
        'numpy',
        'pillow',
        'matplotlib',
        'seaborn'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
            logger.info(f"{dep} already installed")
        except ImportError:
            logger.info(f"Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                          check=True, capture_output=True)
    
    logger.info("Environment setup completed successfully")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'outputs',
        'data/processed/images/predict',
        'data/processed/images/train',
        'data/processed/images/val',
        'configurations',
        'exports',
        'results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def test_detectron2():
    """Test Detectron2 installation"""
    logger.info("Testing Detectron2 installation...")
    
    try:
        import detectron2
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        
        # Test basic functionality
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        logger.info("Detectron2 installation test passed")
        return True
        
    except Exception as e:
        logger.error(f"Detectron2 installation test failed: {e}")
        return False

def run_quick_training():
    """Run a quick training test"""
    logger.info("Running quick training test...")
    
    try:
        from train_detectron2_models import Detectron2Trainer
        
        # Initialize trainer
        trainer = Detectron2Trainer()
        
        # Test with a simple configuration
        result = trainer.train_model(
            model_name="mask_rcnn_R_50_FPN_3x",
            data_yaml="/workspace/configurations/model_data-seg.yaml",
            ims_per_batch=1,
            base_lr=0.00025,
            max_iter=100  # Very short training for test
        )
        
        logger.info("Quick training test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Quick training test failed: {e}")
        return False

def run_full_training():
    """Run full training pipeline"""
    logger.info("Running full Detectron2 training pipeline...")
    
    try:
        from train_detectron2_models import Detectron2Trainer
        
        # Initialize trainer
        trainer = Detectron2Trainer()
        
        # Run benchmark
        results = trainer.run_benchmark(
            data_yaml="/workspace/configurations/model_data-seg.yaml",
            models=[
                "mask_rcnn_R_50_FPN_3x",
                "mask_rcnn_R_101_FPN_3x",
                "cascade_mask_rcnn_R_50_FPN_3x"
            ]
        )
        
        logger.info("Full training pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Full training failed: {e}")
        return False

def run_inference():
    """Run inference and export results"""
    logger.info("Running inference and export...")
    
    try:
        # Find the best model
        model_path = None
        for path in Path('outputs').glob('**/model_final.pth'):
            if 'mask_rcnn' in str(path):
                model_path = str(path)
                break
        
        if model_path is None:
            logger.error("No trained model found")
            return False
        
        # Run export script
        subprocess.run([
            sys.executable, '07_export_detectron2_submission.py'
        ], check=True)
        
        logger.info("Inference and export completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Inference and export failed: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup and run Detectron2 training')
    parser.add_argument('--mode', type=str, 
                       choices=['setup', 'test', 'quick', 'full', 'inference', 'all'],
                       default='setup',
                       help='Mode to run')
    parser.add_argument('--skip-setup', action='store_true',
                       help='Skip environment setup')
    
    args = parser.parse_args()
    
    if not args.skip_setup:
        # Setup environment
        if not setup_environment():
            logger.error("Environment setup failed")
            sys.exit(1)
        
        # Create directories
        create_directories()
        logger.info("Environment setup completed!")
    
    # Run based on mode
    if args.mode == 'setup':
        logger.info("Setup completed. You can now run training with --mode test, quick, full, or inference")
        return
    
    elif args.mode == 'test':
        if not test_detectron2():
            sys.exit(1)
    
    elif args.mode == 'quick':
        if not run_quick_training():
            sys.exit(1)
    
    elif args.mode == 'full':
        if not run_full_training():
            sys.exit(1)
    
    elif args.mode == 'inference':
        if not run_inference():
            sys.exit(1)
    
    elif args.mode == 'all':
        # Run everything in sequence
        if not test_detectron2():
            logger.error("Detectron2 test failed, stopping")
            sys.exit(1)
        
        if not run_quick_training():
            logger.error("Quick training failed, stopping")
            sys.exit(1)
        
        if not run_full_training():
            logger.error("Full training failed, stopping")
            sys.exit(1)
        
        if not run_inference():
            logger.error("Inference failed, stopping")
            sys.exit(1)
    
    logger.info("All tasks completed successfully!")

if __name__ == "__main__":
    main()