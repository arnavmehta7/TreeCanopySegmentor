#!/usr/bin/env python3
"""
Setup and run script for YOLO Tree Canopy Segmentation training
This script handles environment setup and provides easy entry points for different training modes.
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
    logger.info("Setting up environment...")
    
    # Set environment variables
    os.environ['WANDB_API_KEY'] = 'f21e18e97e313644edf2723adf692971cec13175'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
    
    # Install requirements
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'runs/segment',
        'runs/hyperopt',
        'data/images/train',
        'data/images/val',
        'data/images/test',
        'configurations',
        'results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_quick_test():
    """Run a quick test to verify everything is working"""
    logger.info("Running quick test...")
    
    try:
        from train_yolo_models import YOLOTrainer
        
        # Initialize trainer
        trainer = YOLOTrainer()
        
        # Run quick benchmark
        results = trainer.run_quick_benchmark()
        
        logger.info("Quick test completed successfully!")
        logger.info(f"Results: {results}")
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return False

def run_hyperparameter_optimization():
    """Run hyperparameter optimization for key models"""
    logger.info("Running hyperparameter optimization...")
    
    try:
        from train_yolo_models import YOLOTrainer
        
        # Initialize trainer
        trainer = YOLOTrainer()
        
        # Models to optimize
        models = ['yolov8s-seg.pt', 'yolo11s-seg.pt']
        
        for model in models:
            logger.info(f"Optimizing {model}...")
            results = trainer.run_hyperparameter_optimization(model)
            logger.info(f"Optimization completed for {model}")
        
        return True
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        return False

def run_full_training():
    """Run full training pipeline"""
    logger.info("Running full training pipeline...")
    
    try:
        from train_yolo_models import YOLOTrainer
        
        # Initialize trainer
        trainer = YOLOTrainer()
        
        # Run complete pipeline
        results = trainer.run_complete_pipeline()
        
        logger.info("Full training pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Full training failed: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup and run YOLO training')
    parser.add_argument('--mode', type=str, 
                       choices=['setup', 'quick', 'hyperopt', 'full', 'all'],
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
        logger.info("Setup completed. You can now run training with --mode quick, hyperopt, or full")
        return
    
    elif args.mode == 'quick':
        if not run_quick_test():
            sys.exit(1)
    
    elif args.mode == 'hyperopt':
        if not run_hyperparameter_optimization():
            sys.exit(1)
    
    elif args.mode == 'full':
        if not run_full_training():
            sys.exit(1)
    
    elif args.mode == 'all':
        # Run everything in sequence
        if not run_quick_test():
            logger.error("Quick test failed, stopping")
            sys.exit(1)
        
        if not run_hyperparameter_optimization():
            logger.error("Hyperparameter optimization failed, stopping")
            sys.exit(1)
        
        if not run_full_training():
            logger.error("Full training failed, stopping")
            sys.exit(1)
    
    logger.info("All tasks completed successfully!")

if __name__ == "__main__":
    main()