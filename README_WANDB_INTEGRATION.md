# YOLO Tree Canopy Segmentation with W&B Integration

This repository now includes comprehensive YOLO model training and benchmarking capabilities with Weights & Biases (W&B) integration for experiment tracking and model comparison.

## Features

- **Comprehensive Model Benchmarking**: Test YOLOv8 and YOLOv11 models with different configurations
- **Hyperparameter Optimization**: Automated hyperparameter tuning using Optuna
- **W&B Integration**: Full experiment tracking, visualization, and model comparison
- **Multiple Training Modes**: Quick testing, full benchmarking, and hyperparameter optimization
- **Flexible Configuration**: Easy-to-modify configuration files for different experiments

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies and set up directories
python setup_and_run.py --mode setup

# Or run everything at once
python setup_and_run.py --mode all
```

### 2. Run Quick Test

```bash
# Test with a few models to verify everything works
python setup_and_run.py --mode quick
```

### 3. Run Hyperparameter Optimization

```bash
# Optimize hyperparameters for key models
python setup_and_run.py --mode hyperopt
```

### 4. Run Full Training Pipeline

```bash
# Run complete training with all models and optimizations
python setup_and_run.py --mode full
```

## Manual Usage

### Basic Training

```bash
# Quick benchmark with default settings
python train_yolo_models.py --mode quick

# Full benchmark with all model variants
python train_yolo_models.py --mode full

# Hyperparameter optimization for specific model
python train_yolo_models.py --mode hyperopt --model yolov8s-seg.pt

# Complete pipeline (recommended)
python train_yolo_models.py --mode complete
```

### Hyperparameter Optimization Only

```bash
python hyperparameter_optimization.py
```

### Custom Benchmarking

```bash
python yolo_benchmark_wandb.py
```

## Configuration

### Training Configuration

Edit `training_config.yaml` to customize:

- Models to train
- Input sizes to test
- Batch sizes
- Training epochs
- Hardware settings
- W&B project settings

### Data Configuration

Update `configurations/model_data-seg.yaml` with your dataset paths:

```yaml
path: /path/to/your/dataset
train: images/train
val: images/val
test: images/test
nc: 1
names: ['tree_canopy']
```

## W&B Integration

### Project Structure

- **Main Project**: `tree-canopy-segmentation`
- **Hyperparameter Optimization**: `tree-canopy-hyperopt`
- **Entity**: `paapi`

### What's Tracked

- Training metrics (loss, mAP, precision, recall, F1)
- Model performance comparisons
- Hyperparameter optimization results
- Training curves and visualizations
- Model artifacts and checkpoints
- System metrics (GPU usage, memory, etc.)

### Accessing Results

1. Visit [wandb.ai](https://wandb.ai)
2. Navigate to the `paapi/tree-canopy-segmentation` project
3. View runs, compare models, and analyze results

## Model Variants Tested

### YOLOv8 Models
- YOLOv8n-seg (nano)
- YOLOv8s-seg (small)
- YOLOv8m-seg (medium)
- YOLOv8l-seg (large)
- YOLOv8x-seg (extra large)

### YOLOv11 Models
- YOLOv11n-seg (nano)
- YOLOv11s-seg (small)
- YOLOv11m-seg (medium)
- YOLOv11l-seg (large)
- YOLOv11x-seg (extra large)

### Special Variants
- High resolution (1024px input)
- Aggressive augmentation
- AdamW optimizer variants
- Custom hyperparameter combinations

## Hyperparameter Optimization

The system automatically optimizes:

- Learning rate and weight decay
- Optimizer choice (SGD, Adam, AdamW, RMSprop)
- Data augmentation parameters
- Loss function weights
- Input image size
- Batch size
- Dropout rate
- Learning rate scheduling

## Results and Analysis

### Generated Files

- `benchmark_results.json`: Complete benchmark results
- `optimized_config_*.yaml`: Optimized configurations for each model
- `complete_pipeline_results.json`: Full pipeline results
- `hyperopt_study_*.pkl`: Optuna study objects

### W&B Dashboards

- Model comparison tables
- Training curve visualizations
- Hyperparameter importance plots
- Performance vs. model size analysis
- Inference time comparisons

## Customization

### Adding New Models

1. Add model to `training_config.yaml`
2. Create custom `ModelConfig` in `yolo_benchmark_wandb.py`
3. Run training with new configuration

### Custom Hyperparameters

1. Modify the search space in `hyperparameter_optimization.py`
2. Adjust optimization parameters in `training_config.yaml`
3. Run hyperparameter optimization

### Custom Metrics

1. Add new metrics to the training functions
2. Update visualization code
3. Modify W&B logging

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or input size
2. **W&B Login Issues**: Check API key and internet connection
3. **Dataset Path Errors**: Update paths in configuration files
4. **Model Download Issues**: Check internet connection and model availability

### Debug Mode

```bash
# Run with debug logging
PYTHONPATH=. python -u train_yolo_models.py --mode quick 2>&1 | tee debug.log
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is available and properly configured
2. **Batch Size**: Start with smaller batch sizes and increase gradually
3. **Input Size**: Start with 640px and increase if needed
4. **Memory**: Use `cache: ram` for faster training if you have enough RAM
5. **Workers**: Adjust based on your CPU cores

## File Structure

```
/workspace/
├── yolo_benchmark_wandb.py          # Main benchmarking script
├── hyperparameter_optimization.py   # Hyperparameter optimization
├── train_yolo_models.py             # Main training orchestrator
├── setup_and_run.py                 # Easy setup and execution
├── training_config.yaml             # Training configuration
├── configurations/
│   └── model_data-seg.yaml         # Dataset configuration
├── requirements.txt                 # Dependencies
└── README_WANDB_INTEGRATION.md     # This file
```

## Next Steps

1. **Data Preparation**: Ensure your dataset is properly formatted
2. **Configuration**: Update configuration files for your specific needs
3. **Quick Test**: Run a quick test to verify everything works
4. **Full Training**: Run the complete pipeline
5. **Analysis**: Use W&B dashboards to analyze results and select best models

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify configuration files are correct
3. Ensure all dependencies are installed
4. Check W&B project access and API key