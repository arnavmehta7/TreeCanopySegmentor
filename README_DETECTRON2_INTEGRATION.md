# Detectron2 Integration for Tree Canopy Segmentation

This repository now includes comprehensive Detectron2 model training and inference capabilities with Weights & Biases (W&B) integration for experiment tracking and model comparison.

## Features

- **Multiple Detectron2 Models**: Mask R-CNN, Cascade Mask R-CNN, and RetinaNet
- **W&B Integration**: Full experiment tracking, visualization, and model comparison
- **Flexible Training**: Support for different model architectures and configurations
- **Export Compatibility**: Results exported in the same format as YOLO models
- **Easy Setup**: Automated environment setup and dependency management

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies and set up directories
python setup_detectron2.py --mode setup

# Or run everything at once
python setup_detectron2.py --mode all
```

### 2. Test Installation

```bash
# Test Detectron2 installation
python setup_detectron2.py --mode test
```

### 3. Run Quick Training

```bash
# Test with a short training run
python setup_detectron2.py --mode quick
```

### 4. Run Full Training

```bash
# Train multiple models with full configurations
python setup_detectron2.py --mode full
```

### 5. Run Inference and Export

```bash
# Generate predictions and export in required format
python setup_detectron2.py --mode inference
```

## Manual Usage

### Basic Training

```bash
# Train specific models
python train_detectron2_models.py --models mask_rcnn_R_50_FPN_3x mask_rcnn_R_101_FPN_3x

# Train with custom data path
python train_detectron2_models.py --data-yaml /path/to/your/data.yaml
```

### Export Results

```bash
# Export predictions in the required format
python 07_export_detectron2_submission.py
```

## Supported Models

### Mask R-CNN Models
- **mask_rcnn_R_50_FPN_3x**: ResNet-50 backbone with FPN
- **mask_rcnn_R_101_FPN_3x**: ResNet-101 backbone with FPN
- **mask_rcnn_X_101_32x8d_FPN_3x**: ResNeXt-101 backbone with FPN

### Cascade Mask R-CNN Models
- **cascade_mask_rcnn_R_50_FPN_3x**: Cascade Mask R-CNN with ResNet-50
- **cascade_mask_rcnn_R_101_FPN_3x**: Cascade Mask R-CNN with ResNet-101

### RetinaNet Models
- **retinanet_R_50_FPN_3x**: RetinaNet with ResNet-50
- **retinanet_R_101_FPN_3x**: RetinaNet with ResNet-101

## Configuration

### Training Configuration

The training script accepts several parameters:

```bash
python train_detectron2_models.py \
    --data-yaml /path/to/data.yaml \
    --models mask_rcnn_R_50_FPN_3x mask_rcnn_R_101_FPN_3x \
    --wandb-project tree-canopy-detectron2 \
    --wandb-entity paapi
```

### Model Configuration

Each model can be configured with:

- **ims_per_batch**: Images per batch (1-4 depending on GPU memory)
- **base_lr**: Base learning rate (0.0001-0.001)
- **max_iter**: Maximum training iterations (1000-5000)
- **checkpoint_period**: How often to save checkpoints

## W&B Integration

### Project Structure

- **Main Project**: `tree-canopy-detectron2`
- **Entity**: `paapi`

### What's Tracked

- Training metrics (loss, mAP, precision, recall)
- Model performance comparisons
- Training curves and visualizations
- Model artifacts and checkpoints
- System metrics (GPU usage, memory, etc.)

### Accessing Results

1. Visit [wandb.ai](https://wandb.ai)
2. Navigate to the `paapi/tree-canopy-detectron2` project
3. View runs, compare models, and analyze results

## Export Format

The export script generates results in the exact same format as your YOLO export script:

```json
{
  "images": [
    {
      "file_name": "image.tif",
      "width": 1024,
      "height": 1024,
      "cm_resolution": "10",
      "scene_type": "forest",
      "annotations": [
        {
          "class": 0,
          "confidence_score": 0.95,
          "segmentation": [x1, y1, x2, y2, ...]
        }
      ]
    }
  ]
}
```

## File Structure

```
/workspace/
├── train_detectron2_models.py          # Main training script
├── 07_export_detectron2_submission.py  # Export script (matches your format)
├── setup_detectron2.py                 # Setup and execution script
├── configurations/
│   └── model_data-seg.yaml            # Dataset configuration
├── outputs/                            # Model checkpoints and logs
├── exports/                            # Exported results
└── README_DETECTRON2_INTEGRATION.md   # This file
```

## Training Modes

### 1. Quick Test
- Short training run (100 iterations)
- Tests basic functionality
- Good for debugging

### 2. Full Training
- Complete training with multiple models
- Full evaluation and comparison
- Generates final results

### 3. Inference Only
- Loads trained model
- Runs inference on test images
- Exports results in required format

## Performance Tips

1. **GPU Memory**: Adjust `ims_per_batch` based on your GPU memory
2. **Training Time**: More iterations = better results but longer training
3. **Model Size**: Larger models (R-101) perform better but are slower
4. **Data Augmentation**: Automatically applied by Detectron2

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `ims_per_batch` or use smaller models
2. **Detectron2 Installation**: May need to install from source
3. **Dataset Format**: Ensure annotations are in YOLO format
4. **W&B Login**: Check API key and internet connection

### Debug Mode

```bash
# Run with debug logging
PYTHONPATH=. python -u train_detectron2_models.py --models mask_rcnn_R_50_FPN_3x 2>&1 | tee debug.log
```

## Comparison with YOLO

### Detectron2 Advantages
- **Better Segmentation**: More accurate mask generation
- **Research-Grade**: State-of-the-art instance segmentation
- **Flexible Architecture**: Easy to modify and extend
- **Robust Training**: More stable training process

### YOLO Advantages
- **Faster Inference**: Real-time performance
- **Smaller Models**: Less memory usage
- **Easier Deployment**: Simpler model format
- **Better for Detection**: Faster object detection

## Next Steps

1. **Data Preparation**: Ensure your dataset is properly formatted
2. **Model Selection**: Choose appropriate models for your use case
3. **Training**: Run full training pipeline
4. **Evaluation**: Compare results with YOLO models
5. **Deployment**: Use best performing model for inference

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify Detectron2 installation
3. Ensure dataset format is correct
4. Check W&B project access and API key

## Integration with Existing Workflow

The Detectron2 integration is designed to work alongside your existing YOLO workflow:

1. **Same Data Format**: Uses the same YOLO annotation format
2. **Same Export Format**: Generates identical submission files
3. **W&B Integration**: Tracks experiments in the same project
4. **Easy Comparison**: Compare YOLO vs Detectron2 results

This allows you to easily benchmark both approaches and choose the best performing model for your tree canopy segmentation task.