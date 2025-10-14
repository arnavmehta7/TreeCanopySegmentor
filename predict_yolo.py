import ultralytics
from ultralytics import YOLO
import numpy as np
import time
import yaml
from pathlib import Path
import torch

# Assign directory location
REPO_ROOT = Path(__file__).resolve().parent.parent


# Load model parameters / overrides YAML
with open(REPO_ROOT / 'configurations' / 'predict_model_overrides.yaml', 'r') as f: # Modify overrides where necessary
    overrides = yaml.safe_load(f)

# Make YAML Paths Absolute
for key in ("source", "project"):
    if key in overrides:
        overrides[key] = str((REPO_ROOT / overrides[key]).resolve())

# Load Trained Model Weights
# weights = REPO_ROOT / 'runs/segment/training_fastNMS12/weights/best.pt' # Weights from Train Model / Modify where necessary
weights = Path('/home/arnavmehta/clg/solafune_tree_canopy/runs/segment/train_Yolo11s_canopy_832_3/weights/best.pt')
# uns/segment/train_Yolo11s_canopy_832_8/weights/best.pt
# runs/segment/train_Yolo11s_canopy_832_13
# Load Trained Model's Weights
model = YOLO(str(weights))  
                                                                                                                                                                            
# Predictions from the model
with torch.inference_mode():
    predictions = model.predict(**overrides)

# State output location

# Stream = True
for last in predictions:
    pass
print("Saved to:", last.save_dir)

# Stream = False
# print("Outputs saved to:", predictions[0].save_dir)
