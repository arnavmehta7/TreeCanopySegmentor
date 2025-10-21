# Packages
from pathlib import Path
import pandas as pd
import numpy as np
import json
from PIL import Image
import pandas as pd
from datetime import datetime
import cv2
import torch
import yaml

# Detectron2 imports
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

# Create paths
REPO_ROOT = Path.cwd()

CURRENT_DATETIME = datetime.now().strftime('%Y-%m-%d %H:%M')

# Model configuration
MODEL_CONFIG = {
    'model_path': 'outputs/mask_rcnn_R_50_FPN_3x_*/model_final.pth',  # Update with actual path
    'config_file': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    'confidence_threshold': 0.5
}

# Prediction paths
IMGS = REPO_ROOT / 'data/processed/images/predict'
OUT_JSON = REPO_ROOT / 'exports' / str(CURRENT_DATETIME) / 'detectron2_submission.json'

#---Define Functions---#

def load_detectron2_model(model_path: str, config_file: str, confidence_threshold: float = 0.5):
    """Load Detectron2 model for inference"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # tree_canopy class
    
    # Set metadata
    MetadataCatalog.get("tree_canopy").set(thing_classes=["tree_canopy"])
    
    predictor = DefaultPredictor(cfg)
    return predictor

def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to list of points
    polygon = []
    for point in approx:
        x, y = point[0]
        polygon.extend([x, y])
    
    return polygon

def denorm_segmentation(seg, width, height, *, flatten=True, round_int=True):
    """
    seg: list of (x,y) pairs in [0,1]
    returns: flat pixel list [x1,y1,x2,y2,...] (default) or list of pairs
    """
    # If seg is already flat [x1,y1,...], turn into pairs
    if seg and not isinstance(seg[0], (list, tuple)):
        it = iter(seg)
        seg_pairs = list(zip(it, it))
    else:
        seg_pairs = seg

    out = []
    for x, y in seg_pairs:
        X = max(0.0, min(width  - 1, x * width))
        Y = max(0.0, min(height - 1, y * height))
        if round_int:
            X = int(round(X))
            Y = int(round(Y))
        if flatten:
            out.extend([X, Y])
        else:
            out.append((X, Y))
    return out

# Image Metadata
def get_meta(IMGS):
    meta_data = {}
    scene_type_default = "Unknown"

    if IMGS.exists():
        for p in Path(IMGS).iterdir():
            if p.suffix.lower() not in {".tif"}:  # All images .tif
                continue
            file_name = p.name
            cm_resolution = str(file_name[0:2])

            with Image.open(p) as image:  # Pillow / Pill package
                width, height = image.size

            meta_data[file_name] = {
                'file_name': file_name,
                "width": width,
                'height': height,
                "cm_resolution": cm_resolution,
                "scene_type": scene_type_default
            }
    else:
        print('Error - IMGS Not Exist')
    return meta_data

def get_detectron2_annotations(predictor, IMGS):
    """Get annotations using Detectron2 model"""
    annotations = []
    
    if not IMGS.exists():
        print('Error - IMGS Not Exist')
        return annotations
    
    for img_path in sorted(Path(IMGS).iterdir()):
        if img_path.suffix.lower() != ".tif":
            continue
            
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        height, width = image.shape[:2]
        
        # Get predictions
        outputs = predictor(image)
        instances = outputs["instances"]
        
        # Process each detection
        for i in range(len(instances)):
            # Get confidence score
            confidence = instances.scores[i].cpu().numpy()
            
            # Get segmentation mask
            if hasattr(instances, 'pred_masks'):
                mask = instances.pred_masks[i].cpu().numpy()
                
                # Convert mask to polygon
                polygon = mask_to_polygon(mask)
                
                if polygon:
                    # Normalize coordinates
                    normalized_polygon = []
                    for j in range(0, len(polygon), 2):
                        x = polygon[j] / width
                        y = polygon[j + 1] / height
                        normalized_polygon.extend([x, y])
                    
                    # Create annotation
                    annotation = {
                        "image": img_path.name,
                        "class": 0,  # tree_canopy class
                        "confidence_level": float(confidence),
                        "segmentation": normalized_polygon
                    }
                    annotations.append(annotation)
    
    return annotations

def append_imgs_annotations(img_list, annot_list):
    # ensure each image has an annotations list
    for img in img_list:
        if "annotations" not in img or img["annotations"] is None:
            img["annotations"] = []

    for image in img_list:
        w, h = image["width"], image["height"]
        fname = image["file_name"]

        # all annots for this image
        matched = (a for a in annot_list if a["image"] == fname)

        for a in matched:
            seg_px = denorm_segmentation(a["segmentation"], w, h, flatten=True, round_int=True)  # Denormalise Segmentation to Pixels

            image["annotations"].append({
                "class": a["class"],
                "confidence_score": float(a["confidence_level"]),  # or keep as formatted string if you prefer
                "segmentation": seg_px
            })

    return img_list

# class id
ID_to_Name = {
    0: 'individual_tree',
    1: 'group_of_trees'
}

#--Procedure--#
print('Loading Detectron2 Model')
try:
    # Find the actual model path
    model_path = None
    for path in Path('outputs').glob('**/model_final.pth'):
        if 'mask_rcnn' in str(path):
            model_path = str(path)
            break
    
    if model_path is None:
        raise FileNotFoundError("No trained model found in outputs directory")
    
    print(f'Using model: {model_path}')
    predictor = load_detectron2_model(
        model_path=model_path,
        config_file=MODEL_CONFIG['config_file'],
        confidence_threshold=MODEL_CONFIG['confidence_threshold']
    )
    print('Model loaded successfully')
except Exception as e:
    print(f'Error loading model: {e}')
    exit(1)

print('Get Meta Data')
meta = get_meta(IMGS)  # Retrieve Meta Data
meta_2items = list(meta.items())[:1]
meta_2keys = list(meta)[:1]

print(f'Meta Data Items : {meta_2items}')
print(f'Meta Data Keys : {meta_2keys}')

images = []  # Produce Images list

print('Append Meta Data To Images List')
for m in meta.values():  # Append Meta Data to Images List
    images.append({
        'file_name': m['file_name'],
        'width': m['width'],
        'height': m['height'],
        'cm_resolution': m['cm_resolution'],
        'scene_type': m['scene_type'],
        'annotations': []
    })

images_sorted = sorted(images, key=lambda x: x['file_name'])  # Sort Images
print(f'images_sorted')

print('Get Annotations using Detectron2')
annotations_list = get_detectron2_annotations(predictor, IMGS)  # Retrieve Annotations using Detectron2

for annot in annotations_list:  # Clean Annotations
    # .tif suffix -> .tif (already correct)
    # annot['image'] = annot['image'][:-4] + '.tif'

    # map class id -> name
    annot['class'] = ID_to_Name.get(annot['class'], 'Unknown')

    # convert confidence_level -> percentage string
    annot['confidence_level'] = f"{annot['confidence_level']:.2f}"

if annotations_list and len(annotations_list[0]) > 0:
    print('Annotations List Completed')

appended_list = append_imgs_annotations(images_sorted, annotations_list)  # Append Annotations to Images

print(f'Appended_list Outputed')

predict_answer = {}
predict_answer = {'images': appended_list}
print(f'Predict_Answer Formulated')

# Assign Scene Type from Sample Answer to Submission
sample_answer_input = REPO_ROOT / 'exports/sample_answer.json'  # Acquire Sample Answer path

if sample_answer_input.exists():
    with open(sample_answer_input) as f:  # Open Sample Answer Structure
        sample_answer = json.load(f)

    sample_answer = sample_answer['images']

    image_scenes = []  # Create list of images and associated 'scene_type'

    for image in sample_answer:
        file_name = image.get('file_name')  # GET each image's name
        scene_type = image.get('scene_type')  # GET each image's scene_type

        image_scenes.append(  # Append to list
            {
                'filename': file_name,
                'scene_type': scene_type
            }
        )

    pd.json_normalize(image_scenes)  # -- Potential redunant code --#

    image_scenes_df = pd.DataFrame(image_scenes)  # Transform Image_scenes list into DF

    scene_map = image_scenes_df.set_index('filename')['scene_type'].to_dict()  # Restructure image_scenes DF

    submission = predict_answer

    print('Append scene_types to predict_answer')
    # Update each image in submission['images']
    for image in submission['images']:
        filename = image['file_name']
        if image['scene_type'] == "Unknown" and filename in scene_map:
            image['scene_type'] = scene_map[filename]
else:
    print('Sample answer file not found, using default scene types')
    submission = predict_answer

print('Export Submission')
# Rewrite Submission JSON File
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(submission, f, indent=2)

print(f"Export Submission Complete \
      Saved As: 'Detectron2_Submission' \
      Saved At : {OUT_JSON}")

# Print summary statistics
total_images = len(submission['images'])
total_annotations = sum(len(img['annotations']) for img in submission['images'])
avg_annotations = total_annotations / total_images if total_images > 0 else 0

print(f"\nSummary:")
print(f"Total images processed: {total_images}")
print(f"Total annotations: {total_annotations}")
print(f"Average annotations per image: {avg_annotations:.2f}")
print(f"Model confidence threshold: {MODEL_CONFIG['confidence_threshold']}")