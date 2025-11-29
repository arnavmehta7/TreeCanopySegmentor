# Step 1: Install dependencies (if needed)
import os
import zipfile
import requests
import json

# Step 2: Define dataset URLs
urls = {
    "train_annotations": "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/tree-canopy-detection/dataset/train_annotations.json",
    "train_images": "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/tree-canopy-detection/dataset/train_images.zip",
    "evaluation_images": "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/tree-canopy-detection/dataset/evaluation_images.zip",
    "sample_answer": "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/tree-canopy-detection/dataset/sample_answer.json",
}

# Step 3: Create data folder
os.makedirs("data", exist_ok=True)

# Step 4: Download files
def download_file(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading {url} ...")
        r = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    else:
        print(f"{save_path} already exists, skipping.")

download_file(urls["train_annotations"], "data/train_annotations.json")
download_file(urls["train_images"], "data/train_images.zip")
download_file(urls["evaluation_images"], "data/evaluation_images.zip")
download_file(urls["sample_answer"], "data/sample_answer.json")

# Step 5: Unzip image archives
def unzip_file(zip_path, extract_to):
    if os.path.exists(extract_to):
        print(f"{extract_to} already exists, skipping unzip.")
    else:
        print(f"Unzipping {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

unzip_file("data/train_images.zip", "data/train_images")
unzip_file("data/evaluation_images.zip", "data/evaluation_images")

# Step 6: Quick checks
print("Train images:", len(os.listdir("data/train_images")))
print("Eval images:", len(os.listdir("data/evaluation_images")))

# Load annotations as JSON
with open("data/train_annotations.json", "r") as f:
    train_annotations = json.load(f)

print("Number of annotations:", len(train_annotations))
print("Sample keys:", list(train_annotations.keys())[:5])



import os, json, random, shutil
from pathlib import Path
import numpy as np
from pprint import pprint

# Paths (adjust if your files are in a subfolder)
ROOT = "/content/data"
DATA_ROOT = os.path.join(ROOT, "dataset")
os.makedirs(DATA_ROOT, exist_ok=True)

# Files expected in current Colab working dir
TRAIN_ZIP = os.path.join(ROOT, "train_images.zip")
EVAL_ZIP = os.path.join(ROOT, "evaluation_images.zip")
RAW_ANN = os.path.join(ROOT, "train_annotations.json")  # your annotation file
# output locations
IMG_DIR = os.path.join(DATA_ROOT, "images")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "val")

# unzip if needed
import zipfile
if os.path.exists(TRAIN_ZIP):
    print("Unzipping train_images.zip ...")
    with zipfile.ZipFile(TRAIN_ZIP, 'r') as z:
        z.extractall(TRAIN_IMG_DIR)
if os.path.exists(EVAL_ZIP):
    print("Unzipping evaluation_images.zip ...")
    with zipfile.ZipFile(EVAL_ZIP, 'r') as z:
        z.extractall(VAL_IMG_DIR)

# helper: collect all image files
def collect_images(base):
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    files = []
    for root, _, filenames in os.walk(base):
        for f in filenames:
            if f.lower().endswith(exts):
                files.append(os.path.relpath(os.path.join(root, f), start=DATA_ROOT))
    return sorted(files)

print("Train image dir:", TRAIN_IMG_DIR, "exists?", os.path.exists(TRAIN_IMG_DIR))
