# 🌳 Tree Canopy Segmentor — Model Experiments & Findings

This repository documents our full journey of experimenting with multiple segmentation and detection architectures for **individual-tree canopy segmentation** in the **Solafune Canopy Segmentation Challenge**.

We tested a wide range of modern models — from classical UNet to state-of-the-art YOLOv11 & YOLOv12 — and benchmarked everything on Comet ML.  
The task proved extremely challenging due to **tiny object sizes**, **dense canopy overlap**, and **annotation noise**.

---

## 🔗 Competition Link

Solafune Tree Canopy Segmentation Challenge:  
🔗 https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?menu=about&tab=#overview

---

## 🚀 Models We Experimented With

### 1. **UNet (Not Useful)**
- First experiment for segmentation.
- Simple encoder–decoder CNN.
- Not useful as it doesn't perform **instance segmentation**

---

### 2. **Detectron2**
- Problems:
  - Very Slow Convergence.
  - Slow training even on A100.
  - Struggled with overlapping crowns.
- Conclusion → **Not suitable without extremely high-resolution imagery**.

---

### 3. **Mask2Former (Swin Backbone)**
- Attempted high-end transformer-based segmentation.
- **Swin-L / Swin-B** backbones tested.
- Pros:
  - Easy to Train
- Cons:
  - Still Low Accuracy.
  - Dataset too small for transformer models to generalize well.

---

### 4. **YOLOv8 (Segmentation) + Comet ML Analysis**
- Trained multiple YOLOv8-S/M/L segmentation models.
- Full Comet ML evaluation showed:
  - Stable training.
  - Good recall for medium objects.
  - But segmentation masks were coarse and often merged multiple trees.
- Result → Not suitable for fine-grained crown separation.

---

### 5. **YOLOv11 (Best Architecture)**
- **The BEST performing model overall.**
- Best variant: **YOLOv11-M @ 960px resolution**
- Strengths:
  - Strong small-object detection.
  - Better architecture than YOLOv8 for this dataset.
  - High recall and precise masks.
- Final choice for predictions.

---

### 6. **RF-DETR**
- Tested to explore DETR-style approaches.
- TRIANING FAILED / ACCURACY WAS TOO LOW:
  - Unable to generally predict > 100 images at a time. And if we want to increase threshold it was extremely computationally expensive.
  - Our Images have too many annotations per image, which even on A100 the model cannot fit in. We contacted them officially on github issue but it won't work for us.
- Conclusion → **Not suitable**.

---

### 7. **YOLOv12**
- Newer, improved YOLO series.
- Performance was good but still slightly lower than YOLOv11-M after tuning.
- Conclusion → Useful, but YOLOv11-M remained the best.

---

### 8. **SAM2 / SAM3**
- Expected strong segmentation performance.
- Actual results:
  - Completely failed to separate small crowns.
  - Produced merged masks across clusters.
  - Not designed for **micro-instance segmentation**.
- Conclusion → Great foundation model, but **not suitable for small trees**.

---

## 🧪 Our Training Journey

### 🔁 Stage 1 — Augmentation Experiments
We tried:
- flips  
- rotations  
- elastic deformation  
- brightness/contrast  
- mosaic  
- mixup  
- cutouts  

But losses were not decreasing.

---

### 🐞 Stage 2 — We Found the Bug
A misalignment bug in the augmentation pipeline (mask/image mismatch) caused the model to learn incorrectly.

After fixing, we retrained everything.

---

### 📉 Stage 3 — But Accuracy Still Dropped
Even with the **same configuration** as our earlier best experiment, the accuracy did not improve.

### 🧠 Final Hypothesis
The dataset contains:
- **missed labels**
- **wrong detections**
- **merged crowns**
- **labeling inconsistencies**

When augmentations increased data diversity, **these annotation errors amplified,** hurting learning.

---

### Stage 4: PURGE THE BAD DATA IN THE ORIGINAL IMAGES
We found alot of original images had missing and poor labels. We manually searched through 150 of those files and purged the irrelevant ones.

## 🎯 Final Conclusion

### **YOLOv11-M (960px) was the most effective and reliable model.**

It achieved:
- best precision. 
- best recall.
- best small-crown detection.
- best segmentation separation. 

UNet, Detectron2, Mask2Former, SAM2/SAM3, RF-DETR all fell short primarily due to the **tiny object size problem** and **dataset limitations**.


