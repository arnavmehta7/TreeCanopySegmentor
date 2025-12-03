# **Tree Canopy Instance Segmentation — Final Report**  
**Team SupTumber**  
Arnav Mehta, Ananya Singla, Mohil Ahuja, Prasham Sheth, Vaibhav Dabas  

## Video:

[![Video Explaining](https://img.youtube.com/vi/0vSGzhuIN-w/0.jpg)](https://www.youtube.com/watch?v=0vSGzhuIN-w)


**Competition:** Solafune Tree Canopy Segmentation Challenge  
🔗 https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?menu=about&tab=#overview

---

## **1. Understanding of the Problem**

Tree canopy instance segmentation is essential for utility safety, vegetation management, and environmental monitoring. Accurate segmentation supports:

- Correct clearance calculations along powerlines and utility corridors  
- Efficient Rights-of-Way (RoW) vegetation maintenance  
- Downstream tasks like tree species classification and canopy health monitoring  

The challenge becomes complex due to remote sensing characteristics:

### **Key Challenges**
- **Overlapping tree crowns**, making instance boundaries ambiguous  
- **Variable resolutions (10–80 cm)** that dramatically change visible detail  
- **Extremely small tree objects**, especially individual trees  
- **Generalization issues** across agricultural, rural, and urban landscapes  
- **Need for instance segmentation**, not just vegetation segmentation  

The competition uses a **weighted mAP metric** at **IoU = 0.75**, where weights depend on:

- **Scene type:** agriculture, urban, rural, industrial, open field  
- **Resolution:** 10–80 cm, with lower resolution images weighted more  

Thus, the objective is not only high segmentation quality but also optimized performance on high-weight images.

---

## **2. Understanding of Solutions Proposed by Others**

Prior research and community solutions span multiple model families:

### **UNet-based models**
- Great for semantic segmentation  
- Fail at instance separation — cannot distinguish tightly packed trees  

### **Mask R-CNN / Detectron2**
- Strong instance segmentation  
- Slow convergence  
- Struggle on tiny, dense objects  
- Require large datasets

### **Transformer-based models (Mask2Former / Swin / DETR / RF-DETR)**
- Excellent global reasoning  
- Very slow convergence on small datasets (~150 images)  
- Poor recall on tiny objects  
- RF-DETR additionally restricted to ~100 detections per image  

### **Foundation models (SAM / SAM2)**
- Great mask quality  
- No capability to classify "Individual Tree" vs "Group of Trees"  
- Over-segmentation issues in dense areas  

### **YOLO family (v8 → v12)**
- Fastest convergence  
- Most stable on small datasets  
- Strong small-object performance  
- Flexible augmentation pipeline  
- Excellent for instance segmentation tasks  

### **Research papers reviewed**
- DeepTree + SAM (zero-shot tree detection)  
- Masked Attention Transformers (Meta / FAIR)  
- IEEE 2022 Urban Forestry Tree Detection  

Overall insights:

- Dense, tiny tree crowns are extremely challenging  
- Transformers require significantly more data  
- YOLO architectures are best suited for small dataset generalization  

---

## **3. Novelty and Understanding of Our Proposed Methods**

Across 100+ experiments, we developed multiple novel techniques that boosted our score from **0.33 → 0.42**, placing us **#14 out of 143+ teams**, within **0.02** of the top 10.

### **3.1 Resolution-Aware Augmentation Strategy**

We discovered heavy augmentation affects different resolutions differently:

- **10–20 cm images:**  
  - Applied *all augmentations including zoom-out*  
  - Zoom-out helps simulate wider aerial context  

- **30–50 cm images:**  
  - Applied *all augmentations except zoom-out*  
  - Zoom-out harmed already low-detail images  

This dynamic augmentation stabilized training and improved instance mask sharpness.

---

### **3.2 Weighted Sampling Based on Evaluation Weights**

Since the evaluator weights samples by:

- **Scene type** (agriculture: 2.0, urban: 1.5)  
- **Resolution** (80 cm: 3.0, 60 cm: 2.5)

We oversampled higher-weight images so the model optimized the final metric, not just raw accuracy.

This produced a noticeable improvement in weighted mAP.

---

### **3.3 Custom IoU-Based Post-Processing (Fixing Evaluator Issues)**

We discovered the competition evaluator did **not** properly enforce IoU = 0.75 when merging predictions.

We implemented our own post-processing:

- Custom NMS  
- Merging boxes at IoU ≥ 0.75  
- Removing redundant predictions  

This increased leaderboard score from **0.37 → 0.40** without changing the model checkpoint.

---

### **3.4 Tuning Hidden YOLO Mask Parameters**

Two rarely-tuned YOLO parameters drastically improved segmentation:

- `overlap_mask`  
- `mask_ratio` (optimal around 2–4)  

Increasing `mask_ratio` refined mask resolution and helped separate tree clusters.

This boosted accuracy from **0.40 → 0.42**.

---

### **3.5 Manual Dataset Purification**

Since the dataset contained only ~150 images, data errors had major impact.

We manually inspected all images and removed those with:

- Missing annotations  
- Incorrect or partial polygons  
- Misaligned masks  

This step significantly increased training stability and reduced loss spikes.

---

## **4. Implementation of Proposed Methods**

### **4.1 Models Evaluated**

We tested numerous models:

- YOLOv8  
- YOLOv9  
- YOLOv11 (final choice)  
- YOLOv12 (preview)  
- Mask R-CNN  
- Detectron2  
- Mask2Former  
- SAM2  
- RF-DETR  
- UNet  

**YOLOv11-M** achieved the best balance between speed, accuracy, and small-dataset robustness.

---

### **4.2 Augmentation Pipeline**

We used the following augmentations:

- Horizontal flip  
- Vertical flip  
- Rotation (±15°)  
- Zoom-in  
- Zoom-out (conditional)  
- Saturation, hue, contrast adjustments  
- Gaussian noise  

We validated all augmentations visually after identifying early pipeline errors.

---

### **4.3 Fixing Augmentation Breakages**

Initial augmentations caused:

- Mask dropouts  
- Torn polygons  
- Shifts between images and label masks  

We rewrote the augmentation logic, added constraints, and validated the augmented dataset manually.

---

### **4.4 Experiment Tracking (CometML)**

We tracked more than 100 experiments:

- Training loss curves  
- Validation curves  
- Augmentation variation tests  
- Hyperparameter sweeps  
- Model comparison runs  
- Resolution-stratified performance  

This allowed scientific, reproducible analysis of all modifications.

---

### **4.5 Post-Processing Enhancements**

Final inference pipeline included:

- Custom NMS  
- Box merging based on IoU  
- Mask refinement using tuned `mask_ratio`  
- Removal of tiny noisy masks  

This substantially refined final predictions.

---

## **5. Lessons Learnt from Experimentation**

### **5.1 Data quality > model architecture**
Cleaning 150 images improved accuracy more than switching between any two models.

### **5.2 Always visualize augmentation output**
Many augmentation bugs were invisible until visualized.

### **5.3 Transformers are not suited for small datasets**
Mask2Former / DETR models consistently underperformed.

### **5.4 YOLO generalizes best to dense small objects**
Fast convergence + strong small-object recall made it ideal.

### **5.5 Understanding the evaluation metric is crucial**
Fixing IoU behavior increased score without touching the model itself.

### **5.6 Hyperparameters can matter more than architecture**
Tweaking `mask_ratio` and `overlap_mask` was more impactful than using entirely different models.

---

## **6. Conclusion**

Through methodical experimentation, innovative augmentation strategies, metric-aware sampling, custom post-processing, and careful data verification, we built a competitive and robust tree canopy instance segmentation pipeline.

Our final achievements:

- **Weighted mAP: 0.42**  
- **Leaderboard Rank: #14 of 143+ teams**  
- **Gap to Top 10: ~0.02**  

Our work demonstrates:

- Deep problem understanding  
- Strong grasp of prior methods  
- Novel thinking in pipeline design  
- Rigorous implementation  
- Clear evidence-driven learning across 100+ experiments  

This project successfully addresses the complexities of real-world airborne tree canopy instance segmentation and lays groundwork for further expansion and ensembling in future iterations.

