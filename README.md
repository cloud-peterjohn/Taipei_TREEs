# 🌳 Street Tree Detection and Species Classification in Taipei City
<img src="icon.png" alt="icon" width="120"/>

This project focuses on detecting individual street trees and identifying their species in Taipei City using high-resolution satellite imagery and deep learning.

---

## 📌 Project Objective

- Automatically locate individual **street trees** from RGB satellite images.
- **Classify the tree species** of each detected tree.
- Focused on **urban settings**, where trees are often occluded or surrounded by buildings.

---

## 🧠 Methodology

The system consists of three main components:

1. **Tree Localization**  
   - Model: `HR-SFANet`  
   - Input: RGB satellite images  
   - Output: Tree center coordinates

2. **Background Removal**  
   - Model: `U²-Net`  
   - Purpose: Remove urban background, isolate tree crown image for classification

3. **Species Classification**  
   - Model: `ResNet` (or CNN variant)  
   - Input: Cropped tree crown image  
   - Output: Predicted species

---

## 🗂️ Dataset Sources
Pre-processed datasets are available in [here](https://huggingface.co/datasets/zbyzby/TaipeiTrees/tree/main).

| Source | Description |
|--------|-------------|
| [Taipei City Government Open Data](https://data.gov.tw/) | Tree location + species |
| [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static/overview?hl=en) | Satellite RGB imagery |
| [Forest Damages – Larch Casebearer](https://lila.science/datasets/forest-damages-larch-casebearer/) | Pretraining on tree detection |
| [IDTReeS](https://zenodo.org/records/3934932) | Pretraining on tree classification & detection |
| [PureForest](https://huggingface.co/datasets/IGNF/PureForest) | Pretraining on tree classification |
---
