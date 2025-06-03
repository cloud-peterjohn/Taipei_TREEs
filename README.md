# 🌳 Street Tree Detection and Counting in Taipei City

This project focuses on detecting individual street trees in Taipei City using high-resolution satellite imagery and deep learning.

<p align="center">
    <img src="readme_figures/icon.ico" alt="icon" width="150"/>
</p>

You can view our slides [here](https://github.com/cloud-peterjohn/Taipei_Trees/blob/main/Report.pptx).

---

## 📌 Project Objective

- Automatically locate individual **street trees** from RGB satellite images.
- Automatically count the number of **street trees** from RGB satellite images.
- Focused on **urban settings**, where trees are often occluded or surrounded by buildings.
- Especially designed for **Taipei City**, Taiwan.
---

## 🧠 Methodology

Main Model:
**YOLO 11** <img src="readme_figures/yolo.png" alt="yolo" width="200"/>
<p align="center">
    <img src="readme_figures/yolo.png" alt="yolo" width="200"/>
</p>


Technical Details:
![alt text](readme_figures/challenge.png)
---

## 🗂️ Dataset Sources
Pre-processed datasets are available in [here](https://huggingface.co/datasets/zbyzby/TaipeiTrees/tree/main).

| Source | Description |
|--------|-------------|
| [Taipei City Government Open Data](https://data.gov.tw/) | Tree location in Taipei for finetuning |
| [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static/overview?hl=en) | Satellite RGB imagery in Taipei for finetuning |
| [Forest Damages – Larch Casebearer](https://lila.science/datasets/forest-damages-larch-casebearer/) | Pretraining on tree detection |
| [IDTReeS](https://zenodo.org/records/3934932) | Pretraining on tree detection |

---
## 📦 Result
Example 1:
![result_visual](readme_figures/result_visual1.png)

Example 2:
![result_visual](readme_figures/result_visual2.png)

## 🛠️ Web Application
The web application code is available in [here](https://github.com/cloud-peterjohn/Taipei_Trees/tree/main/green-coverage). 
We also record a demo video of the web application [here](https://github.com/cloud-peterjohn/Taipei_Trees/blob/main/Web_Application.mp4).
You can run it with npm. 
