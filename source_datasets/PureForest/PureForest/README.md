---
license: etalab-2.0
pretty_name: PureForest
size_categories:
- 100K<n<1M
task_categories:
- image-classification
- other
tags:
- IGN
- Aerial
- Environement
- Multimodal
- Earth Observation
- Lidar
- ALS
- Point Cloud
- Forest
- Tree Species
---

# PureForest: A Large-Scale Aerial Lidar and Aerial Imagery Dataset for Tree Species Classification in Monospecific Forests

> - PureForest dataset is derived from 449 different forests located in 40 French departments, mainly in the southern regions. 
> - This dataset includes 135,569 patches, each measuring 50 m x 50 m, covering a cumulative exploitable area of 339 km². 
> - Each patch represents a monospecific forest, annotated with a single tree species label.
> - The proposed classification has 13 semantic classes, hierarchically grouping 18 tree species. 
> - PureForest features 3D and 2D modalities:
>   - High density Aerial Lidar Scanning (ALS) point clouds of high density: 10 pulses/m², or about 40 pts/m².
  The Lidar data was acquired via the [Lidar HD program (2020-2025)](https://geoservices.ign.fr/lidarhd), an ambitious initiative undertaken by the IGN - the French Mapping Agency - to obtain a detailed 3D description of the French territory using ALS.
>   - Very High Resolution (VHR) aerial images with RGB + Near-Infrared channels at a spatial resolution of 0.2 m (250 × 250 pixels).
  Aerial images come from the [ORTHO HR®](https://geoservices.ign.fr/bdortho), a mosaic of aerial images acquired during national aerial surveys by the IGN. 
  Lidar and imagery data were acquired over several years in distinct programs, and up to 3 years might separate them. The years of acquisition are given as metadata.  

The dataset is associated with a data paper: [PureForest: A Large-Scale Aerial Lidar and Aerial Imagery Dataset for Tree Species Classification in Monospecific Forests](https://arxiv.org/abs/2404.12064)

## Dataset content
<hr style='margin-top:-1em; margin-bottom:0' />
The PureForest dataset consists of a total of 135,569 patches: 69111 in the train set, 13523 in the val set, and 52935 in the test set.
Each patch includes a high-resolution aerial image (250 pixels x 250 pixels) at 0.2 m resolution, and a point cloud of high density aerial Lidar (10 pulses/m², ~40pts/m²).
Band order is near-infrared, red, green, blue. For convenience, the Lidar point clouds are vertically colorized with the aerial images.


VHR Aerial images (Near-Infrared, Red, Green) [ORTHO HR]       |   ALS points clouds [Lidar HD]
:-------------------------:|:-------------------------:
![](./imagery_18_classes.png)  |  ![](./lidar_18_classes.png)

### Annotations
<hr style='margin-top:-1em; margin-bottom:0' />
Annotations were made at the forest level, and considering only monospecific forests. A semi-automatic approach was adopted in which forest polygons
were selected and then curated by expert photointerpreters from the IGN. The annotation polygons were selected from the [BD Forêt](https://inventaire-forestier.ign.fr/spip.php?article646), 
a forest vector database of tree species occupation in France. Ground truths from the [French National Forest Inventory](https://inventaire-forestier.ign.fr/?lang=en) 
were also used to improve the confidence in the purity of the forests.

| Class | Train (%) | Val (%) | Test (%) |
|-------|------------:|----------:|-----------:|
**(0) Deciduous oak**|22.92%|32.35%|52.59%
**(1) Evergreen oak**|16.80%|2.75%|19.61%
**(2) Beech**|10.14%|12.03%|7.62%
**(3) Chestnut**|4.83%|1.09%|0.38%
**(4) Black locust**|2.41%|2.40%|0.60%
**(5) Maritime pine**|6.61%|7.10%|3.85%
**(6) Scotch pine**|16.39%|17.95%|8.51%
**(7) Black pine**|6.30%|6.98%|3.64%
**(8) Aleppo pine**|5.83%|1.72%|0.83%
**(9) Fir**|0.14%|5.32%|0.05%
**(10) Spruce**|3.73%|4.64%|1.64%
**(11) Larch**|3.67%|3.73%|0.48%
**(12) Douglas**|0.23%|1.95%|0.20%

### Dataset extent and train/val/test split
<hr style='margin-top:-1em; margin-bottom:0' />
The annotation polygons were mostly sampled in the southern half of metropolitan France due to the partial availability of the Lidar HD data at the time of dataset creation. 
They are scattered in 40 distinct French administrative departments and span a large diversity of territories and forests within each semantic class.

To define a common benchmark, we split the data into train, val, and test sets (70%-15%-15%) with stratification on semantic labels. 
We address the high spatial autocorrelation inherent to geographic data by splitting at the annotation polygon level: 
each forest exclusively belongs to either the train, val, or test set.

![](./dataset_extent_map.excalidraw.png)

## Citation
<hr style='margin-top:-1em; margin-bottom:0' />
Please include a citation to the following Data Paper if PureForest was useful to your research:

```
@misc{gaydon2024pureforest,
      title={PureForest: A Large-Scale Aerial Lidar and Aerial Imagery Dataset for Tree Species Classification in Monospecific Forests}, 
      author={Charles Gaydon and Floryne Roche},
      year={2024},
      eprint={2404.12064},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2404.12064}
      primaryClass={cs.CV}
}
```


## Dataset license
<hr style='margin-top:-1em; margin-bottom:0' />
The "OPEN LICENCE 2.0/LICENCE OUVERTE" is a license created by the French government specifically for the purpose of facilitating the dissemination of open data by public administration.<br/>
This licence is governed by French law.<br/>
This licence has been designed to be compatible with any free licence that at least requires an acknowledgement of authorship, and specifically with the previous version of this licence as well as with the following licences: United Kingdom’s “Open Government Licence” (OGL), Creative Commons’ “Creative Commons Attribution” (CC-BY) and Open Knowledge Foundation’s “Open Data Commons Attribution” (ODC-BY).