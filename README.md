
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![GIS](https://img.shields.io/badge/GIS-Rasterio%20%7C%20GeoPandas-green)
![Status](https://img.shields.io/badge/Project-Active-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# 🏔️ Landslide Susceptibility Mapping System

An end-to-end **machine learning + GIS pipeline** for generating **landslide susceptibility maps** using environmental, topographic, and geological factors.

The project integrates:
- Spatially aware machine learning
- Model stacking and evaluation
- Raster-based GIS prediction pipelines
- Quantitative and classified susceptibility outputs

This repository reflects the **current working state** of the project.  
Large datasets and rasters are intentionally excluded and must be obtained separately.

---

## 🎯 Objective

To predict **landslide susceptibility** across a study area by learning relationships between historical landslide occurrences and conditioning factors such as:

- Topography (slope, curvature, TWI, aspect)
- Terrain (DEM)
- Land cover & NDVI
- Distance-based factors (roads, rivers, faults)
- Geological units

The final output is a set of **GIS-ready susceptibility maps** suitable for:
- Risk zoning
- Planning and mitigation studies
- Spatial decision support

---

## 🗂️ Project Structure

```text
textlandslide_project/
│
├── data/
│   ├── raw/                # Training CSV + original raster zip (NOT tracked)
│   └── processed/          # Intermediate processed datasets (local)
│
├── maps/
│   ├── rasters/            # Input factor rasters (local)
│   ├── probability/        # Model probability rasters
│   ├── quantile/           # Quantile-classified maps
│   ├── jenks/              # Jenks natural breaks maps
│   └── boundary/           # Study area boundary shapefiles
│ 
├── results/
│   ├── metrics/            # Model performance metrics
│   ├── plots/              # ROC curves, confusion matrices
│   └── summaries/          # CV ranking & statistical summaries
│
├── src/
│   ├── landslide_pipeline_core.py         # ML training, spatial CV, stacking
│   ├── landslide_pipeline_gis.py          # Raster prediction & classification
│   └── landslide_pipeline_cartography.py # Map styling & cartographic outputs
│
├── README.md
└── requirements.txt
```

> ⚠️ **Note:**  
> Large raster files and raw datasets are excluded via `.gitignore` to keep the repository lightweight.

---

## 📊 Data Description

### Training Data
- Point-based dataset containing:
  - Landslide occurrence labels
  - Environmental & terrain attributes
  - Spatial coordinates (X, Y)

### Raster Data
- DEM
- Slope
- Aspect
- TWI
- NDVI
- Plan & profile curvature
- Landcover
- Geology
- Distance to roads, rivers, faults

---

## 🔗 Data Access

Due to size constraints, datasets are **not stored in this repository**.

You can obtain the data from:

- **Landslide inventory & conditioning factors:**  
  https://drive.google.com/drive/folders/1sj6XODE06IdPlAIsv6tW-eMN3cOd9rAd?usp=sharing

After downloading:
1. Place CSV files in `data/raw/`
2. Extract raster files into `maps/rasters/`
3. Ensure filenames match those referenced in the pipeline scripts

---

## 🧠 Machine Learning Pipeline

Implemented in `landslide_pipeline_core.py`:

- Spatial block-based train/test split
- Spatial cross-validation (GroupKFold)
- Multiple base models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - SVM
  - XGBoost / LightGBM / CatBoost (if available)
- Probability calibration
- Model ranking using mean spatial CV AUC
- Stacked ensemble using top-performing models
- Evaluation:
  - ROC curves
  - Confusion matrices
  - Bootstrap AUC significance testing

---

## 🗺️ GIS Prediction Pipeline

Implemented in `landslide_pipeline_gis.py`:

- Raster alignment using a reference grid
- Tile-based prediction for memory efficiency
- Probability raster generation for:
  - Top-performing individual models
  - Stacked ensemble
- Susceptibility classification:
  - Quantile-based classes
  - Jenks Natural Breaks
- Outputs saved as GIS-ready GeoTIFFs

---
## 🎨 Cartography & Map Visualization

Implemented in `landslide_pipeline_cartography.py`.

This module focuses on **map presentation and visualization**, separate from
modeling and raster generation.

Key responsibilities:
- Applying cartographic color ramps
- Generating publication-quality map figures
- Overlaying boundaries and contextual layers
- Producing interpretable susceptibility maps for visual analysis

This separation ensures:
- Clean ML logic
- Reusable GIS outputs
- Flexible visualization workflows


## 🧪 Outputs

Generated locally after running the pipeline:

- Continuous susceptibility probability maps
- Classified susceptibility maps (5 classes)
- Model performance summaries
- Comparison between individual models and stacked ensemble

These outputs can be visualized in:
- QGIS
- ArcGIS
- Any GIS software supporting GeoTIFF

---

## ▶️ How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ⚠️ GIS dependencies (rasterio, geopandas) may require system-level libraries on some platforms. Refer to their official installation guides if needed.

---

## 👨‍💻 Author

Abhinava Bhat P V

* GitHub: https://github.com/abhiiibhattt
* Portfolio: https://abhiiibhattt.github.io/portfolio/

---

## 📌 License

This project is intended for educational, research, and portfolio purposes.

© 2026 Abhinava Bhat P V. All rights reserved.

