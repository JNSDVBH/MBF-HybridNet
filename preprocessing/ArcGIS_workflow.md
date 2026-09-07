# ArcGIS Preprocessing Workflow


## Overview

This document describes the spatial preprocessing workflow used to prepare remote sensing, meteorological, soil, and extreme climate data for winter wheat yield estimation.

ArcGIS is mainly used for spatial data processing, coordinate alignment, spatial matching, raster analysis, and feature extraction.

The processed multi-source datasets are subsequently organized into model input samples for MBF-HybridNet training.


---

## 1. Remote Sensing Data Preparation


Remote sensing data are first processed to obtain spatially consistent vegetation and environmental information.

The main preprocessing steps include:


- Remote sensing image preprocessing

- Radiometric and geometric correction

- Study area extraction

- Image clipping based on agricultural regions

- Coordinate system alignment


Vegetation-related spectral features and remote sensing indices are extracted after preprocessing.


---

## 2. Spatial Data Processing and Feature Extraction


ArcGIS is used to process and integrate multi-source spatial datasets for winter wheat yield estimation.


The main procedures include:


1. Spatial matching of remote sensing, meteorological, and soil datasets


2. Raster resampling and spatial resolution consistency processing


3. Extraction of pixel-level or field-level environmental variables


4. Spatial overlay analysis between crop distribution and environmental data


5. Generation of spatial feature samples for model training


The extracted spatial features include remote sensing characteristics, meteorological variables, soil properties, and extreme climate indicators.


---

## 3. Multi-source Feature Preparation


The processed datasets are integrated to construct model input features.


The workflow includes:


- Remote sensing feature extraction

- Meteorological feature integration

- Soil feature association

- Extreme climate factor construction

- Feature normalization and formatting


The final multi-source feature dataset is prepared for MBF-HybridNet training.


---

## 4. Data Export and Model Input Preparation


The processed spatial samples generated from ArcGIS are exported into structured formats.

The exported datasets are further processed using Python scripts for:


- Dataset organization

- Feature loading

- Training and validation split

- Model input construction


The prepared samples are used for training and evaluating the MBF-HybridNet model.


---

## Notes


This ArcGIS workflow describes the spatial preprocessing strategy used in this study.

Due to data usage agreements and institutional restrictions, the original remote sensing data, GIS project files, and dataset-specific preprocessing configurations are not publicly released.