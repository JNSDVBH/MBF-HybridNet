# MBF-HybridNet 
 
## Develop a Multi-Branch Fusion Deep Learning Model to Estimate the Winter Wheat Yields under Extreme Climate Events
 
## Overview
模型图 
## Abstract
Extreme climate events (ECEs) have a significant impact on crop yields. However, existing process-based models have restricted ability to accurately quantify the influences of ECEs on yields. Traditional approaches mainly rely on regional average data or sampling-based methods, which fail to capture critical information influencing crop yields at different growth stages. Moreover, these methods generally depend on annual data, resulting in delayed predictions. To address these limitations, this study proposes a Multi-Branch Fusion Hybrid Network (MBF-HybridNet) model that integrates extreme climate indices (ECIs) to better quantify the effects of ECEs on crop yield and to improve the predictive performance. First, the Thiessen Polygon method was employed to partition regional data, enabling the extraction of more detailed spatiotemporal dynamic information. Second, based  on the physiological characteristics of winter wheat, the growth cycle was divided into three phases: vegetative growth phase (VGP), vegetative-reproductive growth phase (VRGP), and reproductive growth phase (RGP) to assess the feasibility of predicting yields prior to maturity. Finally, an innovative MBF-HybridNet model was developed, which integrates an asymmetric 2D convolutional neural network (2D-CNN), a long short-term memory (LSTM) network, and a self-attention (SA) module to achieve multidimensional fusion of spatial and temporal features from dynamic data (remote sensing, meteorological, and ECIs data) and static data (soil properties). The model’s performance was evaluated for winter wheat yield prediction on a county basis in Qingdao from 2004 to 2019. At all progressively cumulative growth stages, results showed that the proposed model consistently outperformed the baseline LSTM model. The model incorporating GA-selected ECIs significantly reduced computational costs while maintaining prediction accuracy, with average improvements across the three progressively cumulative growth stages (R²: +1.87%, MAE: -12 kg·ha⁻¹, RMSE: -16.58 kg·ha⁻¹). Furthermore, the model demonstrated the ability to reliably predict up to 30 days before harvest. Sensitivity analysis revealed that pre-flowering low-temperature events, post-flowering high-temperature events, and water stress were the primary ECEs affecting winter wheat yield.

This repository provides the core implementation framework of MBF-HybridNet, a Multi-Branch Fusion Hybrid Network proposed for winter wheat yield estimation. 
 
The proposed framework integrates multi-source information, including remote sensing observations, meteorological variables, soil properties, and extreme climate factors, to improve winter wheat yield estimation under climate variability. 
 
This repository provides the core model implementation, training pipeline, evaluation procedures, configuration files, and essential data processing interfaces for methodological understanding and reproduction. 
 
Due to data usage agreements and institutional restrictions, the original datasets, dataset-specific preprocessing configurations, and trained model weights are not publicly released. 
 
 