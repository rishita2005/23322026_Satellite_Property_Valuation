# 🏠 Satellite Imagery–Based Property Valuation
**CDC × Yhills | Open Project 2025–26**

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Hybrid-orange.svg)
![TensorFlow](https://img.shields.io/badge/Vision-ResNet50-red.svg)

This project builds an end-to-end **multimodal machine learning system** for predicting house prices by combining traditional tabular housing data with high-resolution satellite imagery.

## 📌 Key Idea
Traditional house price models only learn **what** a house has (bedrooms, sqft). This project teaches the model **where** the house is and **what surrounds it** by extracting environmental context (greenery, urban density, neighborhood structure) from satellite images.



## 🧠 Project Pipeline
1. **Data Acquisition:** Mapbox API fetches satellite tiles based on Lat/Lon.
2. **Feature Extraction:** Pretrained **ResNet** converts images into 512-dimensional visual embeddings.
3. **Feature Fusion:** Tabular data and visual embeddings are concatenated.
4. **Price Prediction:** A Hybrid XGBoost regressor predicts the final property value.

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `data_fetcher.py` | Programmatic image acquisition via Mapbox API. |
| `cdc_feature_extraction_train_test.ipynb` | Extracts 512-D visual embeddings using ResNet. |
| `cdc_tab_data_only_xgboost.ipynb` | Baseline models using only tabular features (XGBoost/RF). |
| `Model_training.ipynb` | Hybrid XGBoost and End-to-End Multimodal CNN+MLP training. |
| `23322026_report.pdf` | Final technical report including EDA and business insights. |
| `23322026_final.csv` | Final predictions for 5,404 properties (strict ID alignment). |

## 📈 Performance Summary
| Model | Input Type | R² Score |
| :--- | :--- | :--- |
| XGBoost | Tabular only | ~0.88 |
| **Hybrid XGBoost** | **Tabular + Images** | **~0.89** |
| CNN + MLP | End-to-End Multimodal | Best qualitative insights |

> **Key Insight:** Satellite imagery contributes significant predictive power, often capturing nuances (like neighborhood "vibe") that numeric data misses.

## 🔍 Key Techniques
- **Vision:** CNN-based embeddings (ResNet), Grad-CAM for explainability.
- **Regression:** Log-transformed price targets, Hybrid XGBoost.
- **Data Science:** PCA & K-Means for market segmentation, Strict ID-wise inference alignment.

## 🚀 How to Run
1. **Download Images:** Run `data_fetcher.py` to acquire satellite tiles.
2. **Extract Visuals:** Run `cdc_feature_extraction_train_test.ipynb` to save image features.
3. **Train Baseline:** (Optional) Run `cdc_tab_data_only_xgboost.ipynb`.
4. **Final Model:** Run `Model_training.ipynb` for the hybrid fusion and predictions.

---
**Submitted By:** Rishita Rathi  
**Enrollment No:** 23322026  
**Major:** Economics (3rd Year)
