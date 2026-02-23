# Predictive Maintenance for Scania Trucks (Component X)

**Status:** 🚧 Under Development

## Overview

This research project aims to develop a robust and interpretable machine learning model to predict "Component X" failures in heavy-duty Scania trucks. The goal is to transition from reactive to preventive maintenance by identifying vehicles at risk of failure before they break down.

Unlike many current approaches that rely on "black box" Deep Learning models, this project focuses on **explainable AI** using classical algorithms (such as XGBoost and CatBoost) combined with extensive temporal feature engineering.

## Objectives

*   **Data Preprocessing:** Transform raw Scania operational data (cumulative counters, histograms) into a usable tabular format.
*   **Feature Engineering:** Capture temporal dependencies through statistical aggregates (rolling averages, trends) without using recurrent neural networks.
*   **Imbalance Management:** Handle the extreme rarity of failure events using techniques like SMOTE.
*   **Model Optimization:** Train and fine-tune Gradient Boosting models.
*   **Explainability:** Use SHAP (SHapley Additive exPlanations) to provide transparent reasons for failure predictions.

## Dataset

This project utilizes the **SCANIA Component X Dataset**, which includes:
*   Operational sensor data
*   Truck specifications
*   Repair records

## Methodology

The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) lifecycle:
1.  Business Understanding
2.  Data Understanding
3.  Data Preparation
4.  Modeling
5.  Evaluation (Focus on Recall and F1-Score)
