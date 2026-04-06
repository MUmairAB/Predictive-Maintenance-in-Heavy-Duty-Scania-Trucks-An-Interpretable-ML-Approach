# Predictive Maintenance for Component X Failures in Heavy-Duty Scania Trucks

**Author:** Umair Akram

**Course:** DV2638 - Advanced Machine Learning

**Institution:** Blekinge Institute of Technology (BTH)

**Date:** March 2026

---

## 1. Project Overview

This research project builds an interpretable machine learning pipeline to predict **Component X** failures in heavy-duty Scania trucks using the [SCANIA Component X Dataset](https://researchdata.se/en/catalogue/dataset/2024-34/2). The goal is to transition from reactive to preventive maintenance by identifying vehicles at risk of failure before they break down.

Unlike many current approaches that rely on "black box" Deep Learning models, this project focuses on **explainable AI** using classical algorithms (Logistic Regression, Random Forest, XGBoost) combined with extensive temporal feature engineering. Two LSTM baselines are also trained to benchmark the interpretable models. SHAP (SHapley Additive exPlanations) is used to provide transparent reasons for failure predictions.

> **Research Paper:** The full methodology, experiments, and findings of this project are documented in the research paper included in this repository: **[Research Paper.pdf](Research%20Paper.pdf)**
>
> *Abstract — XGBoost achieves an ROC–AUC of 0.734 and a recall of 0.648, outperforming the best LSTM model (ROC–AUC 0.651, recall 0.373). SHAP analysis reveals that cumulative sensor usage and recent temporal trends are the dominant predictors, providing maintenance engineers with actionable, physically interpretable explanations.*

---

## 2. Objectives

- **Data Preprocessing:** Transform raw Scania operational data (cumulative counters) into a usable tabular format.
- **Feature Engineering:** Capture temporal dependencies through statistical aggregates (rolling averages, trends) without recurrent neural networks.
- **Imbalance Management:** Handle the extreme rarity of failure events using SMOTE.
- **Model Training:** Train and fine-tune Logistic Regression, Random Forest, and XGBoost classifiers.
- **Explainability:** Use SHAP to provide transparent, per-prediction explanations for the best-performing model.
- **Deep Learning Baseline:** Train LSTM-based models (PyTorch) to benchmark against the interpretable pipeline.

---

## 3. Dataset

**Source:** [SCANIA Component X Dataset (2025)](https://researchdata.se/en/catalogue/dataset/2024-34/2) — Lindgren et al., *Scientific Data*, 2025. DOI: [10.1038/s41597-025-04802-6](https://doi.org/10.1038/s41597-025-04802-6)

The dataset is a real-world, multivariate time series collected from an anonymized engine component of a fleet of Scania trucks. It includes:


| File                                  | Description                                         | Size     |
| ------------------------------------- | --------------------------------------------------- | -------- |
| `train_operational_readouts.csv`      | Training sensor time series (cumulative counters)   | ~1.14 GB |
| `train_specifications.csv`            | Training truck specifications                       | ~1 MB    |
| `train_tte.csv`                       | Training labels (time-to-event / failure indicator) | ~337 KB  |
| `test_operational_readouts.csv`       | Test sensor time series                             | ~205 MB  |
| `test_specifications.csv`             | Test truck specifications                           | ~226 KB  |
| `test_labels.csv`                     | Test labels                                         | ~38 KB   |
| `validation_operational_readouts.csv` | Validation sensor time series                       | ~206 MB  |
| `validation_specifications.csv`       | Validation truck specifications                     | ~226 KB  |
| `validation_labels.csv`               | Validation labels                                   | ~38 KB   |


> **Note:** The dataset is not included in this repository due to its size (~1.5 GB). Use `Download_Dataset.ipynb` to download and set it up automatically (see [Setup & Usage](#5-setup--usage)).

---

## 4. Project Structure

```
Repository Root/
│
├── README.md                        # This file
├── LICENSE                          # MIT License
├── .gitignore
├── Research Paper.pdf               # Final IEEE-format research paper  ← READ THIS
├── Final Presentation.pptx          # Course presentation slides
├── Research Proposal.pdf            # Initial research proposal
├── Research Proposal Presentation.pdf
│
└── Code/
    ├── Download_Dataset.ipynb       # ⬇ Step 1 — Download & extract the dataset
    ├── Code.ipynb                   # ⚙ Step 2 — ML pipeline (main notebook)
    ├── DL_Code.ipynb                # ⚙ Step 3 — LSTM baselines (optional)
    ├── requirements.txt             # Python dependencies
    ├── README.txt                   # Detailed code-level readme
    │
    ├── SCANIA Dataset/              # Dataset directory (auto-created by Download_Dataset.ipynb)
    │   ├── data/                    # CSV data files (downloaded separately)
    │   └── documentation/           # Official dataset documentation PDFs
    │
    └── Plots/                       # All figures auto-generated by the notebooks
        ├── confusion_matrices.png
        ├── roc_curves.png
        ├── shap_beeswarm.png
        ├── shap_waterfall_failure.png
        ├── shap_waterfall_healthy.png
        ├── dl_learning_curves.png
        ├── dl_roc_curves.png
        └── (and others)
```

---

## 5. Setup & Usage

### Prerequisites

Python 3.12 or higher is required.

### Step 1 — Clone the repository

```bash
git clone https://github.com/MUmairAB/Advanced-ML-Project.git
cd Advanced-ML-Project
```

### Step 2 — Install dependencies

```bash
pip install -r Code/requirements.txt
```

**Required packages:**


| Package          | Version |
| ---------------- | ------- |
| pandas           | 2.3.1   |
| numpy            | 1.26.4  |
| matplotlib       | 3.9.2   |
| seaborn          | 0.13.2  |
| scikit-learn     | 1.5.1   |
| xgboost          | 3.1.2   |
| imbalanced-learn | 0.12.3  |
| shap             | 0.50.0  |
| torch            | 2.10.0  |
| scipy            | 1.15.2  |


### Step 3 — Download the dataset

> **This step is required before running any analysis notebook.**

Open and run all cells in `Code/Download_Dataset.ipynb`. This notebook will:

1. Download the full SCANIA Component X Dataset (~1.5 GB) from the official repository.
2. Extract all CSV files into `Code/SCANIA Dataset/data/`.
3. Verify that all required files are present.

The download may take several minutes depending on your internet connection. If the dataset is already present, the notebook detects this and skips the download automatically.

### Step 4 — Run the ML pipeline

Open and run all cells in `Code/Code.ipynb`. This notebook covers:

- Data loading and exploratory analysis
- Preprocessing and missing value imputation
- Temporal feature engineering
- Class imbalance handling (SMOTE)
- Training and evaluation of Logistic Regression, Random Forest, and XGBoost
- SHAP explainability analysis

All plots are saved automatically to `Code/Plots/`.

### Step 5 — Run the Deep Learning baseline (optional)

Open and run all cells in `Code/DL_Code.ipynb`. This notebook trains two LSTM-based models using PyTorch and compares them against the classical ML pipeline.

> **Note:** `Code.ipynb` must be run before `DL_Code.ipynb`, as both notebooks share the same dataset loading path.

---

## 6. Methodology

The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) lifecycle:

1. **Business Understanding** — Define failure prediction as a binary classification problem; prioritise recall over precision since missing a failure is far more costly than a false alarm.
2. **Data Understanding** — Explore the multivariate time series; identify ~10% class imbalance in training and ~2.8% in test.
3. **Data Preparation** — Forward-fill missing values, convert cumulative counters to delta values, engineer 10 groups of temporal statistics per sensor (global stats, rolling windows, trend slope, spike indicator, skewness), one-hot encode truck specifications → 1,770 features → filter-based selection retains 647.
4. **Modeling** — Train Logistic Regression, Random Forest, XGBoost (classical) and two LSTM variants (deep learning). Apply SMOTE (`sampling_strategy=0.3`) to the training fold only. Select classification thresholds by maximising the F2 score on a held-out validation split.
5. **Evaluation** — Evaluate using Recall (primary), Precision, F1-Score, and ROC-AUC. Accuracy is not reported due to severe class imbalance.
6. **Explainability** — Apply SHAP TreeExplainer to the final XGBoost model for global feature importance and per-prediction waterfall explanations.

---

## 7. Results

> For the full analysis and discussion, refer to **[Research Paper.pdf](Research%20Paper.pdf)**.

### 7.1 ML Model Comparison


| Model               | Recall    | Precision | F1        | ROC-AUC   |
| ------------------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 0.303     | 0.057     | 0.097     | 0.626     |
| Random Forest       | 0.669     | 0.049     | 0.091     | 0.702     |
| **XGBoost**         | **0.648** | **0.058** | **0.106** | **0.734** |


XGBoost achieves the best ROC-AUC (0.734) and F1 (0.106). The large gap between Logistic Regression and the ensemble methods confirms that failure patterns are strongly non-linear.

![Model Comparison](assets/model_comparison.png)
*Figure 1: Recall, Precision, F1, and ROC-AUC for all three ML classifiers on the test set.*

![ROC Curves](assets/roc_curves.png)
*Figure 2: ROC curves for the three ML classifiers. XGBoost (AUC = 0.734) consistently dominates.*

![Confusion Matrices](assets/confusion_matrices.png)
*Figure 3: Confusion matrices for all three models at their F2-optimal thresholds.*

### 7.2 Comparison with Deep Learning (LSTM)


| Model         | Recall | Precision | F1    | ROC-AUC |
| ------------- | ------ | --------- | ----- | ------- |
| LSTM Baseline | 0.373  | 0.054     | 0.094 | 0.651   |
| LSTM Balanced | 0.268  | 0.041     | 0.071 | 0.561   |


Both LSTM models underperform XGBoost on every metric. The key reason is that the feature engineering pipeline gives XGBoost access to the truck's full operational history, whereas the LSTM sees only the last 100 time steps. Both models triggered early stopping before epoch 15, indicating they reached their capacity under the given architecture.

![LSTM Learning Curves](assets/dl_learning_curves.png)
*Figure 4: Training and validation loss curves for both LSTM models. Both trigger early stopping, with validation loss stagnating rather than improving.*

### 7.3 Explainability via SHAP

SHAP TreeExplainer is applied to the final XGBoost model to compute feature contributions across all 5,045 test predictions.


| Feature            | Mean |SHAP| | Interpretation                              |
| ------------------ | ----------- | ------------------------------------------- |
| `167_1_sum_global` | 0.183       | Cumulative lifetime usage, sensor 167 bin 1 |
| `459_19_mean_w5`   | 0.153       | 5-step recent average, sensor 459 bin 19    |
| `Spec_7_Cat1`      | 0.124       | Vehicle specification type 7                |
| `459_19_mean_w10`  | 0.120       | 10-step recent average, sensor 459 bin 19   |
| `Spec_3_Cat1`      | 0.114       | Vehicle specification type 3                |


The dominant predictor (`167_1_sum_global`) measures total accumulated sensor activity over the truck's full lifetime — higher usage means more wear and a higher probability of failure. The presence of both 5-step and 10-step window averages of sensor 459 shows the model looks at short-term and medium-term trends simultaneously.

![SHAP Feature Importance](assets/shap_bar_importance.png)
*Figure 5: Top 20 features by mean absolute SHAP value. Cumulative usage (`167_1_sum_global`) is the single most influential predictor.*

![SHAP Beeswarm Plot](assets/shap_beeswarm.png)
*Figure 6: SHAP beeswarm plot. Red = high feature value, blue = low. Trucks with high cumulative usage (red dots, top row) are consistently pushed toward a failure prediction.*

---

## 8. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.