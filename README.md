# Telecom Customer Churn Prediction: Model Comparison

## Problem
Telecom companies lose significant revenue to customer churn. Predicting which customers are likely to leave, and understanding why, lets a business intervene before it happens. This project compares three modern ML approaches for churn prediction and tests whether a model trained on one dataset generalizes to another, a common real-world challenge since models are rarely deployed on the exact data they were trained on.

## Approach
Two experiments across two datasets:

- **Experiment 1 (within-dataset)**: train/test split (80/20) on the IBM Telco dataset
- **Experiment 2 (cross-dataset generalization)**: train on a 1M-row synthetic telecom dataset, test on the IBM Telco dataset, to see how well each model generalizes to unseen, differently-distributed data

**Models compared:**
- CatBoost
- TabNet
- FT-Transformer (implemented from scratch in PyTorch)

**Additional analysis:**
- SHAP explainability for all three models in both experiments
- SHAP consistency analysis (Spearman rank correlation) comparing how similarly each model ranks feature importance
- SMOTE-ENN for class imbalance handling
- Feature harmonisation between the two datasets (17-19 common features) to make cross-dataset training possible

## Tools
Python, pandas, scikit-learn, CatBoost, PyTorch, pytorch-tabnet, imbalanced-learn (SMOTE-ENN), SHAP, matplotlib

## Data
- [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (~7,043 rows)
- [Customer Churn Prediction Dataset: 1M](https://www.kaggle.com/datasets/isandeep06/customer-churn-prediction-dataset-1m) (synthetic). Experiment 2 trains on a 100k-row sample of this dataset

Datasets are not included in this repo (the 1M dataset is too large for GitHub). To run the notebook:
1. Download both datasets from the Kaggle links above
2. Create a `data/` folder in the same directory as the notebook
3. Place the files inside as `data/Telco-Customer-Churn.csv` and `data/customer_churn_1M.csv` (or update the `PATH_1M` / `PATH_IBM` variables in the notebook's second cell if you name them differently)

## Key results

**Experiment 1, within-dataset AUC-ROC:**
| Model | AUC |
|---|---|
| CatBoost | 0.8389 |
| TabNet | 0.8372 |
| FT-Transformer | 0.8261 |

**Experiment 2, cross-dataset generalization AUC-ROC (trained on 1M, tested on IBM Telco):**
| Model | AUC |
|---|---|
| CatBoost | 0.7822 |
| TabNet | 0.7365 |
| FT-Transformer | 0.6664 |

All three models perform closely within a single dataset (AUC within 0.013 of each other), but CatBoost generalizes noticeably better than the other two once trained on one dataset and tested on another. FT-Transformer's cross-dataset AUC (0.6664) is only marginally better than random classification, a meaningful finding for real-world deployment, where train/test distribution shift is the norm rather than the exception.

Precision stayed low (0.33 to 0.48) across all six model-experiment combinations, a direct effect of using SMOTE-ENN to boost recall on the minority churn class. This trade-off favors catching more true churners at the cost of more false positives, which is generally the right call in a retention context, since missing a real churner usually costs more than a false alert.

## How to run
1. Clone this repo
2. Download both datasets from the Kaggle links above and place them in a local `data/` folder as described in the Data section
3. Install dependencies: `pip install pandas numpy scikit-learn catboost pytorch-tabnet torch imbalanced-learn shap matplotlib seaborn`
4. Open `Telcom_Customer_Churn_Prediction.ipynb` and run cells in order
