# Customer Churn Prediction

Predicting customer churn for a telecom provider using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Built as a hands-on ML project comparing Logistic Regression, Random Forest, and XGBoost, with a focus on handling class imbalance correctly and avoiding data leakage.

## Problem

Predict which customers are likely to churn (cancel service) based on account and service usage data, so a business could proactively target retention efforts.

## Key Decisions & Why

- **`TotalCharges` cleanup**: 11 rows had blank values, all corresponding to `tenure = 0` (brand-new customers with no billing history). Imputed with 0 — the mathematically correct value, not a guess — rather than dropping or using the median.
- **Class imbalance (~73% No Churn / 27% Churn)**: chosen not to resample (e.g. SMOTE), since the imbalance is moderate, not extreme. Instead, evaluated on precision/recall/F1 rather than accuracy — a model that always predicts "No churn" scores 73% accuracy while catching zero actual churners, which makes accuracy a misleading metric here.
- **No data leakage**: features scaled using `StandardScaler` fit only on the training set, then applied to the test set — never fit on test data, since that would let test-set statistics leak into training and inflate evaluation scores.
- **One-hot encoding with `drop_first=True`**: avoids the dummy variable trap (multicollinearity from redundant dummy columns), which matters for Logistic Regression's coefficient estimates.

## Results

| Model | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|
| **Logistic Regression** | 0.66 | 0.56 | **0.60** |
| Random Forest | 0.63 | 0.50 | 0.56 |
| XGBoost | 0.60 | 0.52 | 0.56 |

Logistic Regression outperformed both tree-based models on every metric for the churn class, despite being the simplest model. This suggests the churn signal in this dataset is largely linear/additive — driven by contract length, tenure, and internet service type — rather than requiring non-linear feature interactions.

**Top churn drivers (from Logistic Regression coefficients):**
- Increases churn risk: Fiber optic internet, higher total charges, electronic check payment
- Decreases churn risk: Two-year contracts, longer tenure, no internet service

All models trained with default hyperparameters — this is a baseline comparison, not a tuned final model.

## Future Improvements

- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV) for Random Forest and XGBoost — current comparison uses default parameters only
- SHAP/LIME for model-agnostic feature explanations, extending beyond Logistic Regression's coefficients to explain the tree-based models too
- Threshold tuning — currently using the default 0.5 cutoff; adjusting this could trade precision for recall depending on business cost of missed churners vs false alarms
- Cross-validation instead of a single train/test split, for more robust performance estimates

## Tech Stack

Python, pandas, scikit-learn, XGBoost

## How to Run

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Open `Churn_prediction.ipynb` in Jupyter and run all cells top to bottom

## Dataset

[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset, 7,043 customers, 21 features)