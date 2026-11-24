#  Task 3: Customer Churn Prediction

An end-to-end machine learning workflow to predict bank customer churn using the `Churn_Modelling.csv` dataset. The project covers data cleaning, class imbalance awareness, exploratory data analysis, leakage-safe preprocessing, cross-validated model comparison, hyperparameter/threshold tuning, model serialization, and inference on unseen customer data.

---

## Dataset Overview

- **Source File:** `Churn_Modelling.csv`[cite: 1]
- **Dimensions:** `10,000` rows × `14` columns[cite: 1]
- **Target Variable:** `Exited` (`0` = Stayed, `1` = Churned)[cite: 1]
- **Class Distribution:**
  - `0` (Stayed): `7,963` (`79.63%`)[cite: 1]
  - `1` (Churned): `2,037` (`20.37%`)[cite: 1]
- **Data Quality:** Verified zero missing values (`NaN`) and zero duplicate rows[cite: 1].
- **Dropped Identifier Columns:** `RowNumber`, `CustomerId`, and `Surname` were removed because they act as unique identifiers/labels rather than predictive behavioral or demographic signals[cite: 1].

### Features Used (10 Predictors)
- **Numerical:** `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`[cite: 1]
- **Binary / Indicator:** `HasCrCard`, `IsActiveMember`[cite: 1]
- **Categorical:** `Geography`, `Gender`[cite: 1]

---

## Methodology & Pipeline

1. **Exploratory Data Analysis (EDA):**
   - Inspected target imbalance (~20.4% churn rate) and chose **ROC-AUC**, **F1-score**, and **Recall** alongside **Accuracy** so minority-class (churned) detection is properly evaluated[cite: 1].
   - Analyzed churn rates across segments (`Geography`, `Gender`, `IsActiveMember`, `NumOfProducts`) and feature distributions (`Age`, `Balance`, `CreditScore`, `EstimatedSalary`)[cite: 1].

2. **Train/Test Split & Preprocessing:**
   - Split data using an `80/20` stratified train-test split (`stratify=y`) to preserve the churn ratio across splits[cite: 1].
   - Built a `ColumnTransformer` inside scikit-learn `Pipeline` objects to prevent data leakage:
     - Scaled continuous/numeric features with `StandardScaler`[cite: 1].
     - Encoded categorical features (`Geography`, `Gender`) with `OneHotEncoder`[cite: 1].

3. **Model Benchmarking & Cross-Validation:**
   - Evaluated multiple classification models using `StratifiedKFold` cross-validation:
     - `LogisticRegression`[cite: 1]
     - `RandomForestClassifier`[cite: 1]
     - `GradientBoostingClassifier`[cite: 1]

4. **Tuning, Interpretability & Inference:**
   - Tuned hyperparameters on the strongest model family and evaluated decision thresholds based on business precision/recall trade-offs[cite: 1].
   - Inspected feature importance / model coefficients to identify key churn drivers[cite: 1].
   - Saved the final pipeline with `joblib` and verified single/batch inference on new customer records[cite: 1].

---

## Project Structure

```text
├── Churn_Modelling.csv          # Input dataset[cite: 1]
├── churn_prediction.ipynb       # EDA, modeling, tuning, and evaluation notebook[cite: 1]
├── churn_model_pipeline.joblib  # Saved trained preprocessing + model pipeline (optional)[cite: 1]
└── README.md                    # Project documentation
