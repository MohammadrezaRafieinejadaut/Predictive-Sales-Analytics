# Sales-Prediction
End-to-end sales prediction pipeline using tabular data preprocessing, feature engineering, and hyperparameter-tuned gradient boosting models (CatBoost, LightGBM, XGBoost).

# Retail Sales Forecasting & Predictive Modeling Pipeline

An end-to-end machine learning project aimed at predicting product-level sales across diverse retail outlets. The pipeline encompasses structured exploratory data analysis (EDA), domain-specific feature engineering, robust data leakage prevention, and comparative evaluation of multiple regression and gradient boosting models.

---

## 📌 Project Overview
- **Objective:** Predict `Item_Outlet_Sales` based on product characteristics and store attributes to optimize inventory allocation and revenue forecasting.
- **Dataset:** Big Mart Sales Dataset (Tabular, ~8.5k records, 12 features).
- **Core Stack:** Python, Pandas, NumPy, Scikit-Learn, LightGBM, XGBoost, CatBoost, Matplotlib/Seaborn.

---

## ⚙️ Key Pipeline Steps

1. **Exploratory Data Analysis (EDA):**
   - Investigated feature distributions, cardinalities, and missingness patterns.
   - Handled missing values using context-aware strategies (e.g., imputing item weights via product-level identifiers and outlet size via outlet type mode).

2. **Feature Engineering & Preprocessing:**
   - Extracted high-level product categories from item identifiers (Food, Drinks, Non-Consumables).
   - Engineered temporal features such as store operational age (`Outlet_Years`).
   - Standardized categorical inconsistencies and applied categorical encoding.

3. **Model Development & Benchmarking:**
   - Evaluated 9 baseline and ensemble algorithms using 10-Fold Cross-Validation:
     - Linear Models: OLS, Ridge, Lasso
     - Tree-Based Ensembles: Decision Trees, Random Forest, Extra Trees
     - Gradient Boosting: XGBoost, LightGBM, CatBoost
   - Hyperparameter tuning using `RandomizedSearchCV` for optimal convergence and regularization.

4. **Model Serialization:**
   - Serialized the best-performing model pipeline (`Model.pkl`) for inference and deployment readiness.

---

## 📊 Key Highlights & Takeaways
- Demonstrated end-to-end machine learning workflow on tabular retail data.
- Addressed real-world tabular data challenges: missing data imputation, category consolidation, and feature interaction.
- Conducted hyperparameter optimization across modern gradient boosting libraries.
