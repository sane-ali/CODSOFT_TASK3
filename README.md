# Task 3: Customer Churn Prediction 📉

This repository contains the solution for **Task 3: Customer Churn Prediction** completed as part of the Machine Learning Virtual Internship at **CodSoft**.

---

## 📌 Project Overview
The goal of this project is to build a Machine Learning model to predict customer churn for a subscription-based service or business using historical customer usage, demographic data, and account details.

Algorithms explored and evaluated:
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

---

## ⚙️ Workflow & Steps
1. **Data Preprocessing & Encoding:** Handling missing values, dropping non-predictive IDs, and encoding categorical variables using `pd.get_dummies()` or Label Encoding.
2. **Feature Scaling & Splitting:** Scaling numeric features (e.g., tenure, monthly charges) and performing stratified train-test splitting (`test_size=0.2`, `random_state=42`).
3. **Model Training & Comparison:** Training classifiers to distinguish between churned and retained customers.
4. **Evaluation Metrics:** Tracking **Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC Score**, and **Confusion Matrix**.
5. **Visualizations Generated:**
   - Churn distribution plot (Retained vs. Churned customers).
   - Model comparison chart (Accuracy, F1-Score, ROC-AUC).
   - Confusion matrix heatmaps for evaluation.
   - Feature Importance chart highlighting top drivers of customer churn.
6. **Custom Churn Prediction:** Helper function to predict churn probability for new customer profiles.

---

## 💡 Key Takeaways & Conclusion
1. **Top Churn Drivers:** Features like contract type, tenure, monthly charges, and tech support usage strongly influence customer retention.
2. **Model Performance:** Boosting and Ensemble algorithms (e.g., Gradient Boosting, Random Forest) outperform linear models by effectively modeling non-linear interactions between usage behavior and customer demographics.
3. **Business Value:** Identifying high-risk customers allows businesses to take proactive retention measures (e.g., tailored offers, improved service support) before customers cancel their subscriptions.

---

## 🚀 How to Run (Google Colab)
1. Open the Task 3 `.ipynb` notebook in Google Colab.
2. Upload the churn dataset (e.g., `churn.csv` / `Telco-Customer-Churn.csv`).
3. Run all cells sequentially to view evaluation reports, charts, and the conclusion.

---
*Created for CodSoft Machine Learning Internship (#codsoft)*
