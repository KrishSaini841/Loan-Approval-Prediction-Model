# Loan-Approval-Prediction-Model

## Overview
This project focuses on predicting whether a loan applicant is likely to be approved or rejected based on their financial and personal background. Using a dataset of 1,000 applicants, I built and evaluated several machine learning classification models to assist in risk assessment.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
* **Models Tested:** Logistic Regression, KNN (K-Nearest Neighbors), Naive Bayes (GaussianNB)

## Key Steps Taken
1. **Data Preprocessing:** Handled missing values using `SimpleImputer` (mean strategy for numerical, most frequent for categorical).
2. **Feature Engineering:** Performed One-Hot Encoding on categorical features and Label Encoding on the target variable.
3. **Exploratory Data Analysis (EDA):** Visualized feature distributions and correlation matrices to identify key drivers of loan approval.
4. **Modeling:** Trained three models and evaluated them using Precision, Recall, F1-score, and Confusion Matrices.
5. **Feature Transformation:** Experimented with log-transformations and polynomial features to enhance predictive performance.

## Results
The **Naive Bayes** model demonstrated the best precision for this dataset. Detailed performance metrics are available in the `loan_approval.ipynb` notebook.
