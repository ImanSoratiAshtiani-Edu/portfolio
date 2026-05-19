# Loan Approval Classification - Supervised ML Portfolio

## Project Overview

This project predicts whether a loan application is likely to be approved using a supervised machine learning pipeline.

The main goal is not only to maximize accuracy, but also to explore the business trade-off between false approvals and false rejections.

## Dataset

The dataset contains loan applicant information such as income, education, credit history, loan amount, and property area.

Target column:

- Loan_Status
  - Y = Approved
  - N = Rejected

## Machine Learning Workflow

1. Load and inspect the dataset
2. Drop identifier column: Loan_ID
3. Split features and target
4. Use stratified train/test split
5. Build preprocessing pipeline:
   - Numeric features: median imputation + standard scaling
   - Categorical features: most frequent imputation + one-hot encoding
6. Train Logistic Regression model
7. Evaluate using:
   - Accuracy
   - Confusion matrix
   - Precision
   - Recall
   - F1-score
   - ROC AUC
8. Tune threshold using business-cost presets
9. Deploy as a Streamlit portfolio page

## Business Logic

False Positive:

- Actual: Rejected
- Predicted: Approved
- Business meaning: risky loan incorrectly approved

False Negative:

- Actual: Approved
- Predicted: Rejected
- Business meaning: good applicant incorrectly rejected

The app allows users to select different business strategies and observe how the recommended approval threshold changes.

## Streamlit App

Run locally:

`ash
python -m streamlit run app.py
Project Status

First working Streamlit version completed.
