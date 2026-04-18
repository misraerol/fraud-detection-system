# Credit Card Fraud Detection System

An end-to-end machine learning system for detecting fraudulent credit card transactions — from raw data to a deployable REST API and interactive dashboard.

---

## Overview

Financial fraud causes billions in losses annually. This project builds a production-ready fraud detection pipeline using real-world transaction data, covering everything from exploratory analysis to model deployment.

**The core challenge:** The dataset is highly imbalanced (~0.17% fraud). The system addresses this using SMOTE and optimizes for Precision/Recall rather than raw accuracy.

---

## Project Structure

```
fraud-detection-project/
│
├── data/               # Raw dataset (not committed)
├── notebooks/          # EDA and modeling experiments
│   ├── eda.ipynb
├── src/                # Core ML logic
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
├── api/                # FastAPI prediction service
│   └── main.py
├── artifacts/          # Saved model and scaler
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn |
| API | FastAPI, Uvicorn |
| UI | Streamlit |
| Serialization | Joblib |

---

## Features

- Exploratory data analysis with visualizations
- Class imbalance handling via SMOTE
- Model training and comparison (Logistic Regression, Random Forest, XGBoost)
- Evaluation with Precision, Recall, F1-score, and ROC-AUC
- REST API for real-time prediction (`POST /predict`)
- Interactive Streamlit dashboard for manual testing

---

## Dataset

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

284,807 transactions · 492 fraud cases · 28 anonymized PCA features

---

## Installation

> _Coming soon — will be added after initial deployment._

---

## Usage

> _Coming soon — will include API examples and Streamlit instructions._

---

## Results

> _Coming soon — model performance metrics will be added after evaluation._

---

## Future Improvements

- Model monitoring and logging
- Advanced feature engineering
- Cloud deployment (AWS / Azure)
- Real-time streaming data support

---

## Author

**Misra Erol** — .NET Backend Developer transitioning into AI Engineering  
[LinkedIn](https://linkedin.com/in/misra-erol) · [GitHub](#)