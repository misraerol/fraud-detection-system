# Credit Card Fraud Detection System

An end-to-end machine learning system for detecting fraudulent credit card transactions — from raw data exploration to a deployable REST API.

---

## Overview

Financial fraud causes billions in losses annually. This project builds a production-ready fraud detection pipeline using real-world transaction data, covering everything from exploratory analysis to model deployment.

**The core challenge:** The dataset is highly imbalanced (~0.17% fraud — 492 fraud cases out of 284,807 transactions). The system addresses this using SMOTE and optimizes for Precision/Recall balance rather than raw accuracy.

---

## Project Structure

```
fraud-detection-project/
│
├── data/                   # Raw dataset (not committed)
├── notebooks/              # EDA and modeling experiments
│   ├── eda.ipynb
│   └── modeling.ipynb
├── api/                    # FastAPI prediction service
│   └── main.py
├── artifacts/              # Saved model and scalers
│   ├── model.pkl
│   ├── scaler_amount.pkl
│   ├── scaler_time.pkl
│   └── threshold.pkl
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn |
| API | FastAPI, Uvicorn |
| Serialization | Joblib |

---

## Dataset

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 total transactions
- 492 fraud cases (0.17%)
- 28 anonymized PCA features (V1–V28) + `Time` + `Amount`

---

## Exploratory Data Analysis

Key findings from EDA:

- **Severe class imbalance:** 99.83% normal vs 0.17% fraud — addressed with SMOTE during training.
- **Amount distribution:** Normal transactions reach up to ~$25,000; fraud transactions are typically smaller, concentrated under $500.
- **Time feature:** No strong temporal pattern distinguishing fraud from normal transactions.
- **Feature correlations:** V4, V11 show positive correlation with fraud; V12, V14, V17 show strong negative correlation. Most V features are decorrelated by design (PCA).

---

## Modeling

Three models were trained and compared on the imbalanced dataset:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.11 | 0.97 |
| XGBoost | 0.35 | 0.87 | 0.49 | 0.98 |
| **Random Forest** | **0.43** | **0.88** | **0.58** | **0.98** |

**Random Forest** was selected as the final model due to its best F1 score and strong balance between precision and recall.

### Threshold Optimization

The default 0.5 classification threshold was tuned to maximize F1 score. The optimal threshold was found at **0.90**, yielding:

- Precision: ~0.79
- Recall: ~0.79
- F1: ~0.79

This threshold is saved to `artifacts/threshold.pkl` and used at inference time.

---

## API

The prediction service is built with **FastAPI** and exposes three endpoints:

### `GET /`
Returns API metadata and the active threshold.

### `GET /health`
Health check — returns `{"status": "ok"}`.

### `POST /predict`

Accepts a transaction object with all 30 features (`Time`, `V1`–`V28`, `Amount`) and returns a fraud prediction.

**Request body:**
```json
{
  "Time": 406.0,
  "V1": -2.3122,
  "V2": 1.9519,
  ...
  "Amount": 149.62
}
```

**Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0312,
  "threshold": 0.9,
  "result": "NORMAL"
}
```

> **Note:** `Time` and `Amount` are automatically scaled at inference using the saved scalers, matching the preprocessing applied during training.

---

## Installation

```bash
git clone https://github.com/misraerol/fraud-detection-project.git
cd fraud-detection-project
pip install -r requirements.txt
```

---

## Usage

### Run the API

```bash
uvicorn api.main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Run a prediction (curl)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Time": 406, "V1": -2.31, "V2": 1.95, ..., "Amount": 149.62}'
```

---

## Future Improvements

- **Improve precision:** At threshold 0.90, precision is ~0.79 — experimenting with cost-sensitive learning or ensemble stacking could reduce false positives further
- **Feature engineering:** Derive behavioral features (e.g. transaction velocity per card, time-since-last-transaction) to improve signal beyond raw PCA components
- **Streamlit dashboard:** Interactive UI for manual transaction testing and probability visualization
- **Model monitoring & drift detection:** Track prediction distributions over time to catch data drift in production
- **Cloud deployment:** Containerize with Docker and deploy to AWS / Azure / Render
- **Real-time streaming:** Integrate with Kafka for low-latency inference on live transaction streams

---

## Author

**Misra Erol** — .NET Backend Developer transitioning into AI Engineering  
[LinkedIn](https://linkedin.com/in/misra-erol) · [GitHub](https://github.com/misraerol)