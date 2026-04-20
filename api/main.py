from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

import joblib
import numpy as np 

# Load the model and scalers
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
model = joblib.load(ARTIFACTS_DIR / 'model.pkl')
scaler_amount = joblib.load(ARTIFACTS_DIR / 'scaler_amount.pkl')
scaler_time = joblib.load(ARTIFACTS_DIR / 'scaler_time.pkl')
threshold_config = joblib.load(ARTIFACTS_DIR / 'threshold.pkl')
THRESHOLD = threshold_config['threshold']


app =  FastAPI(
    title= "Fraud Detection API",
    description= "Credit card fraud detection using Random Forest",
    version="1.0.0"
)

#BaseModel
class Transaction(BaseModel):
   Time: float
   V1  : float
   V2  : float
   V3  : float
   V4  : float
   V5  : float
   V6  : float
   V7  : float
   V8  : float
   V9  : float
   V10 : float
   V11 : float
   V12 : float
   V13 : float
   V14 : float
   V15 : float
   V16 : float
   V17 : float
   V18 : float
   V19 : float
   V20 : float
   V21 : float
   V22 : float
   V23 : float
   V24 : float
   V25 : float
   V26 : float
   V27 : float
   V28 : float
   Amount: float



@app.get("/")
def root():
   return{
      "message":"Fraud Detection API",
      "version":"1.0.0",
      "threshold": THRESHOLD
   }

#Is the service standing up?
@app.get("/health")
def health():
   return {"status":"ok"}

@app.post("/predict")
def predict(transaction: Transaction):
   data = np.array([[    
      transaction.Time,
      transaction.V1, transaction.V2, transaction.V3,
      transaction.V4, transaction.V5, transaction.V6,
      transaction.V7, transaction.V8, transaction.V9,
      transaction.V10, transaction.V11, transaction.V12,
      transaction.V13, transaction.V14, transaction.V15,
      transaction.V16, transaction.V17, transaction.V18,
      transaction.V19, transaction.V20, transaction.V21,
      transaction.V22, transaction.V23, transaction.V24,
      transaction.V25, transaction.V26, transaction.V27,
      transaction.V28,
      transaction.Amount
   ]])
   # We did scaling during training → the same should be done here
   data[0][0] =scaler_time.transform([[data[0][0]]])[0][0]
   data[0][29] =scaler_amount.transform([[data[0][29]]])[0][0]

   probabilty = model.predict_proba(data)[0][1]
   is_fraud = bool(probabilty >= THRESHOLD)
   
   return{
      "is_fraud":is_fraud,
      "fraud_probability": round(float(probabilty),4),
      "threshold": THRESHOLD,
      "result": "FRAUD" if is_fraud else "NORMAL"      
   }



