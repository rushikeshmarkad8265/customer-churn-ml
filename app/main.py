from pathlib import Path
import pandas as pd
from fastapi import FastAPI,HTTPException

import joblib
from pydantic import BaseModel, Field

from app.feature_engineering import engineer_features

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"

model = joblib.load(MODEL_PATH)


import numpy as np
import shap

preprocessor = model.named_steps["preprocessor"]
rf_model = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(rf_model)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained ML pipeline.",
    version="1.0.0"
)

class CustomerInput(BaseModel):
    age: int = Field(ge=18, le=100)
    gender: str
    tenure: int = Field(gt=0)
    usage_frequency: int = Field(ge=0)
    support_calls: int = Field(ge=0)
    payment_delay: int = Field(ge=0)
    subscription_type: str
    contract_length: str
    total_spend: float = Field(ge=0)
    last_interaction: int = Field(ge=0)

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    row = pd.DataFrame([{
        "Age": customer.age,
        "Gender": customer.gender,
        "Tenure": customer.tenure,
        "Usage Frequency": customer.usage_frequency,
        "Support Calls": customer.support_calls,
        "Payment Delay": customer.payment_delay,
        "Subscription Type": customer.subscription_type,
        "Contract Length": customer.contract_length,
        "Total Spend": customer.total_spend,
        "Last Interaction": customer.last_interaction,
    }])

    try:
        row = engineer_features(row)
                
        # Transform the customer's features using the fitted preprocessor
        transformed_row = preprocessor.transform(row)

        # Explain this customer's prediction
        shap_values = explainer.shap_values(transformed_row)

        # Your installed SHAP version returns:
        # (samples, features, classes)
        customer_shap = shap_values[0, :, 1]

        # Select the five strongest contributions
        top_indices = np.argsort(np.abs(customer_shap))[::-1][:5]

        top_reasons = []

        for idx in top_indices:
            value = float(customer_shap[idx])

            top_reasons.append({
                "feature": str(feature_names[idx]),
                "shap_value": round(value, 4),
                "direction": (
                    "toward_churn" if value > 0
                    else "away_from_churn" if value < 0
                    else "neutral"
                )
            })

        probability = float(model.predict_proba(row)[0, 1])
        prediction = int(model.predict(row)[0])
        print("Raw probabilities:", model.predict_proba(row))
        print("Predicted class:", model.predict(row))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": (
            "High" if probability >= 0.7
            else "Medium" if probability >= 0.4
            else "Low"
        ),
        "top_reasons": top_reasons,
        "note": "Risk thresholds are illustrative and not yet business-validated."
    }
 