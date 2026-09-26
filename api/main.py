from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"


try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    MODEL_ERROR = str(e)
else:
    MODEL_ERROR = None


FEATURES = [
    "n_comp",
    "loyalty",
    "nps",
    "n_communications",
    "total_sales",
    "unique_products",
    "number_of_invoices",
    "purchase_days",
    "average_invoice_value",
    "average_quantity_per_invoice",
    "average_product_price",
    "customer_activity_days"
]


app = FastAPI(
    title="AI Retail Campaign Response Prediction API",
    description=(
        "API for predicting customer response to a "
        "retail marketing campaign using a trained "
        "Random Forest classification model."
    ),
    version="1.0.0"
)


class CustomerData(BaseModel):
    n_comp: float = Field(
        ...,
        ge=0,
        description="Number of campaign components"
    )

    loyalty: float = Field(
        ...,
        ge=0,
        description="Customer loyalty score"
    )

    nps: float = Field(
        ...,
        ge=-100,
        le=100,
        description="Net Promoter Score"
    )

    n_communications: float = Field(
        ...,
        ge=0,
        description="Number of customer communications"
    )

    total_sales: float = Field(
        ...,
        ge=0,
        description="Total customer sales value"
    )

    unique_products: float = Field(
        ...,
        ge=0,
        description="Number of unique products purchased"
    )

    number_of_invoices: float = Field(
        ...,
        ge=0,
        description="Number of customer invoices"
    )

    purchase_days: float = Field(
        ...,
        ge=0,
        description="Number of days on which purchases were made"
    )

    average_invoice_value: float = Field(
        ...,
        ge=0,
        description="Average value per customer invoice"
    )

    average_quantity_per_invoice: float = Field(
        ...,
        ge=0,
        description="Average quantity purchased per invoice"
    )

    average_product_price: float = Field(
        ...,
        ge=0,
        description="Average product price"
    )

    customer_activity_days: float = Field(
        ...,
        ge=0,
        description="Number of days of customer activity"
    )


@app.get("/")
def root():
    return {
        "message": "AI Retail Campaign Response Prediction API",
        "version": "1.0.0",
        "model": "Random Forest",
        "model_loaded": model is not None,
        "number_of_features": len(FEATURES),
        "endpoints": [
            "/",
            "/health",
            "/model-info",
            "/predict"
        ]
    }


@app.get("/health")
def health_check():
    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": MODEL_ERROR
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(model).__name__,
        "features_expected": model.n_features_in_
    }


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model is not loaded: {MODEL_ERROR}"
        )

    return {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "number_of_features": model.n_features_in_,
        "features": FEATURES,
        "n_estimators": getattr(
            model,
            "n_estimators",
            None
        ),
        "random_state": getattr(
            model,
            "random_state",
            None
        ),
        "classes": [
            int(c) for c in model.classes_
        ]
    }


@app.post("/predict")
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model is not loaded: {MODEL_ERROR}"
        )

    try:
        features = pd.DataFrame(
            [[
                customer.n_comp,
                customer.loyalty,
                customer.nps,
                customer.n_communications,
                customer.total_sales,
                customer.unique_products,
                customer.number_of_invoices,
                customer.purchase_days,
                customer.average_invoice_value,
                customer.average_quantity_per_invoice,
                customer.average_product_price,
                customer.customer_activity_days
            ]],
            columns=FEATURES
        )

        prediction = model.predict(features)[0]
        prediction = int(prediction)

        probabilities = model.predict_proba(features)[0]

        class_probabilities = {
            int(class_label): float(probability)
            for class_label, probability
            in zip(model.classes_, probabilities)
        }

        response_probability = class_probabilities.get(1, 0.0)
        non_response_probability = class_probabilities.get(0, 0.0)

        prediction_label = (
            "Response"
            if prediction == 1
            else "Non-response"
        )

        return {
            "prediction": prediction,
            "prediction_label": prediction_label,
            "response_probability": round(
                response_probability,
                4
            ),
            "non_response_probability": round(
                non_response_probability,
                4
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )