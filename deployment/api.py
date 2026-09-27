from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# Project and model paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"


# --------------------------------------------------
# Load trained Random Forest model
# --------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        f"Unable to load Random Forest model from {MODEL_PATH}: {e}"
    )


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Retail Campaign Response API",
    description=(
        "FastAPI service for predicting customer response "
        "to a retail marketing campaign."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Model features
#
# These must match the features and order used when
# the Random Forest model was trained.
# --------------------------------------------------

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


# --------------------------------------------------
# Input validation schema
# --------------------------------------------------

class CustomerInput(BaseModel):
    """
    Input data required to make a campaign-response
    prediction for one customer.
    """

    n_comp: float = Field(
        ge=0,
        description="Number of complaints"
    )

    loyalty: int = Field(
        ge=0,
        le=1,
        description="Customer loyalty indicator"
    )

    nps: float = Field(
        ge=0,
        le=10,
        description="Net Promoter Score"
    )

    n_communications: float = Field(
        ge=0,
        description="Number of previous communications"
    )

    total_sales: float = Field(
        ge=0,
        description="Customer total sales"
    )

    unique_products: float = Field(
        ge=0,
        description="Number of unique products purchased"
    )

    number_of_invoices: float = Field(
        ge=0,
        description="Number of invoices"
    )

    purchase_days: float = Field(
        ge=0,
        description="Number of days on which purchases were made"
    )

    average_invoice_value: float = Field(
        ge=0,
        description="Average invoice value"
    )

    average_quantity_per_invoice: float = Field(
        ge=0,
        description="Average quantity purchased per invoice"
    )

    average_product_price: float = Field(
        ge=0,
        description="Average product price"
    )

    customer_activity_days: float = Field(
        ge=0,
        description="Number of days of customer activity"
    )


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def home():
    """
    API welcome endpoint.
    """

    return {
        "message": "Retail Campaign Response API is running",
        "model": "Random Forest",
        "docs": "/docs"
    }


# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    """
    Check whether the API and model are available.
    """

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(MODEL_PATH)
    }


# --------------------------------------------------
# Model information endpoint
# --------------------------------------------------

@app.get("/model-info")
def model_info():
    """
    Return information about the loaded model.
    """

    return {
        "model_type": type(model).__name__,
        "number_of_features": len(FEATURES),
        "features": FEATURES,
        "number_of_trees": getattr(model, "n_estimators", None)
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerInput):
    """
    Generate a campaign-response prediction for
    one customer.
    """

    try:

        # Convert validated input into a DataFrame
        customer_data = pd.DataFrame(
            [customer.model_dump()]
        )

        # Ensure the columns are in exactly the same
        # order as the model training data
        customer_data = customer_data[FEATURES]

        # Generate prediction probability
        probability = float(
            model.predict_proba(customer_data)[0][1]
        )

        # Generate binary prediction
        prediction = int(
            model.predict(customer_data)[0]
        )

        # Convert prediction into a readable label
        prediction_label = (
            "Response"
            if prediction == 1
            else "No Response"
        )

        return {
            "predicted_response": prediction,
            "prediction_label": prediction_label,
            "response_probability": round(
                probability,
                4
            ),
            "response_probability_percentage": round(
                probability * 100,
                2
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
