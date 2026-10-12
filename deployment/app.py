from pathlib import Path
import base64
import io
import os

import joblib
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"

# Render supplies PORT automatically.
# The fallback supports running the application locally.
PORT = int(os.environ.get("PORT", 10000))
HOST = "0.0.0.0"

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
    "customer_activity_days",
]


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

print("Loading Random Forest model...", flush=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

try:
    model = joblib.load(MODEL_PATH)

    print("Random Forest model loaded successfully.", flush=True)
    print(f"Model path: {MODEL_PATH}", flush=True)
    print(f"Number of trees: {model.n_estimators}", flush=True)
    print(f"Number of features: {model.n_features_in_}", flush=True)

    if hasattr(model, "feature_names_in_"):
        print(
            f"Model features: {list(model.feature_names_in_)}",
            flush=True,
        )

except Exception as exc:
    print(f"Error loading model: {exc}", flush=True)
    raise

if model.n_features_in_ != len(FEATURES):
    raise ValueError(
        f"Model expects {model.n_features_in_} features, "
        f"but the application defines {len(FEATURES)}."
    )

if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError(
            "The model's training feature names or order do not "
            "match the application's FEATURES list. Check the "
            "training pipeline before making predictions."
        )

if not hasattr(model, "predict_proba"):
    raise ValueError(
        "The loaded model does not support predict_proba()."
    )

if 0 not in model.classes_ or 1 not in model.classes_:
    raise ValueError(
        "The model must contain both class 0 (No Response) "
        "and class 1 (Response)."
    )


# ============================================================
# DASH APPLICATION
# ============================================================

app = Dash(__name__)
app.title = "Retail Campaign Response Prediction"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_class_probability(probabilities, target_class):
    """Return a probability using the model's actual class order."""
    class_index = list(model.classes_).index(target_class)
    return probabilities[:, class_index]


def parse_uploaded_csv(contents, filename):
    """Decode and load an uploaded CSV file."""

    if not contents:
        raise ValueError("No file was uploaded.")

    if filename and not filename.lower().endswith(".csv"):
        raise ValueError("Please upload a CSV file.")

    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string, validate=True)

        df = pd.read_csv(
            io.BytesIO(decoded),
            encoding="utf-8-sig",
        )

    except Exception as exc:
        raise ValueError(
            f"Unable to read uploaded CSV file: {exc}"
        ) from exc

    if df.empty:
        raise ValueError("The uploaded CSV contains no customer records.")

    print(f"Uploaded file: {filename}", flush=True)
    print(f"Rows: {len(df)}", flush=True)
    print(f"Columns: {list(df.columns)}", flush=True)

    return df


def prepare_prediction_data(df):
    """
    Validate uploaded data and return the 12 model features
    in the exact training order.
    """

    df = df.copy()

    # Support the alternate column name in the original dataset.
    if (
        "customer_lifetime_days" in df.columns
        and "customer_activity_days" not in df.columns
    ):
        df = df.rename(
            columns={
                "customer_lifetime_days": "customer_activity_days"
            }
        )

        print(
            "Renamed customer_lifetime_days to "
            "customer_activity_days.",
            flush=True,
        )

    missing_columns = [
        column for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Convert model features to numeric values.
    for column in FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Match the project's median-imputation approach for NPS.
    missing_nps = int(df["nps"].isna().sum())

    if missing_nps:
        nps_median = df["nps"].median()

        if pd.isna(nps_median):
            raise ValueError(
                "NPS contains missing or invalid values and "
                "a median cannot be calculated."
            )

        df["nps"] = df["nps"].fillna(nps_median)

        print(
            f"Imputed {missing_nps} missing NPS values "
            f"using the uploaded data median ({nps_median}).",
            flush=True,
        )

    # Do not silently replace other missing or invalid features.
    problematic_columns = [
        column for column in FEATURES
        if df[column].isna().any()
    ]

    if problematic_columns:
        raise ValueError(
            "Missing or non-numeric values were found in: "
            + ", ".join(problematic_columns)
        )

    # Reject infinite values before passing data to the model.
    if not all(
        pd.Series(df[column]).map(
            lambda value: pd.notna(value)
            and abs(value) != float("inf")
        ).all()
        for column in FEATURES
    ):
        raise ValueError(
            "The uploaded data contains infinite or invalid "
            "numeric feature values."
        )

    X = df[FEATURES].copy()

    if X.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: received {X.shape[1]} features; "
            f"model expects {model.n_features_in_}."
        )

    print(f"Prediction rows: {len(X)}", flush=True)
    print(f"Prediction features: {list(X.columns)}", flush=True)

    return X


def create_probability_chart(probability_0, probability_1):
    """Create a chart showing the two class probabilities."""

    fig = go.Figure(
        data=[
            go.Bar(
                x=["No Response", "Response"],
                y=[probability_0, probability_1],
                text=[
                    f"{probability_0:.1%}",
                    f"{probability_1:.1%}",
                ],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Campaign Response Probability",
        yaxis_title="Probability",
        yaxis=dict(
            range=[0, 1],
            tickformat=".0%",
        ),
        xaxis_title="Prediction",
        template="plotly_white",
    )

    return fig


# ============================================================
# DASHBOARD LAYOUT
# ============================================================

app.layout = html.Div(
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "30px",
        "fontFamily": "Arial, sans-serif",
    },
    children=[
        html.H1(
            "Retail Campaign Response Prediction",
            style={"textAlign": "center"},
        ),

        html.P(
            "Predict customer campaign response using the "
            "trained Random Forest machine-learning model.",
            style={
                "textAlign": "center",
                "fontSize": "18px",
            },
        ),

        html.Hr(),

        html.H2("Model Information"),

        html.Div(
            [
                html.P("Model: Random Forest"),
                html.P(f"Number of trees: {model.n_estimators}"),
                html.P(f"Number of features: {model.n_features_in_}"),
                html.P(f"Model file: {MODEL_PATH.name}"),
            ]
        ),

        html.Hr(),

        html.H2("Upload Customer CSV"),

        html.P(
            "Upload a CSV containing the 12 required model features. "
            "An optional CustomerID column is used to identify customers."
        ),

        dcc.Upload(
            id="upload-data",
            children=html.Div(
                [
                    "Drag and drop or ",
                    html.A("select a CSV file"),
                ]
            ),
            style={
                "width": "100%",
                "height": "80px",
                "lineHeight": "80px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "marginBottom": "20px",
            },
            multiple=False,
        ),

        html.Div(
            id="upload-status",
            style={"marginBottom": "20px", "fontWeight": "bold"},
        ),

        html.Div(id="upload-results"),

        html.Hr(),

        html.H2("Individual Customer Prediction"),

        html.P(
            "Enter customer feature values and select the prediction button."
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                "gap": "15px",
            },
            children=[
                html.Div([
                    html.Label("Number of Complaints"),
                    dcc.Input(
                        id="input-n-comp",
                        type="number",
                        value=3,
                        style={"width": "100%"},
                    ),
                ]),

                html.Div([
                    html.Label("Loyalty"),
                    dcc.Input(
                        id="input-loyalty",
                        type="number",
                        value=5,
                        style={"width": "100%"},
                    ),
                ]),

                html.Div([
                    html.Label("NPS"),
                    dcc.Input(
                        id="input-nps",
                        type="number",
                        value=50,
                        style={"width": "100%"},
                    ),
                ]),

                html.Div([
                    html.Label("Number of Communications"),
                    dcc.Input(
                        id="input-n-communications",
                        type="number",
                        value=4,
                        style={"width": "100%"},
                    ),
                ]),

                html.Div([
                    html.Label("Total Sales"),
                    dcc.Input(
                        id="input-total-sales",
                        type="number",
                        value=1000,
                        style={"width": "100%"},
                    ),
                ]),

                html.Div([
                    html.Label("Unique Products"),
                    dcc.Input(
                        id="input-unique-products",
                        type="number",
                        value=20,
                        style={"width": "100%"},
                    ),
                ]),

                html.Div([
                    html.Label("Number of Invoices"),
                    dcc.Input(
                        id="input-number-of-invoices",
                        type="number",
                        value=10,
                        style={"width": "100%"},
                    ),
                ]),

                html.Div([
                    html.Label("Purchase Days"),
                    dcc.Input(
                        id="input-purchase-days",
                        type="number",
                        value=8,
                        style={"width": "100%"},
                    ),
                ]),

                html.Div([
                    html.Label("Average Invoice Value"),
                    dcc.Input(
                        id="input-average-invoice-value",
                        type="number",
                        value=100,
                        style={"width": "100%"},
                    ),
                ]),

                html.Div([
                    html.Label("Average Quantity Per Invoice"),
                    dcc.Input(
                        id="input-average-quantity",
                        type="number",
                        value=5,
                        style={"width": "100%"},
                    ),
                ]),

                html.Div([
                    html.Label("Average Product Price"),
                    dcc.Input(
                        id="input-average-product-price",
                        type="number",
                        value=20,
                        style={"width": "100%"},
                    ),
                ]),

                html.Div([
                    html.Label("Customer Activity Days"),
                    dcc.Input(
                        id="input-customer-activity-days",
                        type="number",
                        value=365,
                        style={"width": "100%"},
                    ),
                ]),
            ],
        ),

        html.Br(),

        html.Button(
            "Predict Campaign Response",
            id="predict-button",
            n_clicks=0,
            style={
                "padding": "12px 25px",
                "fontSize": "16px",
                "cursor": "pointer",
            },
        ),

        html.Div(
            id="prediction-result",
            style={"marginTop": "20px"},
        ),

        dcc.Graph(
            id="probability-chart",
            style={"marginTop": "20px"},
        ),
    ],
)


# ============================================================
# CSV BATCH PREDICTION CALLBACK
# ============================================================

@app.callback(
    Output("upload-status", "children"),
    Output("upload-results", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def process_uploaded_file(contents, filename):

    if contents is None:
        raise PreventUpdate

    try:
        df = parse_uploaded_csv(contents, filename)
        X = prepare_prediction_data(df)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        probability_no_response = get_class_probability(
            probabilities, 0
        )
        probability_response = get_class_probability(
            probabilities, 1
        )

        # Copy the source data before adding prediction outputs.
        results_df = df.copy()
        results_df["predicted_response"] = predictions
        results_df["probability_no_response"] = probability_no_response
        results_df["probability_response"] = probability_response

        response_count = int((predictions == 1).sum())
        no_response_count = int((predictions == 0).sum())
        total_customers = len(predictions)

        response_rate = (
            response_count / total_customers
            if total_customers
            else 0
        )

        # CustomerID is optional; support files without that column.
        if "CustomerID" not in results_df.columns:
            results_df.insert(
                0,
                "CustomerID",
                [f"Row {i + 1}" for i in range(len(results_df))],
            )

        table_columns = [
            {
                "name": "Customer ID",
                "id": "CustomerID",
            },
            {
                "name": "Predicted Response",
                "id": "predicted_response",
            },
            {
                "name": "Probability No Response",
                "id": "probability_no_response",
                "type": "numeric",
                "format": dash_table.Format.Format(
                    precision=4,
                    scheme=dash_table.Format.Scheme.fixed,
                ),
            },
            {
                "name": "Probability Response",
                "id": "probability_response",
                "type": "numeric",
                "format": dash_table.Format.Format(
                    precision=4,
                    scheme=dash_table.Format.Scheme.fixed,
                ),
            },
        ]

        summary = html.Div(
            [
                html.H3("Prediction Results"),
                html.P(f"Total customers: {total_customers}"),
                html.P(f"Predicted responses: {response_count}"),
                html.P(f"Predicted non-responses: {no_response_count}"),
                html.P(f"Predicted response rate: {response_rate:.2%}"),

                dash_table.DataTable(
                    data=results_df[
                        [
                            "CustomerID",
                            "predicted_response",
                            "probability_no_response",
                            "probability_response",
                        ]
                    ].head(20).round(4).to_dict("records"),
                    columns=table_columns,
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "padding": "8px",
                        "minWidth": "130px",
                        "maxWidth": "260px",
                        "whiteSpace": "normal",
                    },
                    style_header={"fontWeight": "bold"},
                ),
            ]
        )

        status = html.Div(
            f"Upload successful: {filename or 'CSV file'} "
            f"({total_customers} customers processed).",
            style={"color": "green"},
        )

        print("CSV prediction completed successfully.", flush=True)

        return status, summary

    except Exception as exc:
        print(f"Upload failed: {exc}", flush=True)

        return (
            html.Div(
                f"Upload failed: {exc}",
                style={"color": "red"},
            ),
            html.Div(),
        )


# ============================================================
# INDIVIDUAL CUSTOMER PREDICTION CALLBACK
# ============================================================

@app.callback(
    Output("prediction-result", "children"),
    Output("probability-chart", "figure"),
    Input("predict-button", "n_clicks"),
    State("input-n-comp", "value"),
    State("input-loyalty", "value"),
    State("input-nps", "value"),
    State("input-n-communications", "value"),
    State("input-total-sales", "value"),
    State("input-unique-products", "value"),
    State("input-number-of-invoices", "value"),
    State("input-purchase-days", "value"),
    State("input-average-invoice-value", "value"),
    State("input-average-quantity", "value"),
    State("input-average-product-price", "value"),
    State("input-customer-activity-days", "value"),
)
def predict_individual_customer(
    n_clicks,
    n_comp,
    loyalty,
    nps,
    n_communications,
    total_sales,
    unique_products,
    number_of_invoices,
    purchase_days,
    average_invoice_value,
    average_quantity_per_invoice,
    average_product_price,
    customer_activity_days,
):

    if not n_clicks:
        return "", go.Figure()

    values = [
        n_comp,
        loyalty,
        nps,
        n_communications,
        total_sales,
        unique_products,
        number_of_invoices,
        purchase_days,
        average_invoice_value,
        average_quantity_per_invoice,
        average_product_price,
        customer_activity_days,
    ]

    if any(value is None for value in values):
        return (
            html.Div(
                "Please provide values for all fields.",
                style={"color": "red"},
            ),
            go.Figure(),
        )

    try:
        X = pd.DataFrame([values], columns=FEATURES)

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)

        probability_no_response = float(
            get_class_probability(probabilities, 0)[0]
        )
        probability_response = float(
            get_class_probability(probabilities, 1)[0]
        )

        result_text = (
            "Predicted Response"
            if prediction == 1
            else "Predicted No Response"
        )

        result = html.Div(
            [
                html.H3(result_text),
                html.P(
                    f"Response probability: "
                    f"{probability_response:.2%}"
                ),
                html.P(
                    f"No-response probability: "
                    f"{probability_no_response:.2%}"
                ),
                html.P(
                    "Probabilities are model estimates, not "
                    "guarantees of customer behaviour."
                ),
            ]
        )

        figure = create_probability_chart(
            probability_no_response,
            probability_response,
        )

        print("Individual prediction completed successfully.", flush=True)

        return result, figure

    except Exception as exc:
        print(
            f"Individual prediction failed: {exc}",
            flush=True,
        )

        return (
            html.Div(
                f"Prediction failed: {exc}",
                style={"color": "red"},
            ),
            go.Figure(),
        )


# ============================================================
# PRODUCTION / RENDER STARTUP
# ============================================================

if __name__ == "__main__":
    print(
        f"Starting Retail Campaign Dashboard on "
        f"{HOST}:{PORT}",
        flush=True,
    )

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
    )
