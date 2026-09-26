import os
import base64
import io
import joblib
import pandas as pd

from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = r"C:\Users\EmmanuelOgbonna\OneDrive - Deighton Associates Ltd\Desktop\AI and ML Data Science Institute\models\random_forest_model.pkl"

FEATURES = [
    "total_sales",
    "unique_products",
    "number_of_invoices",
    "nps",
    "n_comp",
    "n_communications",
    "loyalty"
]


# ============================================================
# Load model
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully.")
print("Number of trees:", len(model.estimators_))
print("Number of features:", model.n_features_in_)


# ============================================================
# Dash application
# ============================================================

app = Dash(__name__)

app.title = "Retail Campaign Response Predictor"


# ============================================================
# Layout
# ============================================================

app.layout = html.Div(
    [

        html.H1(
            "Retail Campaign Response Predictor",
            style={"textAlign": "center"}
        ),

        html.P(
            "Upload customer data or enter individual customer "
            "information to predict campaign response.",
            style={"textAlign": "center", "color": "#555"}
        ),

        html.Hr(),

        # ====================================================
        # CSV Upload
        # ====================================================

        html.Div(
            [

                html.H2("1. Upload Customer CSV"),

                dcc.Upload(
                    id="upload-data",
                    children=html.Div(
                        [
                            "Drag and drop your CSV here, or ",
                            html.A("click to select a file")
                        ]
                    ),
                    style={
                        "width": "100%",
                        "height": "80px",
                        "lineHeight": "80px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "marginBottom": "20px"
                    },
                    multiple=False
                ),

                html.Div(
                    id="upload-status"
                ),

                html.Div(
                    id="uploaded-data-preview"
                ),

                html.Div(
                    id="upload-predictions"
                ),

                html.Br(),

                html.Button(
                    "Download Predictions",
                    id="download-button",
                    style={
                        "padding": "10px 20px",
                        "cursor": "pointer"
                    }
                ),

                dcc.Download(
                    id="download-predictions"
                )

            ],
            style={
                "maxWidth": "1000px",
                "margin": "auto"
            }
        ),

        html.Hr(),

        # ====================================================
        # Individual Prediction
        # ====================================================

        html.Div(
            [

                html.H2("2. Individual Customer Prediction"),

                html.Label("Total Sales"),

                dcc.Input(
                    id="total-sales",
                    type="number",
                    value=1000,
                    min=0,
                    step=0.01,
                    style={"width": "100%"}
                ),

                html.Br(),
                html.Br(),

                html.Label("Unique Products"),

                dcc.Input(
                    id="unique-products",
                    type="number",
                    value=10,
                    min=0,
                    step=1,
                    style={"width": "100%"}
                ),

                html.Br(),
                html.Br(),

                html.Label("Number of Invoices"),

                dcc.Input(
                    id="number-of-invoices",
                    type="number",
                    value=5,
                    min=0,
                    step=1,
                    style={"width": "100%"}
                ),

                html.Br(),
                html.Br(),

                html.Label("NPS"),

                dcc.Input(
                    id="nps",
                    type="number",
                    value=8,
                    min=0,
                    max=10,
                    step=1,
                    style={"width": "100%"}
                ),

                html.Br(),
                html.Br(),

                html.Label("Number of Complaints"),

                dcc.Input(
                    id="n-comp",
                    type="number",
                    value=0,
                    min=0,
                    step=1,
                    style={"width": "100%"}
                ),

                html.Br(),
                html.Br(),

                html.Label("Number of Communications"),

                dcc.Input(
                    id="n-communications",
                    type="number",
                    value=3,
                    min=0,
                    step=1,
                    style={"width": "100%"}
                ),

                html.Br(),
                html.Br(),

                html.Label("Loyalty"),

                dcc.Input(
                    id="loyalty",
                    type="number",
                    value=5,
                    min=0,
                    step=1,
                    style={"width": "100%"}
                ),

                html.Br(),
                html.Br(),

                html.Button(
                    "Predict Campaign Response",
                    id="predict-button",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "12px",
                        "fontSize": "16px",
                        "cursor": "pointer"
                    }
                )

            ],
            style={
                "maxWidth": "600px",
                "margin": "auto"
            }
        ),

        html.Br(),

        html.Div(
            id="prediction-output",
            style={
                "textAlign": "center",
                "fontSize": "24px",
                "fontWeight": "bold"
            }
        ),

        dcc.Graph(
            id="probability-chart"
        )

    ],

    style={
        "padding": "40px",
        "fontFamily": "Arial"
    }
)


# ============================================================
# CSV Upload Callback
# ============================================================

@app.callback(
    [
        Output("upload-status", "children"),
        Output("uploaded-data-preview", "children"),
        Output("upload-predictions", "children")
    ],
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def upload_csv(contents, filename):

    if contents is None:
        return "", "", ""

    try:

        # Decode uploaded CSV
        content_type, content_string = contents.split(",")

        decoded = base64.b64decode(content_string)

        df = pd.read_csv(
            io.StringIO(
                decoded.decode("utf-8")
            )
        )

        # ----------------------------------------------------
        # Validate required columns
        # ----------------------------------------------------

        missing_columns = [
            column
            for column in FEATURES
            if column not in df.columns
        ]

        if missing_columns:

            return (
                html.P(
                    "Upload failed. Missing required columns: "
                    + ", ".join(missing_columns),
                    style={"color": "red"}
                ),
                "",
                ""
            )

        # ----------------------------------------------------
        # Convert model features to numeric
        # ----------------------------------------------------

        for column in FEATURES:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # Check missing values
        # ----------------------------------------------------

        missing_values = df[FEATURES].isnull().sum()

        rows_with_missing = missing_values.sum()

        if rows_with_missing > 0:

            return (
                html.P(
                    "Upload failed. Some required feature values "
                    "are missing or non-numeric.",
                    style={"color": "red"}
                ),
                "",
                ""
            )

        # ----------------------------------------------------
        # Make predictions
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            df[FEATURES]
        )[:, 1]

        predictions = model.predict(
            df[FEATURES]
        )

        results = df.copy()

        results["response_probability"] = probabilities

        results["predicted_response"] = predictions

        results["predicted_response_label"] = results[
            "predicted_response"
        ].map(
            {
                0: "No Response",
                1: "Response"
            }
        )

        # ----------------------------------------------------
        # Preview
        # ----------------------------------------------------

        preview = html.Div(
            [
                html.H4(
                    f"Uploaded file: {filename}"
                ),

                html.P(
                    f"Rows uploaded: {len(df)}"
                ),

                html.P(
                    f"Columns detected: {len(df.columns)}"
                ),

                html.H4("Data Preview"),

                dcc.Markdown(
                    results.head(10).to_markdown(
                        index=False
                    )
                )
            ]
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        response_count = int(
            (predictions == 1).sum()
        )

        no_response_count = int(
            (predictions == 0).sum()
        )

        average_probability = (
            probabilities.mean() * 100
        )

        prediction_summary = html.Div(
            [

                html.H4(
                    "Prediction Results"
                ),

                html.P(
                    f"Predicted responses: "
                    f"{response_count}"
                ),

                html.P(
                    f"Predicted non-responses: "
                    f"{no_response_count}"
                ),

                html.P(
                    f"Average response probability: "
                    f"{average_probability:.2f}%"
                )

            ],
            style={
                "marginTop": "20px",
                "padding": "15px",
                "border": "1px solid #ddd"
            }
        )

        # Store results in hidden component
        prediction_table = html.Div(
            id="prediction-results-data",
            children=""
        )

        # Save results temporarily in app memory
        app.server.config[
            "LATEST_RESULTS"
        ] = results

        return (
            html.P(
                f"Successfully uploaded {filename}.",
                style={"color": "green"}
            ),
            preview,
            html.Div(
                [
                    prediction_summary,
                    prediction_table
                ]
            )
        )

    except Exception as e:

        return (
            html.P(
                f"Error processing CSV: {str(e)}",
                style={"color": "red"}
            ),
            "",
            ""
        )


# ============================================================
# Download Predictions
# ============================================================

@app.callback(
    Output("download-predictions", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_results(n_clicks):

    results = app.server.config.get(
        "LATEST_RESULTS"
    )

    if results is None:
        return None

    return dcc.send_data_frame(
        results.to_csv,
        "campaign_predictions.csv",
        index=False
    )


# ============================================================
# Individual Customer Prediction
# ============================================================

@app.callback(
    [
        Output("prediction-output", "children"),
        Output("probability-chart", "figure")
    ],

    Input("predict-button", "n_clicks"),

    [
        State("total-sales", "value"),
        State("unique-products", "value"),
        State("number-of-invoices", "value"),
        State("nps", "value"),
        State("n-comp", "value"),
        State("n-communications", "value"),
        State("loyalty", "value")
    ]
)
def predict_response(
    n_clicks,
    total_sales,
    unique_products,
    number_of_invoices,
    nps,
    n_comp,
    n_communications,
    loyalty
):

    empty_figure = go.Figure()

    if not n_clicks:

        return (
            "Enter customer information and click Predict.",
            empty_figure
        )

    values = [
        total_sales,
        unique_products,
        number_of_invoices,
        nps,
        n_comp,
        n_communications,
        loyalty
    ]

    if any(value is None for value in values):

        return (
            "Please complete all fields.",
            empty_figure
        )

    input_data = pd.DataFrame(
        [{
            "total_sales": total_sales,
            "unique_products": unique_products,
            "number_of_invoices": number_of_invoices,
            "nps": nps,
            "n_comp": n_comp,
            "n_communications": n_communications,
            "loyalty": loyalty
        }],
        columns=FEATURES
    )

    probability = model.predict_proba(
        input_data
    )[0, 1]

    prediction = model.predict(
        input_data
    )[0]

    probability_percent = probability * 100

    if prediction == 1:

        result = (
            f"Predicted Response: YES — "
            f"{probability_percent:.2f}% probability"
        )

    else:

        result = (
            f"Predicted Response: NO — "
            f"{probability_percent:.2f}% probability"
        )

    figure = go.Figure(
        go.Bar(
            x=[
                "No Response",
                "Response"
            ],
            y=[
                (1 - probability) * 100,
                probability * 100
            ],
            text=[
                f"{(1 - probability) * 100:.2f}%",
                f"{probability * 100:.2f}%"
            ],
            textposition="auto"
        )
    )

    figure.update_layout(
        title="Campaign Response Probability",
        yaxis_title="Probability (%)",
        xaxis_title="Outcome",
        yaxis={"range": [0, 100]}
    )

    return result, figure


# ============================================================
# Run application
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)