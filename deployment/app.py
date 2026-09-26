from pathlib import Path
import base64
import io

import joblib
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"


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


print("Loading Random Forest model...")

try:
    model = joblib.load(MODEL_PATH)

    print("Random Forest model loaded successfully.")
    print(f"Model path: {MODEL_PATH}")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Number of features: {model.n_features_in_}")

    if hasattr(model, "feature_names_in_"):
        print(f"Model features: {list(model.feature_names_in_)}")

except Exception as e:
    print(f"Error loading model: {e}")
    raise


if model.n_features_in_ != len(FEATURES):
    raise ValueError(
        f"Model expects {model.n_features_in_} features, "
        f"but the application defines {len(FEATURES)} features."
    )


app = Dash(__name__)

app.title = "Retail Campaign Response Prediction"


def parse_uploaded_csv(contents, filename):
    """
    Decode and load an uploaded CSV file.
    """

    if contents is None:
        raise ValueError("No file was uploaded.")

    try:
        content_type, content_string = contents.split(",", 1)

        decoded = base64.b64decode(content_string)

        df = pd.read_csv(
            io.StringIO(decoded.decode("utf-8-sig"))
        )

    except Exception as e:
        raise ValueError(
            f"Unable to read uploaded CSV file: {e}"
        )

    print()
    print(f"Uploaded file: {filename}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


def prepare_prediction_data(df):
    """
    Prepare uploaded data for prediction.

    Handles:
    - customer_lifetime_days -> customer_activity_days
    - numeric conversion
    - missing NPS values
    - feature ordering
    """

    df = df.copy()

    # Handle the naming difference between the dataset and trained model.
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
            "Renamed customer_lifetime_days "
            "to customer_activity_days."
        )

    missing_columns = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {', '.join(missing_columns)}"
        )

    # Convert all model features to numeric values.
    for column in FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # NPS had 3 missing values in the master dataset.
    # Use the median, matching the preprocessing used for modelling.
    missing_nps = df["nps"].isna().sum()

    print(f"Missing NPS values: {missing_nps}")

    if missing_nps > 0:

        nps_median = df["nps"].median()

        if pd.isna(nps_median):
            raise ValueError(
                "NPS contains missing values and a median "
                "could not be calculated."
            )

        df["nps"] = df["nps"].fillna(nps_median)

        print(f"NPS median: {nps_median}")
        print(
            f"Imputed {missing_nps} missing NPS values."
        )

    # Check for any remaining invalid values.
    problematic_columns = []

    for column in FEATURES:

        if df[column].isna().any():
            problematic_columns.append(column)

    if problematic_columns:
        raise ValueError(
            "Some required feature values are missing "
            "or non-numeric: "
            + ", ".join(problematic_columns)
        )

    # Ensure exact same feature order used during training.
    X = df[FEATURES].copy()

    print(
        f"Prediction features: {list(X.columns)}"
    )

    print(
        f"Feature count: {X.shape[1]}"
    )

    print(
        f"Model feature count: {model.n_features_in_}"
    )

    if X.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature mismatch. "
            f"Application has {X.shape[1]} features, "
            f"but model expects {model.n_features_in_}."
        )

    return X


def create_probability_chart(probability_0, probability_1):
    """
    Create a probability bar chart.
    """

    fig = go.Figure(
        data=[
            go.Bar(
                x=[
                    "No Response",
                    "Response"
                ],
                y=[
                    probability_0,
                    probability_1
                ],
                text=[
                    f"{probability_0:.1%}",
                    f"{probability_1:.1%}"
                ],
                textposition="auto"
            )
        ]
    )

    fig.update_layout(
        title="Campaign Response Probability",
        yaxis_title="Probability",
        yaxis=dict(
            range=[0, 1],
            tickformat=".0%"
        ),
        xaxis_title="Prediction"
    )

    return fig


app.layout = html.Div(
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "30px",
        "fontFamily": "Arial"
    },

    children=[

        html.H1(
            "Retail Campaign Response Prediction",
            style={
                "textAlign": "center"
            }
        ),

        html.P(
            "Predict customer campaign response using "
            "the trained Random Forest model.",
            style={
                "textAlign": "center",
                "fontSize": "18px"
            }
        ),

        html.Hr(),

        html.H2("Model Information"),

        html.Div(
            [
                html.P(
                    f"Model: Random Forest"
                ),
                html.P(
                    f"Number of trees: {model.n_estimators}"
                ),
                html.P(
                    f"Number of features: {model.n_features_in_}"
                ),
                html.P(
                    f"Model file: {MODEL_PATH.name}"
                )
            ]
        ),

        html.Hr(),

        html.H2("Upload Customer CSV"),

        dcc.Upload(
            id="upload-data",

            children=html.Div(
                [
                    "Drag and Drop or ",
                    html.A("Select a CSV File")
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
                "marginBottom": "20px"
            },

            multiple=False
        ),

        html.Div(
            id="upload-status",
            style={
                "marginBottom": "20px",
                "fontWeight": "bold"
            }
        ),

        html.Div(
            id="upload-results"
        ),

        html.Hr(),

        html.H2("Individual Customer Prediction"),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns":
                    "repeat(3, 1fr)",
                "gap": "15px"
            },

            children=[

                html.Div(
                    [
                        html.Label("Number of Complaints"),
                        dcc.Input(
                            id="input-n-comp",
                            type="number",
                            value=3,
                            style={"width": "100%"}
                        )
                    ]
                ),

                html.Div(
                    [
                        html.Label("Loyalty"),
                        dcc.Input(
                            id="input-loyalty",
                            type="number",
                            value=5,
                            style={"width": "100%"}
                        )
                    ]
                ),

                html.Div(
                    [
                        html.Label("NPS"),
                        dcc.Input(
                            id="input-nps",
                            type="number",
                            value=50,
                            style={"width": "100%"}
                        )
                    ]
                ),

                html.Div(
                    [
                        html.Label(
                            "Number of Communications"
                        ),
                        dcc.Input(
                            id="input-n-communications",
                            type="number",
                            value=4,
                            style={"width": "100%"}
                        )
                    ]
                ),

                html.Div(
                    [
                        html.Label("Total Sales"),
                        dcc.Input(
                            id="input-total-sales",
                            type="number",
                            value=1000,
                            style={"width": "100%"}
                        )
                    ]
                ),

                html.Div(
                    [
                        html.Label("Unique Products"),
                        dcc.Input(
                            id="input-unique-products",
                            type="number",
                            value=20,
                            style={"width": "100%"}
                        )
                    ]
                ),

                html.Div(
                    [
                        html.Label(
                            "Number of Invoices"
                        ),
                        dcc.Input(
                            id="input-number-of-invoices",
                            type="number",
                            value=10,
                            style={"width": "100%"}
                        )
                    ]
                ),

                html.Div(
                    [
                        html.Label("Purchase Days"),
                        dcc.Input(
                            id="input-purchase-days",
                            type="number",
                            value=8,
                            style={"width": "100%"}
                        )
                    ]
                ),

                html.Div(
                    [
                        html.Label(
                            "Average Invoice Value"
                        ),
                        dcc.Input(
                            id="input-average-invoice-value",
                            type="number",
                            value=100,
                            style={"width": "100%"}
                        )
                    ]
                ),

                html.Div(
                    [
                        html.Label(
                            "Average Quantity Per Invoice"
                        ),
                        dcc.Input(
                            id="input-average-quantity",
                            type="number",
                            value=5,
                            style={"width": "100%"}
                        )
                    ]
                ),

                html.Div(
                    [
                        html.Label(
                            "Average Product Price"
                        ),
                        dcc.Input(
                            id="input-average-product-price",
                            type="number",
                            value=20,
                            style={"width": "100%"}
                        )
                    ]
                ),

                html.Div(
                    [
                        html.Label(
                            "Customer Activity Days"
                        ),
                        dcc.Input(
                            id="input-customer-activity-days",
                            type="number",
                            value=365,
                            style={"width": "100%"}
                        )
                    ]
                )
            ]
        ),

        html.Br(),

        html.Button(
            "Predict Campaign Response",
            id="predict-button",
            n_clicks=0,
            style={
                "padding": "12px 25px",
                "fontSize": "16px",
                "cursor": "pointer"
            }
        ),

        html.Div(
            id="prediction-result",
            style={
                "marginTop": "20px"
            }
        ),

        dcc.Graph(
            id="probability-chart",
            style={
                "marginTop": "20px"
            }
        )
    ]
)


@app.callback(
    Output("upload-status", "children"),
    Output("upload-results", "children"),

    Input("upload-data", "contents"),
    State("upload-data", "filename"),

    prevent_initial_call=True
)
def process_uploaded_file(contents, filename):

    if contents is None:
        raise PreventUpdate

    try:

        df = parse_uploaded_csv(
            contents,
            filename
        )

        X = prepare_prediction_data(df)

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)

        df["predicted_response"] = predictions

        df["probability_no_response"] = (
            probabilities[:, 0]
        )

        df["probability_response"] = (
            probabilities[:, 1]
        )

        response_count = int(
            (predictions == 1).sum()
        )

        no_response_count = int(
            (predictions == 0).sum()
        )

        response_rate = (
            response_count / len(predictions)
            if len(predictions) > 0
            else 0
        )

        print("CSV prediction completed successfully.")

        summary = html.Div(
            [

                html.H3(
                    "Prediction Results"
                ),

                html.P(
                    f"Total customers: {len(df)}"
                ),

                html.P(
                    f"Predicted responses: {response_count}"
                ),

                html.P(
                    f"Predicted non-responses: "
                    f"{no_response_count}"
                ),

                html.P(
                    f"Predicted response rate: "
                    f"{response_rate:.2%}"
                ),

                dash_table.DataTable(
                    data=df[
                        [
                            "CustomerID",
                            "predicted_response",
                            "probability_no_response",
                            "probability_response"
                        ]
                    ].head(20).round(4).to_dict(
                        "records"
                    ),

                    columns=[
                        {
                            "name": "Customer ID",
                            "id": "CustomerID"
                        },
                        {
                            "name": "Predicted Response",
                            "id": "predicted_response"
                        },
                        {
                            "name": "Probability No Response",
                            "id": "probability_no_response"
                        },
                        {
                            "name": "Probability Response",
                            "id": "probability_response"
                        }
                    ],

                    page_size=20,

                    style_table={
                        "overflowX": "auto"
                    },

                    style_cell={
                        "textAlign": "left",
                        "padding": "8px"
                    }
                )
            ]
        )

        status = html.Div(
            [
                html.Span(
                    "Upload successful: "
                    f"{filename}"
                )
            ]
        )

        return status, summary

    except Exception as e:

        print(f"Upload failed: {e}")

        return (
            html.Div(
                [
                    "Upload failed: ",
                    str(e)
                ],
                style={
                    "color": "red"
                }
            ),
            html.Div()
        )


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
    State("input-customer-activity-days", "value")
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
    customer_activity_days
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
        customer_activity_days
    ]

    if any(value is None for value in values):

        return (
            html.Div(
                "Please provide values for all fields.",
                style={
                    "color": "red"
                }
            ),
            go.Figure()
        )

    try:

        X = pd.DataFrame(
            [values],
            columns=FEATURES
        )

        print()
        print(
            "Individual prediction features:"
        )
        print(list(X.columns))

        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        probability_no_response = probabilities[0]
        probability_response = probabilities[1]

        if prediction == 1:
            result_text = "Predicted Response"
        else:
            result_text = "Predicted No Response"

        result = html.Div(
            [

                html.H3(
                    result_text
                ),

                html.P(
                    f"Response probability: "
                    f"{probability_response:.2%}"
                ),

                html.P(
                    f"No-response probability: "
                    f"{probability_no_response:.2%}"
                )
            ]
        )

        figure = create_probability_chart(
            probability_no_response,
            probability_response
        )

        return result, figure

    except Exception as e:

        print(
            f"Individual prediction failed: {e}"
        )

        return (
            html.Div(
                f"Prediction failed: {e}",
                style={
                    "color": "red"
                }
            ),
            go.Figure()
        )


if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=8050
    )