# AIML-Retail-Campaign-Project

## Campaign Response Prediction

This is an end-to-end machine learning project analysing retail transaction and campaign response data to understand customer behaviour and predict customer response to marketing campaigns.

The project covers data understanding, data cleaning, customer-level feature engineering, exploratory data analysis (EDA), NPS analysis, machine learning model development, evaluation, and API deployment.

---

## Project Overview

This project analyses customer transaction and campaign response data to identify factors associated with customer response to a retail marketing campaign.

The project progresses from exploratory data analysis through to machine learning and deployment. A Random Forest classification model is used to predict whether a customer is likely to respond to a campaign.

A FastAPI service is included to provide predictions through a REST API, while the Dash application provides an interactive interface for exploring and using the model.

---

## Objectives

* Understand the transaction and campaign datasets
* Perform data quality checks
* Identify missing and duplicate records
* Investigate unexpected values such as negative quantities
* Engineer customer-level features
* Analyse NPS categories
* Calculate campaign response rates
* Compare customer characteristics between responders and non-responders
* Visualise relationships between customer features and campaign response
* Train and evaluate machine learning classification models
* Identify important predictors of campaign response
* Deploy the trained model through a FastAPI REST API
* Provide an interactive Dash application for model analysis and predictions

---

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Jupyter Notebook
* FastAPI
* Uvicorn
* Pydantic
* Plotly
* Dash

---

## Project Structure

```text
AIML-Retail-Campaign-Project/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── models/
│   └── random_forest_model.pkl
│
├── notebooks/
│   └── Retail_Campaign_Project.ipynb
│
├── api.py.ipynb
├── app.py
├── requirements.txt
└── README.md
```

### Key Files

| File                                      | Description                                                      
| `notebooks/Retail_Campaign_Project.ipynb` | Main data analysis, feature engineering, modelling and evaluation notebook |
| `api/main.py`                             | FastAPI application for model deployment                                   |
| `models/random_forest_model.pkl`          | Trained Random Forest classification model                                 |
| `app.py`                                  | Dash web application                                                       |
| `api.py.ipynb`                            | API development and testing notebook                                       |
| `requirements.txt`                        | Python package dependencies                                                |
| `README.md`                               | Project and deployment documentation                                       |

---

# Data Analysis

## Analysis Stages

The analysis includes:

1. Data Understanding
2. Data Quality Assessment
3. Data Cleaning
4. Feature Engineering
5. Exploratory Data Analysis
6. NPS Analysis
7. Campaign Response Analysis
8. Response Rate Analysis
9. Machine Learning Modelling
10. Model Evaluation
11. Model Interpretation
12. API Deployment

---

## Key EDA

Examples of visualisations and analysis include:

* Campaign response distribution
* Response rate by NPS category
* Mean customer features by campaign response
* Customer purchasing behaviour
* Distribution of customer-level variables
* Customer activity and purchasing patterns
* Relationships between customer characteristics and campaign response

---

# Dataset

The project uses retail transaction data and campaign response data.

The transaction dataset contains customer purchasing information, including:

* Customer ID
* Invoice number
* Stock code
* Product description
* Quantity
* Invoice date
* Unit price
* Country

The campaign dataset contains customer campaign information, including:

* Customer ID
* Campaign response
* Number of campaign components
* Loyalty
* NPS
* Number of communications

The original datasets are not included in this repository where they contain restricted or customer-level information.

---

# Feature Engineering

Customer-level features were created from the transaction and campaign data.

Features used in the final Random Forest model are:

```text
total_sales
unique_products
number_of_invoices
nps
n_comp
n_communications
loyalty
```

Additional customer-level variables were also investigated during the analysis, including:

* Purchase days
* Average invoice value
* Average quantity per invoice
* Average product price
* Customer activity days

---

# Campaign Results

The campaign dataset contained 3,834 customers.

| Category       | Customers |
| Responders     |     1,555 |
| Non-responders |     2,279 |
| Total          |     3,834 |

The overall campaign response rate was 40.56%.

Response rates also varied substantially across NPS categories, providing evidence that customer satisfaction and loyalty-related characteristics were relevant areas for further analysis.

---

# Machine Learning

## Problem Definition

The machine learning task is a **binary classification problem**.

The target variable represents whether a customer responded to the campaign:

```text
0 = Non-response
1 = Response
```

The data was divided into training and testing datasets using a stratified split to preserve the class distribution.

## Random Forest Model

The final deployed model is a `RandomForestClassifier`.

The model was trained using the following seven features:

```text
total_sales
unique_products
number_of_invoices
nps
n_comp
n_communications
loyalty
```

The trained model is saved using Joblib:

```text
models/random_forest_model.pkl
```

The model uses:

* `n_estimators = 100`
* `random_state = 42`

The project also evaluates model performance using classification and probability-based metrics.

---

# Model Deployment

The trained Random Forest model is deployed using **FastAPI**.

The API provides routing, input validation, model loading and prediction functionality.

## API Endpoints

| Endpoint      | Method | Description                                       |
| `/`           | GET    | Returns API information and model status          |
| `/health`     | GET    | Checks whether the API and model are healthy      |
| `/model-info` | GET    | Returns model configuration and expected features |
| `/predict`    | POST   | Generates a campaign response prediction          |

---

# Running the FastAPI Service

## 1. Open Anaconda Prompt

Activate the project environment:

```bash
conda activate stats_env
```

## 2. Navigate to the Project Directory

```bash
cd "C:\Users\EmmanuelOgbonna\OneDrive - Deighton Associates Ltd\Desktop\AI and ML Data Science Institute\AIML-Retail-Campaign-Project"
```

## 3. Install Dependencies

Install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The project uses **scikit-learn 1.6.1**, which matches the version used to train and save the deployed Random Forest model.

## 4. Start the API

Run:

```bash
uvicorn api.main:app --reload
```

The API will be available locally at:

```text
http://127.0.0.1:8000
```

---

# FastAPI Swagger Documentation

FastAPI automatically provides interactive API documentation.

Open the following address in a browser:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface can be used to test all API endpoints without requiring an external API client.

---

# Testing the Prediction API

The `/predict` endpoint accepts the following customer information:

```json
{
  "total_sales": 500,
  "unique_products": 10,
  "number_of_invoices": 5,
  "nps": 50,
  "n_comp": 3,
  "n_communications": 4,
  "loyalty": 5
}
```

The API returns:

* Predicted class
* Prediction label
* Response probability
* Non-response probability

Example response structure:

```json
{
  "prediction": 1,
  "prediction_label": "Response",
  "response_probability": 0.75,
  "non_response_probability": 0.25
}
```

The probability values shown above are illustrative; the actual values are generated by the trained model.

---

# Health Check

The `/health` endpoint can be used to confirm that the API has successfully loaded the trained model.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "RandomForestClassifier",
  "features_expected": 7
}
```

---

# Dash Application

The project also contains a Dash web application in:

```text
app.py
```

The application provides an interactive interface for analysing campaign data and working with the trained machine learning model.

To run the Dash application:

```bash
python app.py
```

The application is normally accessible locally at:

```text
http://127.0.0.1:8050/
```

---

# Requirements

The main Python dependencies are listed in `requirements.txt`.

The project requires:

```text
fastapi
uvicorn
pydantic
pandas
numpy
joblib
scikit-learn==1.6.1
```

Additional packages required by the Dash application should also be included in `requirements.txt`.

---

# Deployment Considerations

The current implementation is designed for **local deployment and demonstration**.

The FastAPI service:

* Loads the trained model from the project `models` directory
* Validates prediction inputs using Pydantic
* Uses pandas DataFrames with the correct feature names
* Provides health and model information endpoints
* Returns both the predicted class and prediction probabilities

For production deployment, additional configuration would be required, such as:

* Production application server configuration
* Environment-specific configuration
* Authentication and authorisation
* HTTPS
* Logging and monitoring
* Containerisation
* Cloud hosting
* Secure model and configuration management

---

# Troubleshooting

## Scikit-learn Version Warning

The saved model was trained using scikit-learn **1.6.1**.

If a different version is used, scikit-learn may display an `InconsistentVersionWarning` when loading the model.

Install the matching version with:

```bash
pip install scikit-learn==1.6.1
```

## ModuleNotFoundError: No module named 'api'

Make sure the terminal is located in the project root directory:

```text
AIML-Retail-Campaign-Project
```

Then run:

```bash
uvicorn api.main:app --reload
```

## Model Feature Error

The deployed Random Forest model expects **7 features**.

The feature names and order must remain:

```text
total_sales
unique_products
number_of_invoices
nps
n_comp
n_communications
loyalty
```

The API constructs a pandas DataFrame using these exact feature names before passing the data to the model.
# Reproducibility

The project includes a `requirements.txt` file containing the Python dependencies required to reproduce the analysis and run the deployed applications.

## Environment Setup

It is recommended to use a dedicated Python environment.

Using Anaconda:

```bash
conda create -n retail_campaign python=3.11
conda activate retail_campaign
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The project uses scikit-learn version `1.6.1`, which matches the version used to train and save the Random Forest model.

## Running the Project

### Jupyter Notebook

Start Jupyter Notebook from the project directory:

```bash
jupyter notebook
```

Open:

```text
notebooks/Retail_Campaign_Project.ipynb
```

Run the notebook cells sequentially to reproduce the data preparation, feature engineering, exploratory analysis and machine learning workflow.

### FastAPI

From the project root directory, start the API using:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Dash Application

The interactive Dash application can be started using:

```bash
python app.py
```

The application will normally be available at:

```text
http://127.0.0.1:8050/
```

## Model Reproducibility

The trained Random Forest model is stored in:

```text
models/random_forest_model.pkl
```

The model expects the following seven input features:

```text
total_sales
unique_products
number_of_invoices
nps
n_comp
n_communications
loyalty
```

The model configuration includes:

```text
Model: RandomForestClassifier
n_estimators: 100
random_state: 42
scikit-learn: 1.6.1
```

Using the specified dependency versions and project structure helps ensure that the saved model can be loaded consistently.

---
### Possible Model Enhancements

As a further enhancement, the model could incorporate additional **non-sensitive demographic and customer-preference data** that customers may be comfortable providing voluntarily. Examples could include broad **age ranges**, **nationality**, preferred communication channel, and general customer preferences.

For example, customers could be grouped into broad age ranges such as **0–9, 10–19, 20–29, 30–39, 40–49, 50–59, 60–69, 70–79, 80–89 and 90–99**. This could allow the model to identify whether campaign response patterns differ across age groups.

Other voluntarily provided information, such as **nationality or preferred communication method**, could also be evaluated as potential predictive features. Combining these attributes with existing behavioural features such as **loyalty, NPS, total sales, number of invoices and communication history** could provide a more detailed understanding of customer response patterns.

The enhanced model could then investigate which customer segments are more likely to respond to different campaign types. This could support more targeted campaign research and potentially improve campaign effectiveness.

Any additional customer information should be collected transparently, with appropriate **consent, data-protection controls, anonymisation where appropriate, and consideration of potential bias and fairness**.

# Author

**Emmanuel Ogbonna**

AI & Machine Learning Data Science Institute Project

