AIML Retail Campaign Project

Campaign Response Prediction

This is an end-to-end machine learning project analysing retail transaction and campaign response data to understand customer behaviour and predict customer response to marketing campaigns.

The project covers data understanding, data cleaning, customer-level feature engineering, exploratory data analysis, NPS analysis, campaign response analysis, machine learning model development, evaluation, model interpretation, API deployment and interactive dashboard deployment.

PROJECT OVERVIEW

This project analyses customer transaction and campaign response data to identify factors associated with customer response to a retail marketing campaign.

The project progresses from exploratory data analysis through to machine learning and deployment.

Three classification models were evaluated:

* Random Forest
* XGBoost
* Logistic Regression

The Random Forest model is currently used by the deployed FastAPI and Dash applications.

The FastAPI service provides predictions through a REST API, while the Dash application provides an interactive interface for uploading customer data and generating predictions.

OBJECTIVES

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
* Produce reproducible outputs and evaluation artefacts

TECHNOLOGIES

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* SHAP
* Joblib
* Jupyter Notebook
* FastAPI
* Uvicorn
* Pydantic
* Plotly
* Dash

PROJECT STRUCTURE

AIML-Retail-Campaign-Project

```
DATASETS

    Campaign Response.csv
    Master Campaign Response Data.csv
    Online Retail Sales Data.csv

notebooks

    AIML_Retail_Campaign_Project.ipynb

api

    __init__.py
    main.py

deployment

    app.py

models

    logistic_regression_model.pkl
    random_forest_model.pkl
    xgboost_model.pkl

outputs

    evaluation

        classification_report.csv
        confusion_matrix.csv
        evaluation_results.txt
        feature_importance.csv
        metrics.csv
        test_predictions.csv

    plots

        confusion_matrix.png
        feature_importance.png
        roc_curve.png
        shap_summary.png

requirements.txt
README.md
.gitignore
LICENSE
```

KEY FILES

notebooks/AIML_Retail_Campaign_Project.ipynb

Main data analysis, data cleaning, feature engineering, exploratory analysis, modelling and evaluation notebook.

api/main.py

FastAPI application providing model loading, input validation, health checks, model information and prediction functionality.

deployment/app.py

Dash application providing an interactive interface for CSV uploads and individual customer predictions.

models/random_forest_model.pkl

Trained Random Forest classification model used by the deployed applications.

models/logistic_regression_model.pkl

Saved Logistic Regression model used during model comparison.

models/xgboost_model.pkl

Saved XGBoost model used during model comparison.

outputs/evaluation

Contains final model evaluation results and prediction outputs.

outputs/plots

Contains exported visualisations used during analysis and reporting.

requirements.txt

Contains the Python dependencies required to reproduce the project and run the deployed applications.

README.md

Project documentation and deployment instructions.

DATA ANALYSIS

ANALYSIS STAGES

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
13. Interactive Dashboard Deployment

KEY EDA

Examples of visualisations and analysis include:

* Campaign response distribution
* Response rate by NPS category
* Mean customer features by campaign response
* Customer purchasing behaviour
* Distribution of customer-level variables
* Customer activity and purchasing patterns
* Relationships between customer characteristics and campaign response

DATASET

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

The project also creates a master customer-level dataset by combining campaign information with aggregated transaction features.

The original datasets contain customer-level information and should be handled according to the applicable data protection and project requirements.

FEATURE ENGINEERING

Customer-level features were created from the transaction and campaign data.

The final deployed Random Forest model uses the following 12 features:

* n_comp
* loyalty
* nps
* n_communications
* total_sales
* unique_products
* number_of_invoices
* purchase_days
* average_invoice_value
* average_quantity_per_invoice
* average_product_price
* customer_activity_days

The source master dataset contains the column `customer_lifetime_days`.

During Dash CSV deployment, this column is mapped to `customer_activity_days` so that the uploaded dataset matches the feature name expected by the trained model.

The NPS field contains three missing values in the master dataset. During deployment these missing NPS values are replaced using the median NPS value of 4.0.

CAMPAIGN RESULTS

The campaign dataset contains 3,834 customers.

* Responders: 1,555
* Non-responders: 2,279
* Total customers: 3,834

The overall campaign response rate is approximately 40.56 percent.

Response rates also vary across NPS categories, providing an area for further analysis of customer satisfaction and campaign response behaviour.

MACHINE LEARNING

PROBLEM DEFINITION

The machine learning task is a binary classification problem.

The target variable represents whether a customer responded to the campaign.

* 0 represents non-response
* 1 represents response

The data was divided into training and testing datasets using an 80 percent training and 20 percent testing split.

A stratified split with random state 42 was used to preserve the class distribution between the training and testing datasets.

MODELS EVALUATED

Three classification models were evaluated:

* Random Forest
* XGBoost
* Logistic Regression

The models were compared using:

* Accuracy
* Precision
* Recall
* F1 score
* ROC AUC

MODEL EVALUATION RESULTS

Random Forest

* Accuracy: 0.6323
* Precision: 0.5848
* Recall: 0.3215
* F1 score: 0.4149
* ROC AUC: 0.6603

XGBoost

* Accuracy: 0.6714
* Precision: 0.6903
* Recall: 0.3441
* F1 score: 0.4592
* ROC AUC: 0.6678

Logistic Regression

* Accuracy: 0.6441
* Precision: 0.5990
* Recall: 0.3698
* F1 score: 0.4573
* ROC AUC: 0.6343

XGBoost produced the highest value among the three evaluated models for Accuracy, Precision, F1 score and ROC AUC.

Logistic Regression produced the highest Recall among the three evaluated models.

The Random Forest model is used for the current deployment implementation.

RANDOM FOREST MODEL

The deployed model is a RandomForestClassifier.

The model configuration includes:

* n_estimators = 100
* random_state = 42

The model uses the following 12 features:

* n_comp
* loyalty
* nps
* n_communications
* total_sales
* unique_products
* number_of_invoices
* purchase_days
* average_invoice_value
* average_quantity_per_invoice
* average_product_price
* customer_activity_days

The trained model is saved as:

models/random_forest_model.pkl

MODEL INTERPRETATION

The project includes feature importance analysis and SHAP analysis to investigate which features contribute to model predictions.

The generated interpretation artefacts are stored in:

outputs/plots/feature_importance.png

outputs/plots/shap_summary.png

The feature importance results should be interpreted as model-specific associations rather than evidence of causal relationships.

MODEL DEPLOYMENT

The trained Random Forest model is deployed through FastAPI.

The API provides:

* Routing
* Input validation
* Model loading
* Prediction functionality
* Prediction probabilities
* Health checking
* Model information

FASTAPI ENDPOINTS

GET /

Returns basic API information and model status.

GET /health

Checks whether the API and trained model are available.

GET /model-info

Returns information about the model and the features expected by the API.

POST /predict

Accepts customer feature values and returns a campaign response prediction and probabilities.

RUNNING THE FASTAPI SERVICE

1. Open Anaconda Prompt or PowerShell.

2. Navigate to the project directory.

Use the project root directory containing the `api`, `deployment`, `models` and `requirements.txt` folders.

3. Create or activate a suitable Python environment.

For example:

conda create -n retail_campaign python=3.11

conda activate retail_campaign

4. Install the project dependencies.

pip install -r requirements.txt

5. Start the FastAPI service.

python -m uvicorn api.main:app --reload

The API will be available locally at:

http://127.0.0.1:8000

FASTAPI SWAGGER DOCUMENTATION

FastAPI automatically provides interactive API documentation.

Open the following address in a browser:

http://127.0.0.1:8000/docs

The Swagger interface can be used to test the available API endpoints and the prediction service.

TESTING THE PREDICTION API

The `/predict` endpoint accepts the following 12 customer features:

```json
{
  "n_comp": 3,
  "loyalty": 5,
  "nps": 50,
  "n_communications": 4,
  "total_sales": 1000,
  "unique_products": 20,
  "number_of_invoices": 10,
  "purchase_days": 8,
  "average_invoice_value": 100,
  "average_quantity_per_invoice": 5,
  "average_product_price": 20,
  "customer_activity_days": 365
}
```

The API returns:

* Predicted class
* Prediction label
* Response probability
* Non-response probability

The probability values returned by the API are generated by the trained model.

HEALTH CHECK

The `/health` endpoint can be used to confirm that the API has successfully loaded the trained model.

An example response is:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "RandomForestClassifier",
  "features_expected": 12
}
```

MODEL INFORMATION

The `/model-info` endpoint provides information about the deployed model.

The deployed Random Forest model expects 12 features.

The feature names and order are:

* n_comp
* loyalty
* nps
* n_communications
* total_sales
* unique_products
* number_of_invoices
* purchase_days
* average_invoice_value
* average_quantity_per_invoice
* average_product_price
* customer_activity_days

DASH APPLICATION

The project contains an interactive Dash application in:

deployment/app.py

The Dash application provides:

* CSV customer data upload
* Automatic feature validation
* Customer activity feature name mapping
* Missing NPS value handling
* Batch campaign response predictions
* Individual customer predictions
* Response probabilities
* Prediction result visualisation

RUNNING THE DASH APPLICATION

From the project root directory, run:

python deployment/app.py

The application will be available at:

http://127.0.0.1:8050/

CSV UPLOAD

The Dash application accepts the master customer campaign CSV file.

The application checks that the required model features are available.

The source dataset uses:

customer_lifetime_days

The Dash application automatically converts this to:

customer_activity_days

The application also identifies missing NPS values and applies the median NPS value when required.

For the current master dataset, three missing NPS values are replaced using the median NPS value of 4.0.

The application then creates the 12 model features in the correct order before generating predictions.

DEPLOYMENT DESIGN

The deployment uses project-relative paths rather than machine-specific Windows paths.

The model is loaded relative to the project directory.

This allows the project to be moved to another compatible environment without changing the model path in the application code.

The FastAPI service uses the model located in:

models/random_forest_model.pkl

The Dash service uses the same trained Random Forest model.

REQUIREMENTS

The main Python dependencies are listed in requirements.txt.

The project requires packages including:

* FastAPI
* Uvicorn
* Pydantic
* Pandas
* NumPy
* Joblib
* Scikit-learn version 1.6.1
* XGBoost
* SHAP
* Dash
* Plotly
* Matplotlib

REPRODUCIBILITY

The project includes a requirements.txt file containing the Python dependencies required to reproduce the analysis and run the deployed applications.

The project is designed to avoid dependencies on machine-specific file paths.

A dedicated Python environment is recommended.

For example:

conda create -n retail_campaign python=3.11

conda activate retail_campaign

Install dependencies:

pip install -r requirements.txt

The saved models are included in the models directory.

The evaluation outputs are included in the outputs directory.

REPRODUCING THE ANALYSIS

Start Jupyter Notebook from the project root:

jupyter notebook

Open:

notebooks/AIML_Retail_Campaign_Project.ipynb

Run the notebook cells sequentially to reproduce the data preparation, feature engineering, exploratory analysis, machine learning workflow and evaluation.

OUTPUTS AND ARTEFACTS

The project includes saved model files:

models/logistic_regression_model.pkl

models/random_forest_model.pkl

models/xgboost_model.pkl

Evaluation outputs include:

outputs/evaluation/classification_report.csv

outputs/evaluation/confusion_matrix.csv

outputs/evaluation/evaluation_results.txt

outputs/evaluation/feature_importance.csv

outputs/evaluation/metrics.csv

outputs/evaluation/test_predictions.csv

Visualisation outputs include:

outputs/plots/confusion_matrix.png

outputs/plots/feature_importance.png

outputs/plots/roc_curve.png

outputs/plots/shap_summary.png

DEPLOYMENT CONSIDERATIONS

The current implementation is designed for local deployment and demonstration.

The FastAPI service:

* Loads the trained model from the project models directory
* Validates prediction inputs using Pydantic
* Uses pandas DataFrames with the correct feature names
* Provides health and model information endpoints
* Returns the predicted class and prediction probabilities

For production deployment, additional configuration would be required, such as:

* Production application server configuration
* Environment-specific configuration
* Authentication and authorisation
* HTTPS
* Logging and monitoring
* Containerisation
* Cloud hosting
* Secure model and configuration management

TROUBLESHOOTING

Scikit-learn Version Warning

The saved model was trained using scikit-learn 1.6.1.

If a different version is used, scikit-learn may display an InconsistentVersionWarning when loading the model.

Install the matching version with:

pip install scikit-learn==1.6.1

ModuleNotFoundError

If Python cannot find the `api` package, make sure the terminal is located in the project root directory.

The project root is the directory containing:

* api
* deployment
* models
* notebooks
* requirements.txt
* README.md

Then run:

python -m uvicorn api.main:app --reload

Model Feature Error

The deployed Random Forest model expects 12 features.

The feature names and order must remain:

* n_comp
* loyalty
* nps
* n_communications
* total_sales
* unique_products
* number_of_invoices
* purchase_days
* average_invoice_value
* average_quantity_per_invoice
* average_product_price
* customer_activity_days

The API and Dash application construct pandas DataFrames using these exact feature names before passing data to the model.

POSSIBLE MODEL ENHANCEMENTS

Future model enhancements could investigate additional customer information where it is relevant, lawful, transparently collected and appropriate for the modelling objective.

Potential future variables could include broad customer preference information or preferred communication channels where customers have voluntarily provided that information.

Additional behavioural variables could also be investigated, such as changes in purchasing frequency, changes in average transaction value, campaign engagement history and customer purchasing trends over time.

Any additional customer information should be collected transparently and handled according to applicable data protection requirements.

Future modelling work should also assess potential bias, fairness, feature leakage and whether each proposed feature would be available at the time a campaign prediction is made.

Model performance should be re-evaluated after adding any new features to determine whether they provide meaningful predictive value.

AUTHOR

Emmanuel Ogbonna

AI and Machine Learning Data Science Institute Project
