# Customer Churn & Retention Intelligence

An end-to-end machine learning application that predicts customer churn, estimates churn risk, and explains individual predictions using SHAP. The application exposes predictions through FastAPI and provides an interactive Streamlit dashboard, with Docker Compose for containerized execution.

## Project Overview

Customer churn occurs when customers stop using a company's products or services. Identifying customers who may leave can help businesses investigate potential issues and make better-informed retention decisions.

This project demonstrates a complete machine learning workflow, from data preparation and model training to prediction serving, model explainability, testing, and deployment preparation.

### Key Features

- Customer churn prediction using machine learning.
- Feature engineering based on customer tenure, spending, and support calls.
- Comparison of Logistic Regression and Random Forest models.
- SHAP explanations showing features that contribute to individual predictions.
- FastAPI endpoints for health checks and predictions.
- Streamlit dashboard for interactive predictions.
- Docker Compose setup for running the API and dashboard.
- Automated tests for feature engineering and API behavior.

## System Architecture

```text
Customer Input
      |
      v
Streamlit Dashboard
      |
      v
FastAPI Prediction API
      |
      v
Feature Engineering
      |
      v
Preprocessing Pipeline
      |
      v
Trained ML Model
      |
      +------> Churn Prediction
      |
      +------> Churn Probability
      |
      +------> SHAP Explanations
      |
      v
Results Displayed on Dashboard
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn |
| Model explainability | SHAP |
| API | FastAPI |
| Dashboard | Streamlit |
| Model serialization | joblib |
| Automated testing | pytest, HTTPX |
| Containerization | Docker, Docker Compose |

## Project Structure

```text
customer-churn-ml/
├── app/
│   ├── main.py
│   ├── feature_engineering.py
│   └── dashboard.py
├── data/
│   └── customer_churn_dataset-testing-master.csv
├── models/
│   └── churn_pipeline.joblib
├── notebooks/
│   ├── base.ipynb
│   └── feature_engineering.py
├── tests/
│   ├── test_api.py
│   └── test_feature_engineering.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

The saved model pipeline must be available at `models/churn_pipeline.joblib` for the API to load successfully. The dataset is used for development and analysis; it is not required for prediction serving when the trained pipeline is already present.

## Machine Learning Workflow

1. Load and inspect the customer churn dataset.
2. Check for missing values, duplicate records, and unusual patterns.
3. Engineer additional features:
   - `AverageSpendPerTenure`
   - `SupportCallsPerTenure`
4. Separate features and the churn target.
5. Split the data into training and testing sets.
6. Preprocess numerical and categorical features.
7. Train and compare Logistic Regression and Random Forest.
8. Evaluate the models using accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.
9. Explain individual predictions using SHAP.
10. Serve the trained pipeline through FastAPI and display results through Streamlit.

## Run the Project with Docker

### Prerequisites

- Docker Desktop installed and running.
- Git, if cloning the repository.
- The trained model file in the expected `models/` location.

### Start the application

Clone the repository or open the project directory, then run:

```bash
docker compose up --build -d
```

Check the container status:

```bash
docker compose ps
```

View application logs:

```bash
docker compose logs -f
```

Stop the application:

```bash
docker compose down
```

### Access the application

- **Streamlit dashboard:** https://customer-churn-dashboard-sxx9.onrender.com/
- **FastAPI API:** https://customer-churn-dashboard-sxx9.onrender.com/
- **Interactive API documentation:** https://customer-churn-dashboard-sxx9.onrender.com/docs
- **Health check:** https://customer-churn-dashboard-sxx9.onrender.com/health

The dashboard communicates with the API using the `API_URL` environment variable configured in Docker Compose.

## API Usage

### Health Check

```http
GET /health
```

Use this endpoint to check whether the API is healthy and the model is loaded.

### Predict Customer Churn

```http
POST /predict
Content-Type: application/json
```

Example request:

```json
{
  "age": 30,
  "gender": "Male",
  "tenure": 12,
  "usage_frequency": 15,
  "support_calls": 3,
  "payment_delay": 5,
  "subscription_type": "Standard",
  "contract_length": "Annual",
  "total_spend": 600.0,
  "last_interaction": 10
}
```

The API returns the predicted churn class, churn probability, risk level, and a list of the strongest SHAP contributions for the individual prediction.

Use `/docs` to inspect the actual response schema and try requests interactively.

## Run Automated Tests

Run tests locally from the project root:

```bash
python -m pytest -v
```

Run tests inside the API Docker image:

```bash
docker compose run --rm api pytest -v
```

The tests cover feature engineering calculations, input validation, API health, and prediction response behavior.

## Model Evaluation

The project compares Logistic Regression and Random Forest using a held-out test set and cross-validation.

The Random Forest achieved extremely high evaluation scores on the current dataset. These results should be treated cautiously until potential dataset artifacts, target leakage, and generalization to independent data have been investigated.

Reported performance on this dataset should not be interpreted as a guarantee of real-world prediction accuracy.

## Limitations and Future Improvements

- Validate performance using an independent, representative dataset.
- Investigate potential patterns related to customer identifiers and other dataset artifacts.
- Evaluate fairness and the necessity of sensitive or potentially sensitive features.
- Calibrate predicted probabilities and validate risk thresholds.
- Add more robust logging, monitoring, and error handling.
- Explore actionable customer retention recommendations.
- Deploy the API and dashboard to a public hosting platform.

## Disclaimer

This project is an educational machine learning application. Its predictions are not independently validated for production business decisions. SHAP values explain model behavior; they do not establish causal relationships.

## Author

Developed as an end-to-end machine learning and application deployment project.
