# 🚀 Customer Churn Prediction API

A production-style MLOps REST API for predicting telecom customer churn using **FastAPI**, **Scikit-Learn Pipeline**, **Docker**, and **Docker Compose**.

This project demonstrates how a trained machine learning model can be packaged and served as a production-ready inference service following backend engineering and MLOps best practices.

---

## ✨ Features

- Production-ready Machine Learning Pipeline
- FastAPI REST API
- Automatic request validation using Pydantic
- Scikit-Learn Pipeline with:
  - ColumnTransformer
  - OneHotEncoder
  - RandomForestClassifier
- Dockerized application
- Docker Compose support
- Interactive Swagger API Documentation
- Clean project structure with separated inference logic

---

## 🏗 Architecture

```text
                Client
                   │
                   ▼
            POST /predict
                   │
                   ▼
               FastAPI
                   │
                   ▼
           Pydantic Validation
                   │
                   ▼
           CustomerData Object
                   │
                   ▼
          Pandas DataFrame
                   │
                   ▼
          Scikit-Learn Pipeline
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
ColumnTransformer       RandomForestClassifier
      │
      ▼
OneHotEncoder
      │
      ▼
 Prediction (Churn / No Churn)
      │
      ▼
          JSON Response
```

---

## 🛠 Tech Stack

- Python
- FastAPI
- Scikit-Learn
- Pandas
- Joblib
- Pydantic
- Docker
- Docker Compose

---

## 📁 Project Structure

```text
ml-fastapi-serving/
│
├── app/
│   ├── inference.py
│   ├── main.py
│   ├── schema.py
│   └── __init__.py
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── model/
│   └── churn_pipeline.pkl
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train.py
├── README.md
└── .gitignore
```

---

## 🚀 Running the Project

### Using Docker Compose

```bash
docker compose up -d
```

Open:

```
http://localhost:8000/docs
```

---

### Without Docker

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoint

### POST `/predict`

Example Request

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1074.0
}
```

Example Response

```json
{
  "prediction": "Churn"
}
```

---

## 🎯 Learning Objectives

This project was built to understand and demonstrate:

- Building production-ready Machine Learning Pipelines
- Serving ML models using FastAPI
- Request validation with Pydantic
- Containerization using Docker
- Multi-container application management with Docker Compose
- Production-style project organization
- MLOps fundamentals from a Platform Engineering perspective

---

## 🚀 Future Improvements

- NGINX Reverse Proxy
- AWS EC2 Deployment
- GitHub Actions CI/CD
- Kubernetes Deployment
- Prediction probability (`predict_proba`)
- Model versioning with MLflow

---

## 👨‍💻 Author

**Adhith Pranav**

GitHub: https://github.com/adhithpranav
