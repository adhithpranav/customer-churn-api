from fastapi import FastAPI


from app.schema import CustomerData
from app.inference import predict_customer

app = FastAPI()









@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API"
    }

@app.post("/predict")
def predict(customer: CustomerData):
    return predict_customer(customer)





  