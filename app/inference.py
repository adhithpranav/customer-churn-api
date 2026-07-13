import joblib
import pandas as pd

pipeline = joblib.load("model/churn_pipeline.pkl")


def predict_customer(customer):
    customer_dict = customer.model_dump()

    data = pd.DataFrame([customer_dict])

    prediction = pipeline.predict(data)

    result = "Churn" if prediction[0] == 1 else "No Churn"

    return {
        "prediction": result
    }