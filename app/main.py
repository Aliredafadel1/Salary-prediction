from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("salary_prediction_model.pkl")

@app.get("/")
def home():
    return {"message": "Salary Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]

        return {"predicted_salary": float(prediction)}

    except Exception as e:
        return {"error": str(e)}