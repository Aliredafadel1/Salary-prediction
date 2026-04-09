from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from supabase import create_client
import pandas as pd
import joblib
import os

# -----------------------------
# Load environment variables
# -----------------------------
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().strip('"').strip("'")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip().strip('"').strip("'")

print("SUPABASE_URL:", SUPABASE_URL)
print("SUPABASE_KEY loaded:", bool(SUPABASE_KEY))

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI(title="Salary Prediction API")

# -----------------------------
# Load model
# -----------------------------
model_path = Path(__file__).resolve().parent.parent / "salary_prediction_model.pkl"
model = joblib.load(model_path)

# -----------------------------
# Load dataset for insights
# -----------------------------
csv_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "ds_salaries.csv"
df = pd.read_csv(csv_path)

# -----------------------------
# Supabase client
# -----------------------------
supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client created successfully")
    except Exception as e:
        print("Supabase client creation failed:", str(e))


# -----------------------------
# Request schema
# -----------------------------
class PredictionInput(BaseModel):
    work_year: int
    experience_level: str
    employment_type: str
    job_title: str
    employee_residence: str
    remote_ratio: int
    company_location: str
    company_size: str


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/debug_env")
def debug_env():
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_key_loaded": bool(SUPABASE_KEY)
    }


@app.get("/insights")
def insights():
    try:
        summary = {
            "rows": int(len(df)),
            "avg_salary": float(df["salary_in_usd"].mean()),
            "median_salary": float(df["salary_in_usd"].median()),
            "max_salary": float(df["salary_in_usd"].max()),
            "min_salary": float(df["salary_in_usd"].min()),
            "top_job_titles": (
                df["job_title"]
                .value_counts()
                .head(10)
                .to_dict()
            ),
            "salary_by_experience": (
                df.groupby("experience_level")["salary_in_usd"]
                .mean()
                .round(2)
                .to_dict()
            ),
            "salary_by_company_size": (
                df.groupby("company_size")["salary_in_usd"]
                .mean()
                .round(2)
                .to_dict()
            ),
            "salary_by_remote_ratio": (
                df.groupby("remote_ratio")["salary_in_usd"]
                .mean()
                .round(2)
                .to_dict()
            )
        }
        return summary
    except Exception as e:
        return {"error": str(e)}


@app.get("/recent_logs")
def recent_logs():
    # Safe fallback if Supabase does not work
    if supabase_client is None:
        return {"logs": [], "warning": "Supabase is not connected"}

    try:
        result = (
            supabase_client
            .table("prediction_logs")
            .select("*")
            .order("id", desc=True)
            .limit(20)
            .execute()
        )
        return {"logs": result.data}
    except Exception as e:
        return {"logs": [], "warning": str(e)}


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        input_df = pd.DataFrame([{
            "work_year": data.work_year,
            "experience_level": data.experience_level,
            "employment_type": data.employment_type,
            "job_title": data.job_title,
            "employee_residence": data.employee_residence,
            "remote_ratio": data.remote_ratio,
            "company_location": data.company_location,
            "company_size": data.company_size
        }])

        prediction = float(model.predict(input_df)[0])

        log_data = {
            "work_year": data.work_year,
            "experience_level": data.experience_level,
            "employment_type": data.employment_type,
            "job_title": data.job_title,
            "employee_residence": data.employee_residence,
            "remote_ratio": data.remote_ratio,
            "company_location": data.company_location,
            "company_size": data.company_size,
            "predicted_salary": prediction
        }

        # Supabase logging is optional, not required for prediction success
        if supabase_client is not None:
            try:
                supabase_client.table("prediction_logs").insert(log_data).execute()
                return {
                    "predicted_salary": prediction,
                    "logged_to_supabase": True
                }
            except Exception as supabase_error:
                return {
                    "predicted_salary": prediction,
                    "logged_to_supabase": False,
                    "warning": f"Prediction worked but Supabase logging failed: {str(supabase_error)}"
                }

        return {
            "predicted_salary": prediction,
            "logged_to_supabase": False,
            "warning": "Prediction worked, but Supabase is not configured."
        }

    except Exception as e:
        return {"error": str(e)}