
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from supabase import create_client
import pandas as pd
import joblib
import os
import requests

# -----------------------------
# Load environment variables
# -----------------------------
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI(title="Salary Prediction API with Ollama")

# -----------------------------
# Load ML model
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
# Request schemas
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


class AIRequest(BaseModel):
    prompt: str


# -----------------------------
# Ollama helper
# -----------------------------
def ask_ollama(prompt: str, model_name: str = "phi") -> str:
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response returned from Ollama.")
    except requests.exceptions.ConnectionError:
        return "Ollama is not running. Please start Ollama and run: ollama run phi"
    except Exception as e:
        return f"Ollama error: {str(e)}"


# -----------------------------
# Helper to build input dataframe
# -----------------------------
def build_input_df(data: PredictionInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "work_year": data.work_year,
        "experience_level": data.experience_level,
        "employment_type": data.employment_type,
        "job_title": data.job_title,
        "employee_residence": data.employee_residence,
        "remote_ratio": data.remote_ratio,
        "company_location": data.company_location,
        "company_size": data.company_size
    }])


# -----------------------------
# Helper to build log payload
# -----------------------------
def build_log_data(data: PredictionInput, prediction: float) -> dict:
    return {
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


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "API is running with Ollama support"}


@app.get("/debug_env")
def debug_env():
    return {
        "supabase_url_loaded": bool(SUPABASE_URL),
        "supabase_key_loaded": bool(SUPABASE_KEY)
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/insights")
def insights():
    try:
        summary = {
            "rows": int(len(df)),
            "avg_salary": float(df["salary_in_usd"].mean()),
            "median_salary": float(df["salary_in_usd"].median()),
            "max_salary": float(df["salary_in_usd"].max()),
            "min_salary": float(df["salary_in_usd"].min()),
            "top_job_titles": df["job_title"].value_counts().head(10).to_dict(),
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
        input_df = build_input_df(data)
        prediction = float(model.predict(input_df)[0])

        log_data = build_log_data(data, prediction)

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


@app.post("/ai-explain")
def ai_explain(request: AIRequest):
    answer = ask_ollama(request.prompt, "phi")
    return {"response": answer}


@app.post("/predict-with-ai")
def predict_with_ai(data: PredictionInput):
    try:
        input_df = build_input_df(data)
        prediction = float(model.predict(input_df)[0])

        log_data = build_log_data(data, prediction)

        logged_to_supabase = False
        warning_message = None

        if supabase_client is not None:
            try:
                supabase_client.table("prediction_logs").insert(log_data).execute()
                logged_to_supabase = True
            except Exception as supabase_error:
                warning_message = f"Prediction worked but Supabase logging failed: {str(supabase_error)}"
        else:
            warning_message = "Prediction worked, but Supabase is not configured."

        ai_prompt = f"""
You are an AI assistant inside a salary prediction dashboard.

A machine learning model predicted a salary of {prediction:.2f} USD for this profile.

Profile details:
- Work year: {data.work_year}
- Experience level: {data.experience_level}
- Employment type: {data.employment_type}
- Job title: {data.job_title}
- Employee residence: {data.employee_residence}
- Remote ratio: {data.remote_ratio}
- Company location: {data.company_location}
- Company size: {data.company_size}

Explain the prediction automatically in a simple, natural, and professional way.

Rules:
- Speak as if you are explaining the result directly to the dashboard user.
- Mention the most likely factors affecting the prediction such as experience, job title, company size, and remote ratio.
- Keep it clear and not too long.
- Do not say you are unsure unless there is an obvious issue.
- Do not repeat the raw profile in a boring list. Turn it into a smooth explanation.
"""

        ai_explanation = ask_ollama(ai_prompt, "phi")

        response = {
            "predicted_salary": prediction,
            "ai_explanation": ai_explanation,
            "logged_to_supabase": logged_to_supabase
        }

        if warning_message:
            response["warning"] = warning_message

        return response

    except Exception as e:
        return {"error": str(e)}
