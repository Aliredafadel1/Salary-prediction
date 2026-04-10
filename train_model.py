import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "ds_salaries.csv"
MODEL_PATH = BASE_DIR / "salary_prediction_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")


# -----------------------------
# Features and target
# -----------------------------
feature_columns = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "company_size",
]

target_column = "salary_in_usd"

# keep only needed columns
df = df[feature_columns + [target_column]].copy()

# drop rows with missing target
df = df.dropna(subset=[target_column])

# clean text columns a little
text_columns = [
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "company_location",
    "company_size",
]

for col in text_columns:
    df[col] = df[col].astype(str).str.strip()

# normalize country codes
df["employee_residence"] = df["employee_residence"].str.upper()
df["company_location"] = df["company_location"].str.upper()

X = df[feature_columns]
y = df[target_column]


# -----------------------------
# Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


# -----------------------------
# Column groups
# -----------------------------
numeric_features = ["work_year", "remote_ratio"]

categorical_features = [
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "company_location",
    "company_size",
]


# -----------------------------
# Preprocessing
# -----------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# -----------------------------
# Model pipeline
# -----------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# -----------------------------
# Train model
# -----------------------------
model.fit(X_train, y_train)

print("Model training completed.")


# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

metrics = {
    "model_name": "RandomForestRegressor",
    "target": target_column,
    "features": feature_columns,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "r2": round(float(r2), 4),
    "mae": round(float(mae), 2),
    "rmse": round(float(rmse), 2),
}

print("\nModel Evaluation:")
print(f"R2   : {metrics['r2']}")
print(f"MAE  : {metrics['mae']}")
print(f"RMSE : {metrics['rmse']}")


# -----------------------------
# Quick sanity check
# -----------------------------
sample_1 = pd.DataFrame(
    [
        {
            "work_year": 2024,
            "experience_level": "EN",
            "employment_type": "FT",
            "job_title": "Data Analyst",
            "employee_residence": "US",
            "remote_ratio": 0,
            "company_location": "US",
            "company_size": "S",
        }
    ]
)

sample_2 = pd.DataFrame(
    [
        {
            "work_year": 2024,
            "experience_level": "SE",
            "employment_type": "FT",
            "job_title": "Machine Learning Engineer",
            "employee_residence": "US",
            "remote_ratio": 100,
            "company_location": "US",
            "company_size": "L",
        }
    ]
)

pred_1 = float(model.predict(sample_1)[0])
pred_2 = float(model.predict(sample_2)[0])

print("\nSanity Check Predictions:")
print(f"Sample 1 prediction: ${pred_1:,.2f}")
print(f"Sample 2 prediction: ${pred_2:,.2f}")


# -----------------------------
# Save model and metrics
# -----------------------------
joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Metrics saved to: {METRICS_PATH}")