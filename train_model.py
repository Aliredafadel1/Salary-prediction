import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Salary Dashboard", layout="wide")

st.title(" Salary Prediction & Analysis Dashboard")
st.markdown("Understand **who earns what and why** through data storytelling.")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ds_salaries.csv")

df = load_data()

# ----------------------------
# Overview
# ----------------------------
st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", len(df))
col2.metric("Avg Salary", f"${df['salary_in_usd'].mean():,.0f}")
col3.metric("Max Salary", f"${df['salary_in_usd'].max():,.0f}")

st.dataframe(df.head())

# ----------------------------
# Salary Distribution
# ----------------------------
st.header("📈 Salary Distribution")

fig, ax = plt.subplots()
ax.hist(df["salary_in_usd"], bins=40)
ax.set_title("Salary Distribution")
ax.set_xlabel("Salary (USD)")
ax.set_ylabel("Frequency")
st.pyplot(fig)

st.markdown(" Most salaries are concentrated in a specific range, with a few very high outliers.")

# ----------------------------
# Salary by Experience Level
# ----------------------------
st.header(" Salary by Experience Level")

exp_salary = df.groupby("experience_level")["salary_in_usd"].mean().sort_values()

fig, ax = plt.subplots()
exp_salary.plot(kind="bar", ax=ax)
ax.set_title("Average Salary by Experience")
st.pyplot(fig)

st.markdown("👉 Senior and Expert roles clearly earn more than Entry-level.")

# ----------------------------
# Salary by Company Size
# ----------------------------
st.header("🏢 Salary by Company Size")

size_salary = df.groupby("company_size")["salary_in_usd"].mean()

fig, ax = plt.subplots()
size_salary.plot(kind="bar", ax=ax)
ax.set_title("Salary by Company Size")
st.pyplot(fig)

st.markdown("👉 Larger companies tend to offer higher salaries.")

# ----------------------------
# Remote Work Impact
# ----------------------------
st.header(" Remote Work vs Salary")

remote_salary = df.groupby("remote_ratio")["salary_in_usd"].mean()

fig, ax = plt.subplots()
remote_salary.plot(kind="line", ax=ax)
ax.set_title("Salary vs Remote Ratio")
st.pyplot(fig)

st.markdown("👉 Remote work can influence salary depending on company and location.")

# ----------------------------
# Top Job Titles
# ----------------------------
st.header("💻 Top Paying Job Titles")

top_jobs = (
    df.groupby("job_title")["salary_in_usd"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))
top_jobs.plot(kind="bar", ax=ax)
ax.set_title("Top 10 Job Titles by Salary")
st.pyplot(fig)

st.markdown("👉 Specialized roles in AI and data tend to dominate the top salaries.")

# ----------------------------
# Correlation Heatmap
# ----------------------------
st.header("🔥 Correlation Heatmap")

numeric_df = df.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()

fig, ax = plt.subplots()
cax = ax.matshow(corr)
fig.colorbar(cax)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))

ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)

st.pyplot(fig)

st.markdown("👉 This shows how numerical features relate to salary.")

# ----------------------------
# Filter (Interactive EDA)
# ----------------------------
st.header("🎯 Interactive Analysis")

exp_filter = st.multiselect(
    "Select Experience Level",
    options=df["experience_level"].unique(),
    default=df["experience_level"].unique()
)

filtered_df = df[df["experience_level"].isin(exp_filter)]

fig, ax = plt.subplots()
ax.hist(filtered_df["salary_in_usd"], bins=30)
ax.set_title("Filtered Salary Distribution")
st.pyplot(fig)

# ----------------------------
# Prediction Section (Optional)
# ----------------------------
st.header("🤖 Predict Salary (Connected to FastAPI)")

import requests

with st.form("predict_form"):
    work_year = st.number_input("Work Year", value=2024)
    experience_level = st.selectbox("Experience", ["EN", "MI", "SE", "EX"])
    employment_type = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
    job_title = st.text_input("Job Title", "Data Scientist")
    employee_residence = st.text_input("Residence", "US")
    remote_ratio = st.slider("Remote Ratio", 0, 100, 50)
    company_location = st.text_input("Company Location", "US")
    company_size = st.selectbox("Company Size", ["S", "M", "L"])

    submit = st.form_submit_button("Predict")

if submit:
    payload = {
        "work_year": work_year,
        "experience_level": experience_level,
        "employment_type": employment_type,
        "job_title": job_title,
        "employee_residence": employee_residence,
        "remote_ratio": remote_ratio,
        "company_location": company_location,
        "company_size": company_size
    }

    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = res.json()

        st.success(f"💰 Predicted Salary: ${result['predicted_salary']:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")