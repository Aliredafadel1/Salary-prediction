import streamlit as st 
import requests
import pandas as pd
df = pd.read_csv("data/raw/ds_salaries.csv")

st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Enter job and company details to predict salary in USD.")
work_year = st.number_input("Work Year", min_value=2020, max_value=2035, value=2024)
experience_level = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
employment_type = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
job_title =st.selectbox("Job Title",sorted(df["job_title"].unique()))
employee_residence = st.text_input("Employee Residence", "US")
remote_ratio = st.slider("Remote Ratio", 0, 100, 100)
company_location = st.text_input("Company Location", "US")
company_size = st.selectbox("Company Size", ["S", "M", "L"])
if st.button("Predict Salary"):
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
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()

        if "predicted_salary" in result:
            st.success("Prediction completed successfully.")
            st.metric("Predicted Salary (USD)", f"${result['predicted_salary']:,.2f}")
        else:
            st.error(f"API Error: {result}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Please ensure the FastAPI server is running.")
    except Exception as e:
        st.error(f"Connection error: {e}")

