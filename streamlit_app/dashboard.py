
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# --------------------------------
# API URLs
# --------------------------------
API_BASE = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_BASE}/predict"
PREDICT_WITH_AI_URL = f"{API_BASE}/predict-with-ai"
INSIGHTS_URL = f"{API_BASE}/insights"
RECENT_LOGS_URL = f"{API_BASE}/recent_logs"
AI_URL = f"{API_BASE}/ai-explain"

# --------------------------------
# Page config
# --------------------------------
st.set_page_config(page_title="Salary Prediction Dashboard", layout="wide")

st.title("💼 Salary Prediction Dashboard")
st.write("A local dashboard for salary insights, storytelling, prediction, and AI explanations.")

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.header("About")
st.sidebar.write(
    "This dashboard explores salary patterns in data jobs, predicts salary "
    "based on job and company details, and uses Ollama Phi to explain predictions."
)

# --------------------------------
# Load API insights
# --------------------------------
insights = {}
recent_logs = []

try:
    insights_response = requests.get(INSIGHTS_URL, timeout=15)
    if insights_response.status_code == 200:
        insights = insights_response.json()
    else:
        st.warning(f"Could not load insights. Status code: {insights_response.status_code}")
except Exception as e:
    st.warning(f"Could not load insights from FastAPI: {e}")

try:
    logs_response = requests.get(RECENT_LOGS_URL, timeout=15)
    if logs_response.status_code == 200:
        logs_json = logs_response.json()
        recent_logs = logs_json.get("logs", [])
        if logs_json.get("warning"):
            st.info(f"Recent logs info: {logs_json['warning']}")
    else:
        st.info(f"Could not load recent logs. Status code: {logs_response.status_code}")
except Exception as e:
    st.info(f"Could not load recent logs: {e}")

# --------------------------------
# Top metrics
# --------------------------------
if insights and "error" not in insights:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", insights.get("rows", 0))
    c2.metric("Average Salary", f"${insights.get('avg_salary', 0):,.0f}")
    c3.metric("Median Salary", f"${insights.get('median_salary', 0):,.0f}")
    c4.metric("Max Salary", f"${insights.get('max_salary', 0):,.0f}")

    st.markdown("### Quick Story")
    st.write(
        "This dataset shows how salary changes with experience, company size, "
        "remote work, and job role. The dashboard helps explain who earns more, "
        "where the differences are, and what patterns stand out."
    )

# --------------------------------
# Charts
# --------------------------------
if insights and "error" not in insights:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Salary by Experience Level")
        exp_data = insights.get("salary_by_experience", {})
        if exp_data:
            exp_df = pd.DataFrame({
                "experience_level": list(exp_data.keys()),
                "avg_salary": list(exp_data.values())
            })

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(exp_df["experience_level"], exp_df["avg_salary"])
            ax.set_ylabel("Average Salary (USD)")
            ax.set_xlabel("Experience Level")
            st.pyplot(fig)

            st.caption(
                "Story: experience level usually pushes salary upward. "
                "Senior and executive roles tend to earn more than entry roles."
            )

    with col2:
        st.subheader("Average Salary by Company Size")
        size_data = insights.get("salary_by_company_size", {})
        if size_data:
            size_df = pd.DataFrame({
                "company_size": list(size_data.keys()),
                "avg_salary": list(size_data.values())
            })

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(size_df["company_size"], size_df["avg_salary"])
            ax.set_ylabel("Average Salary (USD)")
            ax.set_xlabel("Company Size")
            st.pyplot(fig)

            st.caption(
                "Story: company size can affect compensation, though the pattern may vary "
                "depending on role mix, country, and market demand."
            )

    st.subheader("Average Salary by Remote Ratio")
    remote_data = insights.get("salary_by_remote_ratio", {})
    if remote_data:
        remote_df = pd.DataFrame({
            "remote_ratio": list(remote_data.keys()),
            "avg_salary": list(remote_data.values())
        }).sort_values("remote_ratio")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(remote_df["remote_ratio"], remote_df["avg_salary"], marker="o")
        ax.set_xlabel("Remote Ratio")
        ax.set_ylabel("Average Salary (USD)")
        st.pyplot(fig)

        st.caption(
            "Story: remote work does not always guarantee higher salary, "
            "but it can reveal useful differences in the job market."
        )

    st.subheader("Top 10 Job Titles")
    titles_data = insights.get("top_job_titles", {})
    if titles_data:
        titles_df = pd.DataFrame({
            "job_title": list(titles_data.keys()),
            "count": list(titles_data.values())
        })

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(titles_df["job_title"], titles_df["count"])
        ax.set_xlabel("Count")
        ax.set_ylabel("Job Title")
        ax.invert_yaxis()
        st.pyplot(fig)

        st.caption(
            "Story: a few job titles dominate the dataset, which strongly shapes "
            "overall salary patterns and average values."
        )

# --------------------------------
# Prediction form
# --------------------------------
st.markdown("---")
st.header("🔮 Predict Salary")

col1, col2 = st.columns(2)

with col1:
    work_year = st.number_input("Work Year", min_value=2020, max_value=2025, value=2024)

    experience_map = {
        "Entry Level": "EN",
        "Mid Level": "MI",
        "Senior Level": "SE",
        "Executive Level": "EX"
    }
    experience_label = st.selectbox("Experience Level", list(experience_map.keys()))
    experience_level = experience_map[experience_label]

    employment_map = {
        "Full Time": "FT",
        "Part Time": "PT",
        "Contract": "CT",
        "Freelance": "FL"
    }
    employment_label = st.selectbox("Employment Type", list(employment_map.keys()))
    employment_type = employment_map[employment_label]

    job_title = st.text_input("Job Title", value="Data Scientist")

with col2:
    employee_residence = st.text_input("Employee Residence", value="US")
    remote_ratio = st.slider("Remote Ratio", 0, 100, 100)
    company_location = st.text_input("Company Location", value="US")
    company_size = st.selectbox("Company Size", ["S", "M", "L"])

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

btn1, btn2 = st.columns(2)

with btn1:
    if st.button("Predict Salary"):
        try:
            response = requests.post(PREDICT_URL, json=payload, timeout=20)

            if response.status_code == 200:
                result = response.json()

                if "predicted_salary" in result:
                    st.success(f"💰 Predicted Salary: ${result['predicted_salary']:,.2f}")

                if result.get("logged_to_supabase"):
                    st.info("Prediction was logged to Supabase successfully.")

                if "warning" in result:
                    st.warning(result["warning"])

                if "error" in result:
                    st.error(result["error"])
            else:
                st.error(f"API returned status code {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Please ensure the FastAPI server is running.")
        except requests.exceptions.Timeout:
            st.error("The API request timed out.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

with btn2:
    if st.button("Predict Salary + AI Explanation"):
        try:
            response = requests.post(PREDICT_WITH_AI_URL, json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()

                if "predicted_salary" in result:
                    st.success(f"💰 Predicted Salary: ${result['predicted_salary']:,.2f}")

                if "ai_explanation" in result:
                    st.info("🧠 AI Explanation")
                    st.write(result["ai_explanation"])

                if "error" in result:
                    st.error(result["error"])
            else:
                st.error(f"API returned status code {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Please ensure FastAPI and Ollama are running.")
        except requests.exceptions.Timeout:
            st.error("The AI request timed out.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# --------------------------------
# AI Assistant (Ollama)
# --------------------------------
st.markdown("---")
st.header("🧠 AI Assistant (Powered by Ollama Phi)")

user_prompt = st.text_area(
    "Ask anything about salaries, jobs, remote work, or your data:",
    "Explain why senior data scientists usually earn more than entry-level roles."
)

if st.button("Ask AI"):
    try:
        response = requests.post(
            AI_URL,
            json={"prompt": user_prompt},
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            st.success("AI Response")
            st.write(result.get("response", "No response returned."))
        else:
            st.error(f"API error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI or Ollama. Please make sure both are running.")
    except requests.exceptions.Timeout:
        st.error("The AI request timed out.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

# --------------------------------
# Recent predictions
# --------------------------------
st.markdown("---")
st.header("📜 Recent Predictions from Supabase")

if recent_logs:
    logs_df = pd.DataFrame(recent_logs)
    st.dataframe(logs_df, use_container_width=True)
else:
    st.info("No recent Supabase logs available yet.")
