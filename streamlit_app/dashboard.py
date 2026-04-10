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
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    page_icon="💼",
    layout="wide"
)

# --------------------------------
# Dark Theme + Dropdown FIX
# --------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #050816;
        color: #f9fafb;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1120;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #f9fafb;
    }

    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
    }

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }

    /* Selectbox main */
    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }

    /* Dropdown popup */
    div[data-baseweb="popover"] {
        background-color: #111827 !important;
    }

    ul {
        background-color: #111827 !important;
    }

    li {
        background-color: #111827 !important;
        color: #ffffff !important;
    }

    li:hover {
        background-color: #1f2937 !important;
    }

    /* Fallback fix */
    [role="listbox"] {
        background-color: #111827 !important;
    }

    [role="option"] {
        background-color: #111827 !important;
        color: white !important;
    }

    [role="option"]:hover {
        background-color: #1f2937 !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Header
# --------------------------------
st.title("Salary Prediction Dashboard")
st.write("A local dashboard for salary insights, prediction, and AI explanations.")

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.header("About")
st.sidebar.write(
    "Explore salary trends, predict salaries, and get AI explanations."
)

if st.sidebar.button("Refresh Data"):
    st.rerun()

# --------------------------------
# Load API data
# --------------------------------
insights = {}
recent_logs = []

try:
    insights = requests.get(INSIGHTS_URL, timeout=10).json()
except:
    st.warning("Could not load insights")

try:
    logs = requests.get(RECENT_LOGS_URL, timeout=10).json()
    recent_logs = logs.get("logs", [])
except:
    st.info("No logs available")

# --------------------------------
# Metrics
# --------------------------------
if insights:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", insights.get("rows", 0))
    c2.metric("Average", f"${insights.get('avg_salary', 0):,.0f}")
    c3.metric("Median", f"${insights.get('median_salary', 0):,.0f}")
    c4.metric("Max", f"${insights.get('max_salary', 0):,.0f}")

    st.markdown("### Overview")
    st.write("Salary patterns based on experience, company size, remote work, and roles.")

# --------------------------------
# Charts
# --------------------------------
if insights:
    col1, col2 = st.columns(2)

    with col1:
        exp = insights.get("salary_by_experience", {})
        if exp:
            df = pd.DataFrame({"level": exp.keys(), "salary": exp.values()})
            fig, ax = plt.subplots()
            ax.bar(df["level"], df["salary"])
            ax.set_title("Experience vs Salary")
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)
            plt.close()

    with col2:
        size = insights.get("salary_by_company_size", {})
        if size:
            df = pd.DataFrame({"size": size.keys(), "salary": size.values()})
            fig, ax = plt.subplots()
            ax.bar(df["size"], df["salary"])
            ax.set_title("Company Size vs Salary")
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)
            plt.close()

    remote = insights.get("salary_by_remote_ratio", {})
    if remote:
        df = pd.DataFrame({"remote": remote.keys(), "salary": remote.values()}).sort_values("remote")
        fig, ax = plt.subplots()
        ax.plot(df["remote"], df["salary"], marker="o")
        ax.set_title("Remote Ratio vs Salary")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

# --------------------------------
# Prediction Form
# --------------------------------
st.markdown("---")
st.header("Predict Salary")

col1, col2 = st.columns(2)

with col1:
    work_year = st.number_input("Work Year", 2020, 2026, 2024)

    exp_map = {"Entry": "EN", "Mid": "MI", "Senior": "SE", "Executive": "EX"}
    exp_label = st.selectbox("Experience", list(exp_map.keys()))
    experience = exp_map[exp_label]

    emp_map = {"Full Time": "FT", "Part Time": "PT", "Contract": "CT", "Freelance": "FL"}
    emp_label = st.selectbox("Employment", list(emp_map.keys()))
    employment = emp_map[emp_label]

    job = st.text_input("Job Title", "Data Scientist")

with col2:
    residence = st.text_input("Residence", "US").strip().upper()
    remote = st.slider("Remote %", 0, 100, 100, step=25)
    location = st.text_input("Company Location", "US").strip().upper()
    size = st.selectbox("Company Size", ["S", "M", "L"])

payload = {
    "work_year": work_year,
    "experience_level": experience,
    "employment_type": employment,
    "job_title": job.strip() if job.strip() else "Data Scientist",
    "employee_residence": residence,
    "remote_ratio": remote,
    "company_location": location,
    "company_size": size
}

b1, b2 = st.columns(2)

with b1:
    if st.button("Predict"):
        try:
            with st.spinner("Predicting..."):
                res = requests.post(PREDICT_URL, json=payload)
            data = res.json()
            st.success(f"Salary: ${data['predicted_salary']:,.2f}")
        except:
            st.error("API error")

with b2:
    if st.button("Predict + AI"):
        try:
            with st.spinner("Thinking..."):
                res = requests.post(PREDICT_WITH_AI_URL, json=payload)
            data = res.json()
            st.success(f"Salary: ${data['predicted_salary']:,.2f}")
            st.write(data.get("ai_explanation", ""))
        except:
            st.error("AI error")

# --------------------------------
# AI Assistant
# --------------------------------
st.markdown("---")
st.header("AI Assistant")

prompt = st.text_area("Ask something:", "Why do seniors earn more?")

if st.button("Ask"):
    try:
        with st.spinner("Generating..."):
            res = requests.post(AI_URL, json={"prompt": prompt})
        st.write(res.json().get("response", "No response"))
    except:
        st.error("Connection error")

# --------------------------------
# Logs
# --------------------------------
st.markdown("---")
st.header("Recent Predictions")

if recent_logs:
    df = pd.DataFrame(recent_logs)
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No logs yet")