import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="💼 Salary Prediction App",
    page_icon="💰",
    layout="wide"
)

st.title("💼 Salary Prediction App")
st.caption("Powered by Machine Learning (Random Forest Model)")

# ==================================================
# LOAD MODEL BUNDLE
# ==================================================
@st.cache_resource
def load_model():
    bundle = joblib.load("salary_model.pkl")
    return bundle["best_model"], bundle["scalar"], bundle["feature_columns"]

model, scaler, feature_columns = load_model()

# ==================================================
# LOAD DATA (ONLY FOR DROPDOWNS)
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv("salaries.csv")

df = load_data()

# ==================================================
# SIDEBAR INPUTS
# ==================================================
st.sidebar.header("🔧 Enter Employee Details")

work_year = st.sidebar.selectbox(
    "Work Year", sorted(df["work_year"].unique())
)

experience_level = st.sidebar.selectbox(
    "Experience Level", sorted(df["experience_level"].unique())
)

employment_type = st.sidebar.selectbox(
    "Employment Type", sorted(df["employment_type"].unique())
)

job_title = st.sidebar.selectbox(
    "Job Title", sorted(df["job_title"].unique())
)

salary = st.sidebar.number_input(
    "Base Salary (Original Currency)",
    min_value=int(df["salary"].min()),
    max_value=int(df["salary"].max()),
    value=int(df["salary"].median())
)

salary_currency = st.sidebar.selectbox(
    "Salary Currency", sorted(df["salary_currency"].unique())
)

employee_residence = st.sidebar.selectbox(
    "Employee Residence", sorted(df["employee_residence"].unique())
)

remote_ratio = st.sidebar.selectbox(
    "Remote Ratio (%)", sorted(df["remote_ratio"].unique())
)

company_location = st.sidebar.selectbox(
    "Company Location", sorted(df["company_location"].unique())
)

company_size = st.sidebar.selectbox(
    "Company Size", sorted(df["company_size"].unique())
)

# ==================================================
# PREDICTION
# ==================================================
if st.sidebar.button("🚀 Predict Salary"):

    input_df = pd.DataFrame([{
        "work_year": work_year,
        "experience_level": experience_level,
        "employment_type": employment_type,
        "job_title": job_title,
        "salary": salary,
        "salary_currency": salary_currency,
        "employee_residence": employee_residence,
        "remote_ratio": remote_ratio,
        "company_location": company_location,
        "company_size": company_size
    }])

    # 🔴 VERY IMPORTANT
    # Apply SAME encoding logic as training
    from sklearn.preprocessing import LabelEncoder

    categorical_cols = [
        'experience_level', 'employment_type', 'job_title',
        'salary_currency', 'employee_residence',
        'company_location', 'company_size'
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df[col])
        input_df[col] = le.transform(input_df[col])

    # Align columns
    input_df = input_df[feature_columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"💰 Predicted Salary (USD): ${prediction:,.2f}")

    st.info("Model: RandomForestRegressor (Best Model)")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption("Created with ❤️ using Streamlit & Scikit-Learn")

