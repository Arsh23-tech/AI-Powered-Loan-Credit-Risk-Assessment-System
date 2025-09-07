import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import json

# Load trained model
model = joblib.load("models/best_lgbm_model.pkl")

# Load SHAP explainer (precomputed)
explainer = joblib.load("models/explainer.pkl")

# Load feature list
with open("models/features.json") as f:
    feature_list = json.load(f)

# ---- Streamlit App Layout ----
st.set_page_config(page_title="AI Credit Risk Assessment", layout="centered")
st.title("📊 AI-Powered Credit Risk Assessment Dashboard")

st.markdown("Enter applicant details below to assess risk score and view explanation.")

# ---- Applicant Input Form ----
st.sidebar.header("Applicant Information")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
limit_bal = st.sidebar.number_input("Credit Limit", min_value=10000, max_value=1000000, step=10000, value=50000)
avg_pay_amt = st.sidebar.number_input("Average Monthly Payment", min_value=0, max_value=100000, value=10000)
max_bill_amt = st.sidebar.number_input("Max Bill Amount", min_value=0, max_value=1000000, value=60000)
education = st.sidebar.selectbox("Education Level", ["Graduate", "University", "High School"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Others"])

# ---- Map inputs to DataFrame ----
# Initialize full input with zeros
user_data = pd.DataFrame(np.zeros((1, len(feature_list))), columns=feature_list)

# Fill in selected features
user_data["AGE"] = age
user_data["LIMIT_BAL"] = limit_bal
user_data["avg_pay_amt"] = avg_pay_amt
user_data["max_bill_amt"] = max_bill_amt
user_data["EDUCATION"] = 1 if education == "Graduate" else 2 if education == "University" else 3
user_data["MARRIAGE"] = 1 if marital_status == "Married" else 2 if marital_status == "Single" else 3

# ---- Prediction ----
prob_default = model.predict_proba(user_data)[0][1]
risk_score = int((1 - prob_default) * 1000)

if risk_score >= 700:
    risk_level = "Low"
elif risk_score >= 500:
    risk_level = "Medium"
else:
    risk_level = "High"

# ---- Prewritten AI Summaries ----
ai_summaries = {
    "Low": "The applicant is assessed as low credit risk, with timely payments and stable financial behavior. Their high monthly payments and responsible credit use support a strong credit profile.",
    "Medium": "The applicant is assessed to have moderate credit risk. While they do not exhibit strong financial distress, there are some inconsistencies in repayments or moderate credit usage that warrant attention.",
    "High": "The applicant is identified as high credit risk. Indicators such as delayed payments, low repayments relative to bills, or inconsistent credit behavior have significantly raised their default risk."
}

# ---- Display Outputs ----
st.subheader("📉 Risk Assessment Results")
st.metric("Risk Score", f"{risk_score}/1000")
if risk_level == "Low":
    st.success(f"Risk Level: {risk_level}")
elif risk_level == "Medium":
    st.warning(f"Risk Level: {risk_level}")
else:
    st.error(f"Risk Level: {risk_level}")

st.markdown("---")
st.markdown("### 🧠 AI-Generated Risk Summary")
st.info(ai_summaries[risk_level])

# ---- SHAP Explanation ----
st.markdown("---")
st.markdown("### 🔍 Top Feature Contributions")
shap_values = explainer(user_data)
shap.plots.bar(shap_values, max_display=5, show=False)
st.pyplot(plt.gcf())
plt.clf()
