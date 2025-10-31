# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("Loan Approval Prediction (Logistic Regression)")
st.write("Enter applicant details and click **Predict**")

# Load the trained pipeline
def load_model():
    model1 = joblib.load("model.pkl")
    return model1

model1 = load_model()

# Build a simple form for manual inputs
with st.form("loan_form"):
    st.header("Applicant Information")

    # Categorical inputs (use typical options from dataset)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    married = st.selectbox("Married", options=["Yes", "No"])
    dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", options=["Yes", "No"])

    # Numeric inputs
    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=3000.0, step=500.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=500.0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=120.0, step=5.0)
    loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0.0, value=360.0, step=12.0)

    credit_history = st.selectbox("Credit History", options=["1.0", "0.0"])  # as strings to map easily
    property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create single-row DataFrame matching original column names & dtypes
    input_dict = {
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [float(applicant_income)],
        "CoapplicantIncome": [float(coapplicant_income)],
        "LoanAmount": [float(loan_amount)],
        "Loan_Amount_Term": [float(loan_amount_term)],
        "Credit_History": [float(credit_history)],
        "Property_Area": [property_area]
    }

    input_df = pd.DataFrame(input_dict)

    # Model prediction
    proba = model1.predict_proba(input_df)[0,1]   # probability of class 1 (approved)
    pred = model1.predict(input_df)[0]           # 0 or 1

    st.subheader("Prediction Result")
    if pred == 'Y':
        st.success(f" Loan Approved — Probability: {proba*100:.2f}%")
    else:
        st.error(f" Loan Not Approved — Probability: {proba*100:.2f}%")

    # Optional: show model confidence & raw logits
    st.write("Details:")
    st.write(f"Predicted class: {pred}")
    st.write(f"Approval probability: {proba:.4f}")
    # Show input data for review
    with st.expander("Input data"):
        st.write(input_df)

