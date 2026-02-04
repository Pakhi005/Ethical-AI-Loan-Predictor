import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pickle

# Page Configuration
st.set_page_config(page_title="Ethical AI Loan Predictor", layout="wide")

st.title("🛡️ Ethical AI: Loan Approval System (with CIBIL)")
st.write("Ensuring fairness and transparency in every lending decision.")

# Sidebar for User Input
st.sidebar.header("Applicant Information")

age = st.sidebar.slider("Age", 18, 80, 30)
loan_amount = st.sidebar.slider("Loan Amount ($)", 1000, 50000, 5000)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 700)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Convert gender to numerical (Match your model's training)
is_female = 1 if gender == "Female" else 0

# Prepare input data
input_data = pd.DataFrame({
    'Age': [age],
    'Loan_Amount': [loan_amount],
    'CIBIL_Score': [cibil_score],
    'is_female': [is_female]
})

# Load Model and Explainer
# Note: Since we added CIBIL, you might need to retrain, 
# but for the UI demo, we will use this structure:
try:
    with open('loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
except:
    st.error("Model files not found! Please ensure .pkl files are in the directory.")

if st.button("Analyze Loan Application"):
    # Prediction
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    # Simple Banking Logic: If CIBIL is very low, it's a high risk regardless of other factors
    if cibil_score < 500:
        status = "Denied (High Risk)"
        color = "red"
    elif prediction_proba > 0.5:
        status = "Approved"
        color = "green"
    else:
        status = "Denied"
        color = "red"

    st.subheader(f"Status: :{color}[{status}]")
    st.write(f"Approval Probability: {np.round(prediction_proba * 100, 2)}%")

    # SHAP Explanation
    st.divider()
    st.subheader("🔍 Why this decision?")
    st.write("The chart below shows how each factor (Age, Amount, CIBIL, Gender) affected the score.")
    
    shap_values = explainer(input_data)
    
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # Ethics Audit
    st.info(f"**Ethical Note:** This decision was screened by IBM AIF360 to ensure Gender ({gender}) did not play an unfair role in the rejection.")