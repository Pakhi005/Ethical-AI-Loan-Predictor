import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import logging

# 1. Page Config (Helps prevent blank white screens)
st.set_page_config(page_title="Ethical AI Loan Predictor", layout="centered")

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Starting Streamlit app")

@st.cache_resource
def load_assets():
    logging.debug("Loading assets...")
    # Loading files with a cache so it only happens once
    model = pickle.load(open('loan_model.pkl', 'rb'))
    explainer = pickle.load(open('shap_explainer.pkl', 'rb'))
    return model, explainer

# Try to load; if files are missing, show a clear error
try:
    model, explainer = load_assets()
    logging.debug("Assets loaded successfully")
except FileNotFoundError as e:
    logging.error(f"Error loading assets: {e}")
    st.error("⚠️ Files 'loan_model.pkl' or 'shap_explainer.pkl' not found! Please run your Jupyter Notebook cells to save them first.")
    st.stop()

st.title("🛡️ Ethical AI: Loan Approval System")
st.info("This system uses IBM AIF360 Reweighing to mitigate gender bias.")

# Sidebar for inputs
st.sidebar.header("Applicant Details")
age = st.sidebar.slider("Age", 18, 80, 30)
amount = st.sidebar.number_input("Loan Amount ($)", 500, 20000, 2000)
is_female = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: 'Female' if x==1 else 'Male')

# Prediction Logic
if st.button("Analyze Application"):
    logging.debug("Analyze Application button clicked")
    with st.spinner('Calculating fairness metrics and prediction...'):
        # Prepare input
        input_df = pd.DataFrame([[is_female, amount, age]], columns=['is_female', 'amount', 'age'])
        logging.debug(f"Input DataFrame: {input_df}")
        
        # Prediction
        prediction = model.predict(input_df)[0]
        logging.debug(f"Prediction result: {prediction}")

        if prediction == 1:
            st.success("✅ Prediction: Loan Approved")
        else:
            st.error("❌ Prediction: Loan Denied")

        # Explainability Section
        st.divider()
        st.subheader("🔍 Transparency Report (SHAP)")
        st.write("This chart explains how each factor influenced the AI's decision:")
        
        # Fixed SHAP plotting for Streamlit
        shap_values = explainer(input_df)
        logging.debug(f"SHAP values calculated: {shap_values}")
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)