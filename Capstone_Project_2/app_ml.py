import joblib
import os
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Card Fraud Detection - ML Deployment")
st.write("Interactive web app using the best trained Machine Learning model.")

@st.cache_resource
def load_ml_artifacts():
    # Automatically locate files whether running locally or inside a subfolder structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'best_ml_model.pkl')
    scaler_path = os.path.join(base_dir, 'scaler_ml.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Load model and scaler
try:
    model, scaler = load_ml_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Sidebar input controls
st.sidebar.header("Transaction Parameters")
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=1.0)
time_val = st.sidebar.number_input("Transaction Time (Seconds elapsed)", min_value=0.0, value=0.0, step=1.0)

st.sidebar.subheader("PCA Principal Components (V1 - V28)")
pca_features = {}
for i in range(1, 29):
    pca_features[f'V{i}'] = st.sidebar.slider(f'V{i}', min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

if st.button('Predict Fraud (ML)'):
    # Construct input dataframe matching training schema
    input_data = pca_features.copy()
    input_data['Amount'] = amount
    input_data['Time'] = time_val
    
    input_df = pd.DataFrame([input_data])
    
    # Apply saved scaler transformations
    input_df['NormalizedAmount'] = scaler.transform(input_df[['Amount']])
    input_df['NormalizedTime'] = scaler.transform(input_df[['Time']])
    
    # Drop unscaled columns
    input_df = input_df.drop(['Time', 'Amount'], axis=1)
    
    # Predict using the champion model
    prediction = model.predict(input_df)[0]
    
    # Display Results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("🚨 **Alert: Potential Fraudulent Transaction Detected!** This transaction exhibits high-risk behavioral patterns.")
    else:
        st.success("✅ **Legitimate Transaction.** This transaction appears normal.")