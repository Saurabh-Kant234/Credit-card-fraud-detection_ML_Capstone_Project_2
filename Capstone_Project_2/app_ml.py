import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.title('Credit Card Fraud Detection - ML Deployment')
st.write('Interactive web app using the best trained Machine Learning model.')

@st.cache_resource
def load_ml_artifacts():
    model = joblib.load('best_ml_model.pkl')
    scaler = joblib.load('scaler_ml.pkl')
    return model, scaler

model, scaler = load_ml_artifacts()

st.sidebar.header('Transaction Features')
amount = st.sidebar.number_input('Transaction Amount', value=100.0)
time_val = st.sidebar.number_input('Transaction Time (Sec)', value=0.0)

features = {}
for i in range(1, 29):
    features[f'V{i}'] = st.sidebar.slider(f'V{i}', -5.0, 5.0, 0.0)

if st.button('Predict Fraud (ML)'):
    input_df = pd.DataFrame([features])
    input_df['NormalizedAmount'] = scaler.transform([[amount]])[0][0]
    input_df['NormalizedTime'] = scaler.transform([[time_val]])[0][0]
    
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.error('Fraudulent Transaction Detected!')
    else:
        st.success('Legitimate Transaction.')