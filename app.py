import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the trained models
fcr_model = joblib.load('fcr_model.pkl')
churn_model = joblib.load('churn_model.pkl')

# Function to preprocess the input data
def preprocess_data(input_data, label_encoder, scaler):
    input_data['Industry'] = label_encoder.transform(input_data['Industry'])
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

# App title and description
st.title("Performance Optimizer Pro")
st.write("Predict and optimize your First Call Resolution (FCR) and Churn rates based on your performance metrics.")

# Industry selection
industries = ['Technology', 'Healthcare', 'Retail', 'Transportation', 'Finance']
industry = st.selectbox("Select Industry", industries)

# Input section
st.subheader("Input your current performance metrics:")
average_call_duration = st.number_input("Average Call Duration (min)")
hold_time = st.number_input("Hold Time (sec)")
abandonment_rate = st.number_input("Abandonment Rate (%)")
asa = st.number_input("ASA (sec)")
acw = st.number_input("ACW (sec)")
sentiment_score = st.number_input("Sentiment Score")
csat = st.number_input("CSAT (%)")
average_waiting_time = st.number_input("Average Waiting Time (AWT sec)")
average_handle_time = st.number_input("Average Handle Time (AHT min)")
call_transfer_rate = st.number_input("Call Transfer Rate (%)")

# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'Industry': [industry],
    'Average Call Duration (min)': [average_call_duration],
    'Hold Time (sec)': [hold_time],
    'Abandonment Rate (%)': [abandonment_rate],
    'ASA (sec)': [asa],
    'ACW (sec)': [acw],
    'Sentiment Score': [sentiment_score],
    'CSAT (%)': [csat],
    'Average Waiting Time (AWT sec)': [average_waiting_time],
    'Average Handle Time (AHT min)': [average_handle_time],
    'Call Transfer Rate (%)': [call_transfer_rate]
})

# Fit and transform the label encoder and scaler within the app
label_encoder = LabelEncoder()
input_data['Industry'] = label_encoder.fit_transform(input_data['Industry'])

scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Prediction and optimization
st.subheader("Select the performance indicator to optimize:")
option = st.selectbox("", ["First Call Resolution (FCR)", "Churn"])

if option == "First Call Resolution (FCR)":
    prediction = fcr_model.predict(input_data_scaled)
    st.write(f"Predicted FCR: {prediction[0]:.2f}%")
elif option == "Churn":
    prediction = churn_model.predict(input_data_scaled)
    st.write(f"Predicted Churn Rate: {prediction[0]:.2f}%")

# Documentation
st.subheader("Documentation:")
st.write("""
- **Industry selection**: Choose the industry your data belongs to.
- **Input section**: Enter your current performance metrics.
- **Prediction and optimization**: Select the performance indicator you want to optimize and get the prediction.
""")
