import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained models and preprocessing objects
fcr_model = joblib.load('fcr_model.pkl')
churn_model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to preprocess the input data
def preprocess_data(input_data):
    input_data['Industry'] = label_encoder.transform(input_data['Industry'])
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

# Function to display feature importance
def plot_feature_importance(model, features):
    importances = model.feature_importances_
    feature_names = features.columns
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Feature Importance')
    plt.tight_layout()
    st.pyplot(plt)

# Sidebar for user inputs
st.sidebar.title("Performance Optimizer Pro")
st.sidebar.write("Predict and optimize your First Call Resolution (FCR) and Churn rates based on your performance metrics.")

# Industry selection
industries = ['Technology', 'Healthcare', 'Retail', 'Transportation', 'Finance']
industry = st.sidebar.selectbox("Select Industry", industries)

# Input section
st.sidebar.subheader("Input your current performance metrics:")
average_call_duration = st.sidebar.slider("Average Call Duration (min)", 0.0, 60.0, 5.0)
hold_time = st.sidebar.slider("Hold Time (sec)", 0.0, 1000.0, 50.0)
abandonment_rate = st.sidebar.slider("Abandonment Rate (%)", 0.0, 100.0, 5.0)
asa = st.sidebar.slider("ASA (sec)", 0.0, 1000.0, 50.0)
acw = st.sidebar.slider("ACW (sec)", 0.0, 1000.0, 50.0)
sentiment_score = st.sidebar.slider("Sentiment Score", 0.0, 100.0, 50.0)
csat = st.sidebar.slider("CSAT (%)", 0.0, 100.0, 50.0)
average_waiting_time = st.sidebar.slider("Average Waiting Time (AWT sec)", 0.0, 1000.0, 50.0)
average_handle_time = st.sidebar.slider("Average Handle Time (AHT min)", 0.0, 60.0, 5.0)
call_transfer_rate = st.sidebar.slider("Call Transfer Rate (%)", 0.0, 100.0, 5.0)

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

# Preprocess input data
input_data_scaled = preprocess_data(input_data)

# Prediction and optimization
st.subheader("Select the performance indicator to optimize:")
option = st.selectbox("", ["First Call Resolution (FCR)", "Churn"])

if option == "First Call Resolution (FCR)":
    prediction = fcr_model.predict(input_data_scaled)
    st.write(f"Predicted FCR: {prediction[0]:.2f}%")
    plot_feature_importance(fcr_model, input_data)

elif option == "Churn":
    prediction = churn_model.predict(input_data_scaled)
    st.write(f"Predicted Churn Rate: {prediction[0]:.2f}%")
    plot_feature_importance(churn_model, input_data)

# Documentation
st.subheader("Documentation:")
st.write("""
- **Industry selection**: Choose the industry your &#8203;:citation[oaicite:0]{index=0}&#8203;
