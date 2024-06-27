import streamlit as st
import pandas as pd
import numpy as np

# Function to fetch data from GitHub
def load_data():
    url = "https://raw.githubusercontent.com/skappal7/CCPAaaS/main/Call%20Center%20Data%202022%20-%202024.csv"
    data = pd.read_csv(url)
    return data

# Load the data
data = load_data()

# Data preprocessing
data = data.drop(columns=['year'])
means = data.mean()
stds = data.std()
correlation_matrix = data.corr()

# Function to calculate z-scores
def calculate_z_score(value, mean, std):
    return (value - mean) / std

# Function to predict FCR and Churn based on user inputs
def predict_fcr_churn(inputs):
    z_scores = [(calculate_z_score(inputs[metric], means[metric], stds[metric])) for metric in inputs.keys()]
    fcr_pred = sum([z * correlation_matrix['FCR'][metric] for z, metric in zip(z_scores, inputs.keys())])
    churn_pred = sum([z * correlation_matrix['Churn'][metric] for z, metric in zip(z_scores, inputs.keys())])
    fcr_pred = np.clip(fcr_pred, 0, 100)
    churn_pred = np.clip(churn_pred, 0, 100)
    return fcr_pred, churn_pred

# Streamlit app
st.title("Call Center FCR and Churn Predictor")

# User inputs via sliders
inputs = {}
st.subheader("Adjust the Metrics")
for column in data.columns.drop(['FCR', 'Churn']):
    inputs[column] = st.slider(f"Adjust {column}", min_value=float(data[column].min()), max_value=float(data[column].max()), value=float(means[column]))

# Predict button
if st.button("Predict FCR and Churn"):
    fcr_pred, churn_pred = predict_fcr_churn(inputs)
    st.write(f"Predicted FCR: {fcr_pred:.2f}%")
    st.write(f"Predicted Churn: {churn_pred:.2f}%")

    # Impact analysis
    st.subheader("Impact Analysis")
    for metric in inputs.keys():
        st.write(f"Adjusting {metric} by 1% changes FCR by {correlation_matrix['FCR'][metric]:.2f}% and Churn by {correlation_matrix['Churn'][metric]:.2f}%")
