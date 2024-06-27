import streamlit as st
import pandas as pd
import numpy as np

# Function to process data
def process_data(data):
    # Data preprocessing
    if 'year' in data.columns:
        data = data.drop(columns=['year'])
    if 'industry' not in data.columns:
        st.error("The uploaded data does not contain an 'industry' column.")
        st.stop()
    
    return data

# Streamlit app
st.title("Call Center FCR and Churn Predictor")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data = process_data(data)

    # Industry selection
    industries = data['industry'].unique()
    selected_industry = st.selectbox("Select Industry", industries)

    # Filter data by selected industry
    industry_data = data[data['industry'] == selected_industry]

    # Drop industry column for calculations
    industry_data = industry_data.drop(columns=['industry'])

    # Calculate mean and standard deviation for each metric
    means = industry_data.mean()
    stds = industry_data.std()

    # Calculate correlation matrix
    correlation_matrix = industry_data.corr()

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

    # User inputs via sliders
    inputs = {}
    st.subheader("Adjust the Metrics")
    for column in industry_data.columns.drop(['FCR', 'Churn']):
        inputs[column] = st.slider(f"Adjust {column}", min_value=float(industry_data[column].min()), max_value=float(industry_data[column].max()), value=float(means[column]))

    # Predict button
    if st.button("Predict FCR and Churn"):
        fcr_pred, churn_pred = predict_fcr_churn(inputs)
        st.write(f"Predicted FCR: {fcr_pred:.2f}%")
        st.write(f"Predicted Churn: {churn_pred:.2f}%")

        # Impact analysis
        st.subheader("Impact Analysis")
        for metric in inputs.keys():
            fcr_impact = correlation_matrix['FCR'][metric]
            churn_impact = correlation_matrix['Churn'][metric]
            st.write(f"Adjusting {metric} by 1% changes FCR by {fcr_impact:.2f}% and Churn by {churn_impact:.2f}%")
else:
    st.info("Please upload a CSV file to proceed.")
