import streamlit as st
import pandas as pd
import numpy as np

# Define theme colors based on the provided website
st.set_page_config(
    page_title="Call Center FCR and Churn Predictor",
    page_icon=":phone:",
    layout="wide"
)

# Apply custom CSS for theming
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f9;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    .css-1d391kg {
        background-color: #00a6d6;
    }
    .css-1cpxqw2 {
        color: #333333;
    }
    .css-145kmo2 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #06516F;
    }
    .css-1vbd788 {
        font-size: 1.2rem;
        font-weight: 500;
        color: #06516F;
    }
    .stButton>button {
        background-color: #06516F;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        font-size: 1rem;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #0098DB;
        color: #ffffff;
    }
    .css-1n4pd67 {
        background-color: #f4f4f9;
        border: 1px solid #06516F;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to process data
def process_data(data):
    # Data preprocessing
    if 'Year' in data.columns:
        data = data.drop(columns=['Year'])
    if 'Industry' not in data.columns:
        st.error("The uploaded data does not contain an 'Industry' column.")
        st.stop()
    
    return data

# Streamlit app
st.title("Call Center FCR and Churn Predictor")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data = process_data(data)

    # Expander section for industry selection and metrics adjustment
    with st.expander("Select Metrics and Industry Type"):
        # Industry selection
        industries = data['Industry'].unique()
        selected_industry = st.selectbox("Select Industry", industries)

        # Filter data by selected industry
        industry_data = data[data['Industry'] == selected_industry]

        # Drop industry column for calculations
        industry_data = industry_data.drop(columns=['Industry'])

        # Calculate mean and standard deviation for each metric
        means = industry_data.mean()
        stds = industry_data.std()

        # Calculate correlation matrix
        correlation_matrix = industry_data.corr()

        # User inputs via sliders
        inputs = {}
        st.subheader("Adjust the Metrics")
        for column in industry_data.columns.drop(['First Call Resolution (FCR %)', 'Churn Rate (%)']):
            inputs[column] = st.slider(f"Adjust {column}", min_value=float(industry_data[column].min()), max_value=float(industry_data[column].max()), value=float(means[column]))

    # Function to predict FCR and Churn based on weighted sum approach
    def predict_fcr_churn(inputs):
        fcr_pred = np.dot([inputs[metric] for metric in inputs.keys()], [correlation_matrix['First Call Resolution (FCR %)'][metric] for metric in inputs.keys()])
        churn_pred = np.dot([inputs[metric] for metric in inputs.keys()], [correlation_matrix['Churn Rate (%)'][metric] for metric in inputs.keys()])
        fcr_pred = np.clip(fcr_pred, 0, 100)
        churn_pred = np.clip(churn_pred, 0, 100)
        return fcr_pred, churn_pred

    # Predict button
    if st.button("Predict FCR and Churn"):
        fcr_pred, churn_pred = predict_fcr_churn(inputs)
        st.write(f"Predicted FCR: {fcr_pred:.2f}%")
        st.write(f"Predicted Churn: {churn_pred:.2f}%")

        # Impact analysis
        st.subheader("Impact Analysis")
        impact_data = {
            "Metric": [],
            "FCR Impact (%)": [],
            "Churn Impact (%)": []
        }
        for metric in inputs.keys():
            fcr_impact = correlation_matrix['First Call Resolution (FCR %)'][metric]
            churn_impact = correlation_matrix['Churn Rate (%)'][metric]
            impact_data["Metric"].append(metric)
            impact_data["FCR Impact (%)"].append(fcr_impact * 100)
            impact_data["Churn Impact (%)"].append(churn_impact * 100)
        
        impact_df = pd.DataFrame(impact_data)
        st.table(impact_df)
else:
    st.info("Please upload a CSV file to proceed.")
