import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration and theme
st.set_page_config(page_title="Performance Optimizer Pro", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        color: #000000;
        font-family: 'Poppins', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #07B1FC;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("Performance Optimizer Pro")
st.write("Predict and optimize your First Call Resolution (FCR) and Churn rates based on your performance metrics.")

# File uploader for data
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # Preprocess data to calculate industry averages and standard deviations
    industry_stats = data.groupby('Industry').agg(['mean', 'std']).reset_index()

    # Identify common columns across all industries
    common_columns = set(industry_stats.columns.levels[0])
    common_columns.discard('Industry')  # Remove 'Industry' from common columns
    
    # Function to calculate z-scores
    def calculate_z_scores(input_data, industry):
        # Select the relevant industry statistics
        industry_mean = industry_stats[industry_stats['Industry'] == industry].xs('mean', level=1, axis=1)[common_columns]
        industry_std = industry_stats[industry_stats['Industry'] == industry].xs('std', level=1, axis=1)[common_columns]
        
        # Align columns
        input_data = input_data[common_columns]
        
        # Calculate z-scores
        z_scores = (input_data - industry_mean) / industry_std
        return z_scores

    # Sidebar for user inputs
    st.sidebar.title("Input Metrics")
    st.sidebar.write("Enter your current performance metrics.")

    # Industry selection
    industries = industry_stats['Industry'].unique()
    industry = st.sidebar.selectbox("Select Industry", industries)

    # Input section
    input_data = {}
    for column in common_columns:
        max_value = data[column].max()
        min_value = data[column].min()
        if 'Rate' in column or 'Score' in column or 'CSAT' in column:
            input_data[column] = st.sidebar.slider(f"{column}", 0.0, 100.0, 50.0)
        elif 'Time' in column or 'Duration' in column or 'Handle' in column:
            input_data[column] = st.sidebar.slider(f"{column}", 0.0, max_value, min_value + (max_value - min_value) / 2)
    
    input_data = pd.DataFrame(input_data, index=[0])

    # Calculate z-scores
    z_scores = calculate_z_scores(input_data, industry).squeeze()

    # Debug prints to check lengths
    st.write(f"Number of input metrics: {len(input_data.columns)}")
    st.write(f"Number of z-scores: {len(z_scores)}")

    # Ensure z_scores and weights have the same length
    if len(z_scores) != len(input_data.columns):
        st.error("Mismatch in the number of z-scores and metrics. Please check the input data.")
    else:
        # Weighted sum for predictions (weights can be adjusted based on domain expertise)
        weights = {
            'Average Call Duration (min)': 0.2,
            'Hold Time (sec)': 0.1,
            'Abandonment Rate (%)': 0.15,
            'ASA (sec)': 0.1,
            'ACW (sec)': 0.05,
            'Sentiment Score': 0.1,
            'CSAT (%)': 0.1,
            'Average Waiting Time (AWT sec)': 0.05,
            'Average Handle Time (AHT min)': 0.05,
            'Call Transfer Rate (%)': 0.1
        }

        # Filter weights to match common columns
        filtered_weights = {k: v for k, v in weights.items() if k in common_columns}
        weights_series = pd.Series(filtered_weights)
        
        predicted_fcr = np.sum(z_scores * weights_series)
        predicted_churn = np.sum(z_scores * weights_series)

        # Display predictions
        st.subheader("Predicted First Call Resolution (FCR) and Churn Rates")
        st.write(f"Predicted FCR: {predicted_fcr:.2f}%")
        st.write(f"Predicted Churn Rate: {predicted_churn:.2f}%")

        # Visualization of impact
        st.subheader("Impact of Metrics on Predictions")
        impact_data = pd.DataFrame({
            'Metric': input_data.columns,
            'Impact on FCR': z_scores * weights_series.values,
            'Impact on Churn': z_scores * weights_series.values
        })
        impact_data = impact_data.melt(id_vars='Metric', var_name='Prediction', value_name='Impact')

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Impact', y='Metric', hue='Prediction', data=impact_data)
        plt.title('Impact of Metrics on FCR and Churn Predictions')
        st.pyplot(plt)

        # Documentation
        st.subheader("Documentation:")
        st.write("""
        - **Industry selection**: Choose the industry your data belongs to.
        - **Input section**: Enter your current performance metrics using the sliders.
        - **Prediction and optimization**: The app uses statistical methods to predict FCR and Churn rates.
        - **Impact Visualization**: See which metrics have the most impact on the predictions.
        """)
else:
    st.write("Please upload a CSV or Excel file to start the simulation.")
