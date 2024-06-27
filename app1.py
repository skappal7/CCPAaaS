import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        font-family: "Poppins", sans-serif;
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
    .css-1v0mbdj {
        width: 80% !important;
    }
    .css-1pjc44v {
        font-size: 9pt;
        font-family: "Poppins", sans-serif;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        font-family: "Poppins", sans-serif;
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
    if 'Industry' in data.columns:
        data = data.drop(columns(['Industry']))
    
    return data

# Streamlit app
st.title("Call Center FCR and Churn Predictor")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data = process_data(data)

    # Calculate mean and standard deviation for each metric
    means = data.mean()
    stds = data.std()

    # Calculate correlation matrix
    correlation_matrix = data.corr()

    # Create tabs
    tab1, tab2 = st.tabs(["FCR and Churn Predictor", "Industry Trends"])

    with tab1:
        # Expander section for metrics adjustment
        with st.expander("Select Metrics"):
            st.subheader("Adjust the Metrics")
            inputs = {}
            for column in data.columns.drop(['First Call Resolution (FCR %)', 'Churn Rate (%)']):
                inputs[column] = st.slider(f"Adjust {column}", min_value=float(data[column].min()), max_value=float(data[column].max()), value=float(means[column]), key=column)

        # User input for desired improvement percentage
        desired_fcr_improvement = st.number_input("Desired Improvement in FCR (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        desired_churn_improvement = st.number_input("Desired Improvement in Churn (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        # Function to predict FCR and Churn based on weighted sum approach
        def predict_fcr_churn(inputs):
            fcr_pred = np.dot([inputs[metric] for metric in inputs.keys()], [correlation_matrix['First Call Resolution (FCR %)'][metric] for metric in inputs.keys()])
            churn_pred = np.dot([inputs[metric] for metric in inputs.keys()], [correlation_matrix['Churn Rate (%)'][metric] for metric in inputs.keys()])
            fcr_pred = np.clip(fcr_pred, 0, 100)
            churn_pred = np.clip(churn_pred, 0, 100)
            return fcr_pred, churn_pred

        # Function to calculate required metric improvements to improve FCR or Churn by a specified percentage
        def improvement_for_target(target, desired_improvement):
            improvements = {
                "Metric": [],
                "Required Improvement": [],
                "Change Direction": [],
                "Units": []
            }
            for metric in inputs.keys():
                required_change = desired_improvement / correlation_matrix[target][metric]
                direction = "Increase" if correlation_matrix[target][metric] < 0 else "Decrease"
                units = "sec" if "Time" in metric or "ASA" in metric or "ACW" in metric or "AWT" in metric else ("%" if "%" in metric else "min")
                improvements["Metric"].append(metric)
                improvements["Required Improvement"].append(f"{abs(required_change):.2f}")
                improvements["Change Direction"].append(direction)
                improvements["Units"].append(units)
            
            return pd.DataFrame(improvements)

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

            # Required improvements for FCR and Churn improvement
            st.subheader(f"Required Improvements for {desired_fcr_improvement}% FCR Improvement")
            fcr_improvement_df = improvement_for_target('First Call Resolution (FCR %)', desired_fcr_improvement)
            st.table(fcr_improvement_df)

            st.subheader(f"Required Improvements for {desired_churn_improvement}% Churn Improvement")
            churn_improvement_df = improvement_for_target('Churn Rate (%)', desired_churn_improvement)
            st.table(churn_improvement_df)

            # Fine print explanation
            st.caption("The required improvements are calculated based on the correlation coefficients between each metric and the target variable (FCR or Churn). A positive correlation indicates the metric needs to be decreased, while a negative correlation indicates the metric needs to be increased. The units of measurement are derived from the original metric definitions.")

            # Generate auto-comments based on predictions
            st.subheader("Insights and Recommendations")
            st.write("### FCR Insights")
            if fcr_pred > means['First Call Resolution (FCR %)']:
                st.write(f"The predicted FCR is above the average of {means['First Call Resolution (FCR %)']:.2f}%, indicating an efficient resolution process.")
            else:
                st.write(f"The predicted FCR is below the average of {means['First Call Resolution (FCR %)']:.2f}%. Consider optimizing your resolution strategies.")

            st.write("### Churn Insights")
            if churn_pred < means['Churn Rate (%)']:
                st.write(f"The predicted Churn rate is below the average of {means['Churn Rate (%)']:.2f}%, indicating good customer retention.")
            else:
                st.write(f"The predicted Churn rate is above the average of {means['Churn Rate (%)']:.2f}%. Consider implementing strategies to improve customer satisfaction and retention.")

        # Visualizations in a collapsible section
        with st.expander("Visualizations"):
            st.subheader("Correlation Heatmap")
            plt.figure(figsize=(8, 5))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            st.pyplot(plt.gcf())

            st.subheader("Relationships between Metrics")
            selected_metric = st.selectbox("Select Metric to Compare with FCR and Churn", data.columns.drop(['First Call Resolution (FCR %)', 'Churn Rate (%)']))
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
