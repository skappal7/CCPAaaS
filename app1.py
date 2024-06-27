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
        data = data.drop(columns=['Industry'])
    
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

        # Function to predict FCR and Churn based on weighted sum approach
        def predict_fcr_churn(inputs):
            fcr_pred = np.dot([inputs[metric] for metric in inputs.keys()], [correlation_matrix['First Call Resolution (FCR %)'][metric] for metric in inputs.keys()])
            churn_pred = np.dot([inputs[metric] for metric in inputs.keys()], [correlation_matrix['Churn Rate (%)'][metric] for metric in inputs.keys()])
            fcr_pred = np.clip(fcr_pred, 0, 100)
            churn_pred = np.clip(churn_pred, 0, 100)
            return fcr_pred, churn_pred

        # Function to calculate required metric improvements to improve FCR by 1%
        def improvement_for_fcr():
            improvements = {
                "Metric": [],
                "Required Improvement": []
            }
            for metric in inputs.keys():
                required_change = 1 / correlation_matrix['First Call Resolution (FCR %)'][metric]
                improvements["Metric"].append(metric)
                improvements["Required Improvement"].append(f"{required_change:.2f}")
            
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

            # Required improvements for 1% FCR improvement
            st.subheader("Required Improvements for 1% FCR Improvement")
            improvement_df = improvement_for_fcr()
            st.table(improvement_df)

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
            sns.scatterplot(data=data, x=selected_metric, y='First Call Resolution (FCR %)', ax=ax[0])
            ax[0].set_title(f'{selected_metric} vs FCR')
            sns.scatterplot(data=data, x=selected_metric, y='Churn Rate (%)', ax=ax[1])
            ax[1].set_title(f'{selected_metric} vs Churn')
            st.pyplot(fig)

    with tab2:
        st.subheader("Industry Trends")
        st.write("Analyze the uploaded data to make sense of the industry trends.")
        st.dataframe(data)

    # Tech stack badges
    st.markdown(
        """
        <div style='display: flex; justify-content: center;'>
            <img src='https://img.shields.io/badge/Python-3.8-blue' style='margin: 5px;'>
            <img src='https://img.shields.io/badge/Pandas-1.3.3-green' style='margin: 5px;'>
            <img src='https://img.shields.io/badge/Streamlit-0.89.0-red' style='margin: 5px;'>
            <img src='https://img.shields.io/badge/Matplotlib-3.4.3-orange' style='margin: 5px;'>
            <img src='https://img.shields.io/badge/Seaborn-0.11.2-yellow' style='margin: 5px;'>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("Please upload a CSV file to proceed.")
