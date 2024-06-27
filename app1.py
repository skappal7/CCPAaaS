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
    /* Custom styles for visualizations */
    .matplotlib-figure {
        font-family: "Poppins", sans-serif;
        font-size: 9pt;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to process data
def process_data(data):
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

    # Sidebar for metric adjustments
    st.sidebar.header("Adjust Metrics")
    inputs = {}
    for column in data.columns.drop(['First Call Resolution (FCR %)', 'Churn Rate (%)']):
        inputs[column] = st.sidebar.slider(f"{column}", min_value=float(data[column].min()), max_value=float(data[column].max()), value=float(means[column]), key=column)

    # Main content area
    tab1, tab2 = st.tabs(["FCR and Churn Predictor", "Industry Trends"])

    with tab1:
        # User input for desired improvement percentage
        col1, col2 = st.columns(2)
        with col1:
            desired_fcr_improvement = st.number_input("Desired Improvement in FCR (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        with col2:
            desired_churn_improvement = st.number_input("Desired Improvement in Churn (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        # Function to predict FCR and Churn based on weighted sum approach
        def predict_fcr_churn(inputs):
            fcr_pred = np.dot([inputs[metric] for metric in inputs.keys()], [correlation_matrix['First Call Resolution (FCR %)'][metric] for metric in inputs.keys()])
            churn_pred = np.dot([inputs[metric] for metric in inputs.keys()], [correlation_matrix['Churn Rate (%)'][metric] for metric in inputs.keys()])
            fcr_pred = np.clip(fcr_pred, 0, 100)
            churn_pred = np.clip(churn_pred, 0, 100)
            return fcr_pred, churn_pred

        # Function to calculate required metric improvements
        def improvement_for_target(target, desired_improvement):
            improvements = {
                "Metric": [],
                "Required Change": [],
                "Change Direction": [],
                "Units": []
            }
            for metric in inputs.keys():
                correlation = correlation_matrix[target][metric]
                if correlation != 0:
                    required_change = desired_improvement / correlation
                    direction = "Increase" if correlation > 0 else "Decrease"
                    units = "sec" if "Time" in metric or "ASA" in metric or "ACW" in metric or "AWT" in metric else ("%" if "%" in metric else "min")
                    improvements["Metric"].append(metric)
                    improvements["Required Change"].append(f"{abs(required_change):.2f}")
                    improvements["Change Direction"].append(direction)
                    improvements["Units"].append(units)
            
            return pd.DataFrame(improvements)

        # Predict button
        if st.button("Predict FCR and Churn"):
            fcr_pred, churn_pred = predict_fcr_churn(inputs)
            st.write(f"Predicted FCR: {fcr_pred:.2f}%")
            st.write(f"Predicted Churn: {churn_pred:.2f}%")

            # Required improvements for FCR and Churn improvement
            st.subheader(f"Required Changes for {desired_fcr_improvement}% FCR Improvement")
            fcr_improvement_df = improvement_for_target('First Call Resolution (FCR %)', desired_fcr_improvement)
            st.table(fcr_improvement_df)

            st.subheader(f"Required Changes for {desired_churn_improvement}% Churn Improvement")
            churn_improvement_df = improvement_for_target('Churn Rate (%)', desired_churn_improvement)
            st.table(churn_improvement_df)

            # Fine print explanation
            st.caption("These values are derived based on the correlation coefficients between each metric and the target variable (FCR or Churn). The required change is calculated as the desired improvement divided by the correlation coefficient. A positive correlation indicates the metric needs to be increased, while a negative correlation indicates the metric needs to be decreased. The actual impact may vary due to the complex relationships between variables.")

        # Visualizations in a collapsible section
        with st.expander("Visualizations"):
            st.subheader("Correlation Heatmap")
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title("Correlation Heatmap", fontsize=12, fontweight='bold')
            plt.xticks(fontsize=9, fontfamily='Poppins')
            plt.yticks(fontsize=9, fontfamily='Poppins')
            st.pyplot(plt.gcf())

            st.subheader("Relationships between Metrics")
            selected_metric = st.selectbox("Select Metric to Compare with FCR and Churn", data.columns.drop(['First Call Resolution (FCR %)', 'Churn Rate (%)']))
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.scatterplot(data=data, x=selected_metric, y='First Call Resolution (FCR %)', ax=ax[0])
            ax[0].set_title(f'{selected_metric} vs FCR', fontsize=10, fontweight='bold')
            ax[0].tick_params(labelsize=9)
            sns.scatterplot(data=data, x=selected_metric, y='Churn Rate (%)', ax=ax[1])
            ax[1].set_title(f'{selected_metric} vs Churn', fontsize=10, fontweight='bold')
            ax[1].tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)

    with tab2:
        st.write("Industry Trends tab content goes here.")

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.write("Developed by Your Company Name")
