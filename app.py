import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# Streamlit app
st.title('Call Center Performance Predictor')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load and preprocess data
    df = pd.read_csv(uploaded_file, sep='\t')

    # Calculate industry statistics
    def calculate_industry_stats(df):
        metrics = ['Average Call Duration (min)', 'Hold Time (sec)', 'Abandonment Rate (%)', 
                   'ASA (sec)', 'ACW (sec)', 'Sentiment Score', 'CSAT (%)', 
                   'Average Waiting Time (AWT sec)', 'Average Handle Time (AHT min)', 
                   'Call Transfer Rate (%)', 'First Call Resolution (FCR %)', 'Churn Rate (%)']
        
        industry_stats = df.groupby('Industry')[metrics].agg(['mean', 'std'])
        return industry_stats

    industry_stats = calculate_industry_stats(df)

    # User inputs
    industry = st.selectbox('Select your industry', df['Industry'].unique())

    st.subheader('Enter your metrics:')
    user_metrics = {}
    for metric in industry_stats.columns.levels[0]:
        if metric not in ['First Call Resolution (FCR %)', 'Churn Rate (%)']:
            user_metrics[metric] = st.slider(f'{metric}', 
                                             0.0,  # Set minimum to 0
                                             float(df[metric].max()), 
                                             float(df[metric].mean()))

    # Calculate z-scores
    z_scores = {}
    for metric, value in user_metrics.items():
        mean = industry_stats.loc[industry, (metric, 'mean')]
        std = industry_stats.loc[industry, (metric, 'std')]
        z_scores[metric] = (value - mean) / std

    # Define weights (you may want to adjust these based on domain knowledge)
    weights = {
        'Average Call Duration (min)': -0.1,
        'Hold Time (sec)': -0.15,
        'Abandonment Rate (%)': -0.2,
        'ASA (sec)': -0.1,
        'ACW (sec)': -0.05,
        'Sentiment Score': 0.15,
        'CSAT (%)': 0.2,
        'Average Waiting Time (AWT sec)': -0.05,
        'Average Handle Time (AHT min)': -0.1,
        'Call Transfer Rate (%)': -0.1
    }

    # Ensure weights match the input metrics
    for metric in user_metrics:
        if metric not in weights:
            weights[metric] = 0  # Assign a default weight of 0 for any missing metrics

    # Calculate predictions
    fcr_prediction = sum(z_scores[metric] * weights[metric] for metric in z_scores)
    churn_prediction = -fcr_prediction  # Assuming inverse relationship

    # Display predictions
    st.subheader('Predictions:')
    st.write(f'Predicted FCR: {50 + fcr_prediction*10:.2f}%')
    st.write(f'Predicted Churn Rate: {10 + churn_prediction:.2f}%')

    # Visualize impact
    st.subheader('Metric Impact on Predictions:')
    fig, ax = plt.subplots(figsize=(10, 6))
    impact = [z_scores[metric] * weights[metric] for metric in z_scores]
    ax.bar(z_scores.keys(), impact)
    plt.xticks(rotation=45, ha='right')
    plt.title('Impact of Metrics on Predictions')
    plt.tight_layout()
    st.pyplot(fig)

    # Actionable insights
    st.subheader('Actionable Insights:')
    sorted_impact = sorted(zip(z_scores.keys(), impact), key=lambda x: abs(x[1]), reverse=True)
    for metric, imp in sorted_impact[:3]:
        if imp > 0:
            st.write(f"- Maintain or improve your performance in {metric}")
        else:
            st.write(f"- Focus on improving your {metric}")

    # Display input counts
    st.subheader('Verification:')
    st.write(f"Number of user inputs: {len(user_metrics)}")
    st.write(f"Number of z-scores calculated: {len(z_scores)}")
    if len(user_metrics) == len(z_scores):
        st.write("✅ Input count matches z-score count.")
    else:
        st.write("❌ Input count does not match z-score count. Please check the code.")

else:
    st.write("Please upload a CSV file to proceed.")

# Documentation
st.subheader('How to use this app:')
st.write("""
1. Upload your CSV file containing the call center metrics data.
2. Select your industry from the dropdown menu.
3. Adjust the sliders to input your current performance metrics. All sliders start at 0 for you to set your exact values.
4. The app will calculate z-scores based on your industry's averages and standard deviations.
5. Predicted FCR and Churn rates are calculated using a weighted sum of the z-scores.
6. The bar chart shows the impact of each metric on the predictions.
7. Actionable insights highlight the top areas for improvement.

Note: This model uses simplified assumptions and should be used as a general guide rather than a precise predictor.
""")
