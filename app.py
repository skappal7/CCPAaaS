import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained models and preprocessing objects
try:
    fcr_model = joblib.load('fcr_model.pkl')
    churn_model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found. Please make sure all required .pkl files are in the same directory as the app.")
    st.stop()

# Function to preprocess the input data
def preprocess_data(input_data):
    try:
        input_data['Industry'] = label_encoder.transform([input_data['Industry'].iloc[0]])
        input_data_scaled = scaler.transform(input_data)
        return input_data_scaled
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None

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
if input_data_scaled is None:
    st.stop()

# Prediction and optimization
st.subheader("Select the performance indicator to optimize:")
option = st.selectbox("", ["First Call Resolution (FCR)", "Churn"])

try:
    if option == "First Call Resolution (FCR)":
        prediction = fcr_model.predict(input_data_scaled)
        st.write(f"Predicted FCR: {prediction[0]:.2f}%")
        plot_feature_importance(fcr_model, input_data)
        current_model = fcr_model

    elif option == "Churn":
        prediction = churn_model.predict(input_data_scaled)
        st.write(f"Predicted Churn Rate: {prediction[0]:.2f}%")
        plot_feature_importance(churn_model, input_data)
        current_model = churn_model

except Exception as e:
    st.error(f"Error making prediction: {str(e)}")
    st.stop()

# Optimization suggestions
st.subheader("Optimization Suggestions")
st.write("Based on the feature importance, here are some suggestions to improve your performance:")

# Get the top 3 most important features
feature_importances = pd.DataFrame({'Feature': input_data.columns, 'Importance': current_model.feature_importances_})
top_features = feature_importances.nlargest(3, 'Importance')

for _, feature in top_features.iterrows():
    st.write(f"- Focus on improving {feature['Feature']}")
    
    # Add specific suggestions based on the feature
    if feature['Feature'] == 'Average Call Duration (min)':
        st.write("  - Implement better call routing to reduce unnecessary transfers")
        st.write("  - Provide additional training to agents to handle calls more efficiently")
    elif feature['Feature'] == 'Hold Time (sec)':
        st.write("  - Improve your knowledge base to reduce the need for putting customers on hold")
        st.write("  - Implement a callback feature for complex issues")
    elif feature['Feature'] == 'Sentiment Score':
        st.write("  - Enhance agent training on empathy and customer handling")
        st.write("  - Implement real-time sentiment analysis to alert supervisors for intervention")
    # Add more specific suggestions for other features as needed

# Model comparison section
st.subheader("Model Comparison")
st.write("Compare the performance of different models:")

# Implement multiple models (e.g., Random Forest, Gradient Boosting, etc.)
models = {
    "Random Forest": current_model,  # Using the current model (either FCR or Churn)
    "Gradient Boosting": None,  # Add your Gradient Boosting model here
    "XGBoost": None,  # Add your XGBoost model here
}

selected_models = st.multiselect("Select models to compare", list(models.keys()))

if selected_models:
    for model_name in selected_models:
        model = models[model_name]
        if model is not None:
            prediction = model.predict(input_data_scaled)
            st.write(f"{model_name} prediction: {prediction[0]:.2f}%")
        else:
            st.write(f"{model_name} is not implemented yet.")

# Documentation
st.subheader("Documentation:")
st.write("""
- **Industry selection**: Choose the industry your data belongs to.
- **Input section**: Enter your current performance metrics using the sliders.
- **Prediction and optimization**: Select the performance indicator you want to optimize and get the prediction.
- **Feature Importance**: View the chart to see which variables impact FCR or Churn the most.
- **Optimization Suggestions**: Get specific suggestions on how to improve your top influencing factors.
- **Model Comparison**: Compare predictions from different machine learning models.
""")
