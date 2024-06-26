import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# Version check
if sklearn.__version__ != '1.2.2':
    st.error(f"Incorrect scikit-learn version. Expected 1.2.2, but got {sklearn.__version__}. Please update your requirements.txt file.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Call Center Performance Prediction as a Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (unchanged)
st.markdown(
    """
    <style>
    /* Your custom CSS here */
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to safely load models
@st.cache_resource
def load_model(filename):
    try:
        return joblib.load(filename)
    except Exception as e:
        st.warning(f"Error loading model {filename}: {str(e)}. Using a dummy model instead.")
        if 'churn' in filename:
            return GradientBoostingClassifier() if 'gb' in filename else RandomForestClassifier()
        else:
            return GradientBoostingRegressor() if 'gb' in filename else RandomForestRegressor()

# Load the models
best_gb_churn_model = load_model('best_gb_churn_model.pkl')
best_gb_fcr_model = load_model('best_gb_fcr_model.pkl')
best_rf_churn_model = load_model('best_rf_churn_model.pkl')
best_rf_fcr_model = load_model('best_rf_fcr_model.pkl')

# Function to make predictions
def make_predictions(model, input_data):
    try:
        return model.predict(input_data)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Streamlit app
st.title("Call Center Performance Predictor 📊")

# Tabs
tab1, tab2 = st.tabs(["Predictions", "Model Evaluation"])

with tab1:
    # (Your existing code for tab1 goes here, unchanged)

with tab2:
    st.title("Model Evaluation")
    st.write("This section would typically contain model evaluation metrics and visualizations.")
    st.write("For a complete implementation, you would need to have access to the test data and performance metrics.")

# Debug information
if st.checkbox("Show Debug Information"):
    st.write("### Debug Information")
    st.write(f"Python version: {sys.version}")
    st.write(f"Pandas version: {pd.__version__}")
    st.write(f"Numpy version: {np.__version__}")
    st.write(f"Scikit-learn version: {sklearn.__version__}")
    st.write(f"Joblib version: {joblib.__version__}")

# Model information
if st.checkbox("Show Model Information"):
    st.write("### Model Information")
    for name, model in {'GB Churn': best_gb_churn_model, 'GB FCR': best_gb_fcr_model, 
                        'RF Churn': best_rf_churn_model, 'RF FCR': best_rf_fcr_model}.items():
        st.write(f"{name}: {type(model).__name__}")
