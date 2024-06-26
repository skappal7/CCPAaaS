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

# Custom CSS for the theme
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f5f5f5;
        color: #333333;
    }
    .sidebar .sidebar-content {
        background: #0052cc;
        color: white;
    }
    .sidebar .sidebar-content .sidebar-collapse-control .icon {
        color: white;
    }
    .sidebar .sidebar-content .sidebar-collapse-control .text {
        color: white;
    }
    .sidebar .sidebar-content .sidebar-collapse-control:hover .icon {
        color: #333333;
    }
    .sidebar .sidebar-content .sidebar-collapse-control:hover .text {
        color: #333333;
    }
    .stButton>button {
        color: #0052cc;
        background: white;
        border: 2px solid #0052cc;
        padding: 0.25rem 0.5rem;
        border-radius: 3px;
    }
    .stButton>button:hover {
        color: white;
        background: #0052cc;
    }
    .stTabs>div [data-baseweb="tab"] {
        font-weight: bold;
        color: #0052cc;
    }
    .stTabs>div [data-baseweb="tab"]:hover {
        color: #002d80;
    }
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
    st.sidebar.title("Model and Metric Selection")
    metric = st.sidebar.selectbox("Select the Metric to Improve", ("First Call Resolution (FCR)", "Churn Rate"))
    model_type = st.sidebar.selectbox("Select the Model Type", ("Gradient Boosting", "Random Forest"))

    st.sidebar.title("Input Your Current Performance Metrics")
    call_duration = st.sidebar.number_input("Average Call Duration (min)", min_value=0.0, value=5.0)
    hold_time = st.sidebar.number_input("Hold Time (sec)", min_value=0.0, value=30.0)
    abandonment_rate = st.sidebar.number_input("Abandonment Rate (%)", min_value=0.0, value=5.0)
    asa = st.sidebar.number_input("ASA (sec)", min_value=0.0, value=20.0)
    acw = st.sidebar.number_input("ACW (sec)", min_value=0.0, value=15.0)
    sentiment_score = st.sidebar.number_input("Sentiment Score", min_value=0.0, max_value=1.0, value=0.5)
    csat = st.sidebar.number_input("CSAT (%)", min_value=0.0, max_value=100.0, value=80.0)
    churn_rate = st.sidebar.number_input("Churn Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
    awt = st.sidebar.number_input("Average Waiting Time (AWT sec)", min_value=0.0, value=40.0)
    aht = st.sidebar.number_input("Average Handle Time (AHT min)", min_value=0.0, value=10.0)
    call_transfer_rate = st.sidebar.number_input("Call Transfer Rate (%)", min_value=0.0, value=10.0)
    
    # Industry selection (updated to match the model's expected categories)
    industry_mapping = {
        "Telecommunications": "Telecommunications",
        "Healthcare": "Healthcare",
        "Financial Services": "Finance",
        "Retail": "Retail",
        "Technology": "Technology",
        "Transportation": "Transportation",
        "Utilities": "Utilities",
        "Other": "Other"
    }
    industry = st.sidebar.selectbox("Select Industry", list(industry_mapping.keys()))
    
    # Create one-hot encoded industry feature
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    all_industries = list(industry_mapping.values())
    encoder.fit([[ind] for ind in all_industries])
    industry_encoded = encoder.transform([[industry_mapping[industry]]])
    industry_columns = encoder.get_feature_names_out(['Industry'])
    
    # Prepare input data
    input_data = np.array([[call_duration, hold_time, abandonment_rate, asa, acw, sentiment_score, 
                            csat, churn_rate, awt, aht, call_transfer_rate]])
    input_data = np.hstack((input_data, industry_encoded))

    # Create feature names (matching the original training data)
    feature_names = [
        'Average Call Duration (min)', 'Hold Time (sec)', 'Abandonment Rate (%)', 'ASA (sec)', 'ACW (sec)', 
        'Sentiment Score', 'CSAT (%)', 'Churn Rate (%)', 'Average Waiting Time (AWT sec)', 
        'Average Handle Time (AHT min)', 'Call Transfer Rate (%)'] + list(industry_columns)

    # Convert to DataFrame with feature names
    input_df = pd.DataFrame(input_data, columns=feature_names)

    # Debug: Print input data shape and columns
    st.write(f"Input data shape: {input_df.shape}")
    st.write(f"Input data columns: {input_df.columns}")

    if metric == "First Call Resolution (FCR)":
        if model_type == "Gradient Boosting":
            model = best_gb_fcr_model
        else:
            model = best_rf_fcr_model
        st.write("### Predictions for First Call Resolution (FCR)")
    else:
        if model_type == "Gradient Boosting":
            model = best_gb_churn_model
        else:
            model = best_rf_churn_model
        st.write("### Predictions for Churn Rate")

    prediction = make_predictions(model, input_df)
    if prediction is not None:
        st.write(f"Predicted {metric}: {prediction[0]:.2f}")

    # Feature importance
    if st.checkbox("Show Feature Importance"):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
            st.write("### Feature Importance")
            st.write(feature_importance)
        else:
            st.write("Feature importance not available for this model.")

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
