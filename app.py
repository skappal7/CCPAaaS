import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sys
import traceback
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

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

# Debug information
st.sidebar.write("### Debug Information")
st.sidebar.write(f"Python version: {sys.version}")
st.sidebar.write(f"Pandas version: {pd.__version__}")
st.sidebar.write(f"Numpy version: {np.__version__}")
st.sidebar.write(f"Scikit-learn version: {sklearn.__version__}")
st.sidebar.write(f"Joblib version: {joblib.__version__}")

# Function to safely load models
def load_model(filename):
    try:
        model_path = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            st.warning(f"Model file not found: {filename}")
            return None
    except Exception as e:
        st.error(f"Error loading model {filename}: {str(e)}")
        st.text(traceback.format_exc())
        return None

# Load the models
models = {
    'best_gb_churn_model': load_model('best_gb_churn_model.pkl'),
    'best_gb_fcr_model': load_model('best_gb_fcr_model.pkl'),
    'best_rf_churn_model': load_model('best_rf_churn_model.pkl'),
    'best_rf_fcr_model': load_model('best_rf_fcr_model.pkl')
}

# Function to make predictions
def make_predictions(model, input_data):
    if model is None:
        return None
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

    # Prepare input data
    input_data = np.array([[call_duration, hold_time, abandonment_rate, asa, acw, sentiment_score, csat, churn_rate, awt, aht, call_transfer_rate]])

    # Select the appropriate model
    model_key = f"best_{'gb' if model_type == 'Gradient Boosting' else 'rf'}_{('fcr' if metric == 'First Call Resolution (FCR)' else 'churn')}_model"
    model = models[model_key]

    if model is not None:
        st.write(f"### Predictions for {metric}")
        prediction = make_predictions(model, input_data)
        if prediction is not None:
            st.write(f"Predicted {metric}: {prediction[0]:.2f}")

        # Feature importance
        if st.checkbox("Show Feature Importance"):
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = ['Average Call Duration', 'Hold Time', 'Abandonment Rate', 'ASA', 'ACW', 'Sentiment Score', 'CSAT', 'Churn Rate', 'AWT', 'AHT', 'Call Transfer Rate']
                feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
                st.write("### Feature Importance")
                st.write(feature_importance)
            else:
                st.write("Feature importance not available for this model.")

        # Model accuracy
        if st.checkbox("Show Model Accuracy"):
            st.write("Model accuracy information not available. Please retrain the model with a test set to get accuracy metrics.")

with tab2:
    st.title("Model Evaluation with Shapash")
    st.write("Shapash integration is not implemented in this version of the app.")

# Add this at the end of your script
if st.button("Print Model Information"):
    for name, model in models.items():
        if model is not None:
            st.write(f"{name}: {type(model).__name__}")
        else:
            st.write(f"{name}: Not loaded")
