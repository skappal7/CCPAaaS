import streamlit as st
import pandas as pd
import joblib
import numpy as np
from shapash.explainer.smart_explainer import SmartExplainer

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

# Load the models
best_gb_churn_model = joblib.load('best_gb_churn_model.pkl')
best_gb_fcr_model = joblib.load('best_gb_fcr_model.pkl')
best_rf_churn_model = joblib.load('best_rf_churn_model.pkl')
best_rf_fcr_model = joblib.load('best_rf_fcr_model.pkl')

# Function to make predictions
def make_predictions(model, input_data):
    return model.predict(input_data)

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

    prediction = make_predictions(model, input_data)[0]
    st.write(f"Predicted {metric}: {prediction:.2f}")

    # Feature importance
    if st.checkbox("Show Feature Importance"):
        if model_type == "Gradient Boosting":
            if metric == "First Call Resolution (FCR)":
                importances = best_gb_fcr_model.feature_importances_
            else:
                importances = best_gb_churn_model.feature_importances_
        else:
            if metric == "First Call Resolution (FCR)":
                importances = best_rf_fcr_model.feature_importances_
            else:
                importances = best_rf_churn_model.feature_importances_

        feature_names = ['Average Call Duration', 'Hold Time', 'Abandonment Rate', 'ASA', 'ACW', 'Sentiment Score', 'CSAT', 'Churn Rate', 'AWT', 'AHT', 'Call Transfer Rate']
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        st.write("### Feature Importance")
        st.write(feature_importance)

    # Model accuracy
    if st.checkbox("Show Model Accuracy"):
        if model_type == "Gradient Boosting":
            if metric == "First Call Resolution (FCR)":
                st.write(f"Model R-squared: {best_gb_fcr_model.score(X_test_fcr, y_test_fcr):.2f}")
            else:
                st.write(f"Model Accuracy: {best_gb_churn_model.score(X_test_churn, y_test_churn):.2f}")
        else:
            if metric == "First Call Resolution (FCR)":
                st.write(f"Model R-squared: {best_rf_fcr_model.score(X_test_fcr, y_test_fcr):.2f}")
            else:
                st.write(f"Model Accuracy: {best_rf_churn_model.score(X_test_churn, y_test_churn):.2f}")

with tab2:
    st.title("Model Evaluation with Shapash")
    
    if st.button("Generate Shapash Report"):
        # Prepare the SmartExplainer
        if metric == "First Call Resolution (FCR)":
            if model_type == "Gradient Boosting":
                explainer = SmartExplainer(model=best_gb_fcr_model)
            else:
                explainer = SmartExplainer(model=best_rf_fcr_model)
        else:
            if model_type == "Gradient Boosting":
                explainer = SmartExplainer(model=best_gb_churn_model)
            else:
                explainer = SmartExplainer(model=best_rf_churn_model)

        explainer.compile(X_train_fcr, y_train_fcr)
        
        st.write(explainer.plot.features_importance())
        st.write(explainer.plot.contribution_plot())

        st.write("### Shapash Report Generated")
