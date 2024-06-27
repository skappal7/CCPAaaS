Vimport streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Function to process data
def process_data(data):
    columns_to_drop = ['Year', 'Industry']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_columns]
    return data

# Function to perform logistic regression
def logistic_regression(data, target):
    X = data.drop(columns=[target])
    X = sm.add_constant(X)
    y = (data[target] > data[target].median()).astype(int)  # Convert target to binary based on median
    model = sm.Logit(y, X).fit()
    return model

# Function to perform linear regression
def linear_regression(data, target):
    X = data.drop(columns=[target])
    X = sm.add_constant(X)
    y = data[target]
    model = sm.OLS(y, X).fit()
    return model

# Rule-Based Recommendations
def rule_based_recommendations(target):
    if target == 'First Call Resolution (FCR %)':
        return [
            {'metric': 'Average Handling Time (AHT)', 'suggestion': 'Reduce by 10%'},
            {'metric': 'After Call Work (ACW)', 'suggestion': 'Reduce by 5%'},
            {'metric': 'Average Speed of Answer (ASA)', 'suggestion': 'Reduce by 5 seconds'}
        ]
    elif target == 'Churn Rate (%)':
        return [
            {'metric': 'Customer Satisfaction (CSAT)', 'suggestion': 'Increase by 5%'},
            {'metric': 'Average Speed of Answer (ASA)', 'suggestion': 'Reduce by 3 seconds'},
            {'metric': 'After Call Work (ACW)', 'suggestion': 'Reduce by 5%'}
        ]
    return []

# Streamlit app
st.set_page_config(page_title="Call Center FCR and Churn Predictor", page_icon=":phone:", layout="wide")
st.title("Call Center FCR and Churn Predictor")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        data = process_data(data)
        
        # Calculate median for each metric
        medians = data.median()

        # Sidebar for current performance input
        st.sidebar.header("Current Performance")
        current_fcr = st.sidebar.number_input("Current FCR (%)", min_value=0.0, max_value=100.0, value=float(medians['First Call Resolution (FCR %)']))
        current_churn = st.sidebar.number_input("Current Churn Rate (%)", min_value=0.0, max_value=100.0, value=float(medians['Churn Rate (%)']))

        # Main content area
        tab1, tab2 = st.tabs(["FCR and Churn Predictor", "Industry Trends"])

        with tab1:
            st.subheader("Performance Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your FCR", f"{current_fcr:.2f}%", f"{current_fcr - medians['First Call Resolution (FCR %)']:.2f}%")
            with col2:
                st.metric("Your Churn Rate", f"{current_churn:.2f}%", f"{current_churn - medians['Churn Rate (%)']:.2f}%")

            # Logistic Regression for Churn Prediction
            st.subheader("Logistic Regression for Churn Prediction")
            target_churn = 'Churn Rate (%)'
            logit_model = logistic_regression(data, target_churn)
            st.write("Logistic Regression Model Summary:")
            st.write(logit_model.summary())

            # Linear Regression for FCR Prediction
            st.subheader("Linear Regression for FCR Prediction")
            target_fcr = 'First Call Resolution (FCR %)'
            linreg_model = linear_regression(data, target_fcr)
            st.write("Linear Regression Model Summary:")
            st.write(linreg_model.summary())

            # Rule-Based Recommendations
            st.subheader("Rule-Based Recommendations")
            selected_target = st.selectbox("Select Target Variable", options=["First Call Resolution (FCR %)", "Churn Rate (%)"])
            recommendations = rule_based_recommendations(selected_target)
            st.write("Recommendations:")
            for rec in recommendations:
                st.write(f"{rec['metric']}: {rec['suggestion']}")

            # Simplified Simulation
            st.subheader("Simplified Simulation")
            st.write("Adjust the metrics to see the projected impact on FCR and Churn Rate:")

            aht_change = st.slider("Change in AHT (sec)", min_value=-100, max_value=100, value=0, step=5)
            acw_change = st.slider("Change in ACW (sec)", min_value=-20, max_value=20, value=0, step=1)
            asa_change = st.slider("Change in ASA (sec)", min_value=-10, max_value=10, value=0, step=1)
            csat_change = st.slider("Change in CSAT (%)", min_value=-10, max_value=10, value=0, step=1)

            changes = {
                'Average Handling Time (AHT)': aht_change,
                'After Call Work (ACW)': acw_change,
                'Average Speed of Answer (ASA)': asa_change,
                'Customer Satisfaction (CSAT)': csat_change
            }

            if st.button("Simulate Impact"):
                with st.spinner("Simulating impact..."):
                    # Apply changes to data
                    simulated_data = data.copy()
                    for feature, change in changes.items():
                        if feature in simulated_data.columns:
                            simulated_data[feature] += change
                    
                    # Predict new FCR and Churn values
                    simulated_fcr = linreg_model.predict(sm.add_constant(simulated_data.drop(columns=[target_fcr]))).mean()
                    simulated_churn = logit_model.predict(sm.add_constant(simulated_data.drop(columns=[target_churn]))).mean()

                    st.metric("Simulated FCR", f"{simulated_fcr:.2f}%")
                    st.metric("Simulated Churn Rate", f"{simulated_churn:.2f}%")

            # Visualizations in a collapsible section
            with st.expander("Visualizations"):
                st.subheader("Correlation Heatmap")
                correlation_matrix = data.corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
                plt.title("Correlation Heatmap", fontsize=12, fontweight='bold')
                plt.xticks(fontsize=9, fontfamily='Poppins')
                plt.yticks(fontsize=9, fontfamily='Poppins')
                st.pyplot(plt)

                st.subheader("Key Metrics vs FCR and Churn")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot for FCR
                top_fcr_corr = correlation_matrix['First Call Resolution (FCR %)'].sort_values(key=abs, ascending=False)[1:6]
                sns.barplot(x=top_fcr_corr.index, y=top_fcr_corr.values, ax=ax1)
                ax1.set_title("Top 5 Correlations with FCR", fontsize=10, fontweight='bold')
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
                ax1.tick_params(labelsize=9)
                
                # Plot for Churn
                top_churn_corr = correlation_matrix['Churn Rate (%)'].sort_values(key=abs, ascending=False)[1:6]
                sns.barplot(x=top_churn_corr.index, y=top_churn_corr.values, ax=ax2)
                ax2.set_title("Top 5 Correlations with Churn Rate", fontsize=10, fontweight='bold')
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
                ax2.tick_params(labelsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)

        with tab2:
            st.subheader("Industry Trends")
            st.write("This section could include more detailed analysis of industry trends based on the uploaded data.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your CSV file and try again. Ensure all relevant columns contain numeric data.")

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.write("Developed by Your Company Name")
