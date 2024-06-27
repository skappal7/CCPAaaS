import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to process and normalize data
def process_data(data):
    # Keep only numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_columns]
    
    # Normalize data
    return (data - data.min()) / (data.max() - data.min())

# Function to handle outliers
def handle_outliers(data, lower_percentile=1, upper_percentile=99):
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    return np.clip(data, lower, upper)

# Custom weighted feature importance
def custom_feature_importance(X, y, n_iterations=100, sample_size=0.8):
    n_samples, n_features = X.shape
    feature_importances = np.zeros(n_features)
    
    for _ in range(n_iterations):
        sample_indices = np.random.choice(n_samples, size=int(sample_size * n_samples), replace=True)
        X_sample, y_sample = X.iloc[sample_indices], y.iloc[sample_indices]
        
        correlations = X_sample.corrwith(y_sample)
        potential_improvements = 1 - X_sample.abs()
        weighted_importances = correlations.abs() * potential_improvements.mean()
        feature_importances += weighted_importances / weighted_importances.sum()
    
    return feature_importances / n_iterations

# Function to calculate improvements
def calculate_improvements(data, target, desired_improvement, threshold=0.05):
    X = data.drop(columns=[target])
    y = data[target]
    
    feature_importances = custom_feature_importance(X, y)
    
    improvements = {
        "Metric": [],
        "Current Value": [],
        "Suggested Change (%)": [],
        "New Value": [],
        "Importance": []
    }
    
    for feature, importance in zip(X.columns, feature_importances):
        if importance < threshold:
            continue
        
        current_value = X[feature].median()
        correlation = np.corrcoef(X[feature], y)[0, 1]
        direction = 1 if (target == 'First Call Resolution (FCR %)') == (correlation > 0) else -1
        change_percentage = direction * importance * desired_improvement
        new_value = current_value + change_percentage
        
        improvements["Metric"].append(feature)
        improvements["Current Value"].append(f"{current_value:.2f}")
        improvements["Suggested Change (%)"].append(f"{change_percentage:+.2f}%")
        improvements["New Value"].append(f"{new_value:.2f}")
        improvements["Importance"].append(f"{importance:.4f}")
    
    return pd.DataFrame(improvements).sort_values("Importance", ascending=False)

# Streamlit app
st.set_page_config(page_title="Call Center FCR and Churn Predictor", page_icon=":phone:", layout="wide")

st.title("Call Center FCR and Churn Predictor")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        original_data = data.copy()
        data = process_data(data)
        data = data.apply(handle_outliers)
        
        # Calculate median for each metric
        medians = original_data.median()

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
                st.metric("Dataset Median FCR", f"{medians['First Call Resolution (FCR %)']:.2f}%")
            with col2:
                st.metric("Your Churn Rate", f"{current_churn:.2f}%", f"{current_churn - medians['Churn Rate (%)']:.2f}%")
                st.metric("Dataset Median Churn Rate", f"{medians['Churn Rate (%)']:.2f}%")

            # User input for desired improvement percentage
            col1, col2 = st.columns(2)
            with col1:
                desired_fcr_improvement = st.number_input("Desired Improvement in FCR (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            with col2:
                desired_churn_improvement = st.number_input("Desired Reduction in Churn (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            # Calculate and display improvements
            if st.button("Calculate Improvements"):
                with st.spinner("Calculating improvements... This may take a moment."):
                    st.subheader(f"Suggested Changes for {desired_fcr_improvement}% FCR Improvement")
                    fcr_improvement_df = calculate_improvements(data, 'First Call Resolution (FCR %)', desired_fcr_improvement)
                    if not fcr_improvement_df.empty:
                        st.table(fcr_improvement_df)
                    else:
                        st.write("No significant changes suggested for FCR improvement.")
                    
                    st.subheader(f"Suggested Changes for {desired_churn_improvement}% Churn Reduction")
                    churn_improvement_df = calculate_improvements(data, 'Churn Rate (%)', -desired_churn_improvement)  # Negative because we want to reduce churn
                    if not churn_improvement_df.empty:
                        st.table(churn_improvement_df)
                    else:
                        st.write("No significant changes suggested for Churn reduction.")

                # Explanation
                st.markdown("""
                These suggestions are based on a custom feature importance algorithm. The 'Importance' score indicates the relative impact of each metric. 
                Use these as general guidance and consider the practical implications of each change in your specific context.
                
                **Methodology:**
                - Data is normalized and outliers are handled to ensure fair comparisons.
                - A weighted importance is calculated for each metric, considering both correlation with FCR/Churn and potential for improvement.
                - Suggested changes are expressed as percentages of the current value.
                - Only metrics with importance above a certain threshold are shown to focus on the most impactful changes.
                - The direction of change (increase/decrease) is determined based on the metric's relationship with FCR/Churn.
                """)

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
            st.subheader("Dataset Trends")
            st.write("This section could include more detailed analysis of trends based on the uploaded data.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your CSV file and try again. Ensure all relevant columns contain numeric data.")

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.write("Developed by Your Company Name")
