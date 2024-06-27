import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define theme colors and page config
st.set_page_config(
    page_title="Call Center FCR and Churn Predictor",
    page_icon=":phone:",
    layout="wide"
)

# Apply custom CSS for theming (same as before, omitted for brevity)
st.markdown("...", unsafe_allow_html=True)

# Function to process data
def process_data(data):
    if 'Year' in data.columns:
        data = data.drop(columns=['Year'])
    if 'Industry' in data.columns:
        data = data.drop(columns=['Industry'])
    return data

# Function to calculate improvements
def calculate_improvements(data, target, desired_improvement):
    correlations = data.corr(method='spearman')[target].drop(target)
    
    improvements = {
        "Metric": [],
        "Current Value": [],
        "Suggested Change": [],
        "New Value": [],
        "Correlation": [],
        "Units": []
    }
    
    for feature, correlation in correlations.items():
        if abs(correlation) > 0.1:  # Only consider correlations stronger than 0.1
            current_value = data[feature].median()
            change = correlation * desired_improvement * current_value / 100
            new_value = current_value + change
            
            units = "sec" if "Time" in feature or "ASA" in feature or "ACW" in feature or "AWT" in feature else ("%" if "%" in feature else "min")
            
            improvements["Metric"].append(feature)
            improvements["Current Value"].append(f"{current_value:.2f}")
            improvements["Suggested Change"].append(f"{change:+.2f}")
            improvements["New Value"].append(f"{new_value:.2f}")
            improvements["Correlation"].append(f"{correlation:.4f}")
            improvements["Units"].append(units)
    
    return pd.DataFrame(improvements).sort_values("Correlation", key=abs, ascending=False)

# Streamlit app
st.title("Call Center FCR and Churn Predictor")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data = process_data(data)

    # Calculate median for each metric
    medians = data.median()

    # Calculate correlation matrix
    correlation_matrix = data.corr(method='spearman')

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
            st.metric("Industry Median FCR", f"{medians['First Call Resolution (FCR %)']:.2f}%")
        with col2:
            st.metric("Your Churn Rate", f"{current_churn:.2f}%", f"{current_churn - medians['Churn Rate (%)']:.2f}%")
            st.metric("Industry Median Churn Rate", f"{medians['Churn Rate (%)']:.2f}%")

        # User input for desired improvement percentage
        col1, col2 = st.columns(2)
        with col1:
            desired_fcr_improvement = st.number_input("Desired Improvement in FCR (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        with col2:
            desired_churn_improvement = st.number_input("Desired Reduction in Churn (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        # Calculate and display improvements
        if st.button("Calculate Improvements"):
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

            # Explanations
            if not fcr_improvement_df.empty or not churn_improvement_df.empty:
                st.subheader("Improvement Explanations")
                for _, row in fcr_improvement_df.iterrows():
                    metric, change, units, correlation = row['Metric'], float(row['Suggested Change']), row['Units'], float(row['Correlation'])
                    direction = "increase" if change > 0 else "decrease"
                    st.write(f"- To improve FCR, consider {direction}ing {metric} by {abs(change):.2f} {units}. (Correlation: {correlation:.4f})")
                
                for _, row in churn_improvement_df.iterrows():
                    metric, change, units, correlation = row['Metric'], float(row['Suggested Change']), row['Units'], float(row['Correlation'])
                    direction = "increase" if change > 0 else "decrease"
                    st.write(f"- To reduce Churn, consider {direction}ing {metric} by {abs(change):.2f} {units}. (Correlation: {correlation:.4f})")

            # Fine print explanation
            st.caption("These suggestions are based on Spearman correlations between metrics and FCR/Churn rates. The 'Correlation' score indicates the strength and direction of the relationship. Use these as general guidance and consider the practical implications of each change in your specific context.")

        # Visualizations in a collapsible section
        with st.expander("Visualizations"):
            st.subheader("Correlation Heatmap")
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title("Spearman Correlation Heatmap", fontsize=12, fontweight='bold')
            plt.xticks(fontsize=9, fontfamily='Poppins')
            plt.yticks(fontsize=9, fontfamily='Poppins')
            st.pyplot(plt.gcf())

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

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.write("Developed by Your Company Name")
