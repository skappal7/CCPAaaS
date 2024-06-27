import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to process data
def process_data(data):
    columns_to_drop = ['Year', 'Industry']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_columns]
    return data

# Function to calculate improvements based on benchmarking
def calculate_benchmark_improvements(data, target, relevant_metrics, industry_benchmark):
    improvements = {
        "Metric": [],
        "Current Value": [],
        "Benchmark Value": [],
        "Difference": [],
        "Suggested Change": [],
        "Units": []
    }

    for feature in relevant_metrics:
        if feature in data.columns:
            current_value = data[feature].median()
            benchmark_value = industry_benchmark.get(feature, np.nan)
            if pd.notna(benchmark_value):
                difference = current_value - benchmark_value
                suggested_change = -difference  # To align with the benchmark

                units = "sec" if "Time" in feature or "ASA" in feature or "ACW" in feature or "AWT" in feature else ("%" if "%" in feature else "min")

                improvements["Metric"].append(feature)
                improvements["Current Value"].append(f"{current_value:.2f}")
                improvements["Benchmark Value"].append(f"{benchmark_value:.2f}")
                improvements["Difference"].append(f"{difference:.2f}")
                improvements["Suggested Change"].append(f"{suggested_change:+.2f}")
                improvements["Units"].append(units)
    
    return pd.DataFrame(improvements).sort_values("Difference", ascending=False)

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

        # Industry benchmarks (example values)
        industry_benchmark = {
            'First Call Resolution (FCR %)': 85.0,
            'Churn Rate (%)': 5.0,
            'Average Handling Time (AHT)': 300.0,
            'After Call Work (ACW)': 30.0,
            'Average Speed of Answer (ASA)': 20.0,
            'Customer Satisfaction (CSAT)': 90.0
        }

        # Define relevant metrics for FCR and Churn
        relevant_metrics_fcr = ['Average Handling Time (AHT)', 'After Call Work (ACW)', 'Average Speed of Answer (ASA)']
        relevant_metrics_churn = ['Customer Satisfaction (CSAT)', 'Average Speed of Answer (ASA)', 'After Call Work (ACW)']

        # Main content area
        tab1, tab2 = st.tabs(["FCR and Churn Predictor", "Industry Trends"])

        with tab1:
            st.subheader("Performance Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your FCR", f"{current_fcr:.2f}%", f"{current_fcr - industry_benchmark['First Call Resolution (FCR %)']:.2f}%")
                st.metric("Industry Median FCR", f"{industry_benchmark['First Call Resolution (FCR %)']:.2f}%")
            with col2:
                st.metric("Your Churn Rate", f"{current_churn:.2f}%", f"{current_churn - industry_benchmark['Churn Rate (%)']:.2f}%")
                st.metric("Industry Median Churn Rate", f"{industry_benchmark['Churn Rate (%)']:.2f}%")

            # User input for desired improvement percentage
            desired_fcr_improvement = st.number_input("Desired Improvement in FCR (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            desired_churn_improvement = st.number_input("Desired Reduction in Churn (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            # Calculate and display improvements based on benchmarking
            if st.button("Calculate Improvements"):
                with st.spinner("Calculating improvements... This may take a moment."):
                    st.subheader(f"Suggested Changes for FCR Improvement")
                    fcr_improvement_df = calculate_benchmark_improvements(data, 'First Call Resolution (FCR %)', relevant_metrics_fcr, industry_benchmark)
                    if not fcr_improvement_df.empty:
                        st.table(fcr_improvement_df)
                    else:
                        st.write("No significant changes suggested for FCR improvement.")
                    
                    st.subheader(f"Suggested Changes for Churn Reduction")
                    churn_improvement_df = calculate_benchmark_improvements(data, 'Churn Rate (%)', relevant_metrics_churn, industry_benchmark)
                    if not churn_improvement_df.empty:
                        st.table(churn_improvement_df)
                    else:
                        st.write("No significant changes suggested for Churn reduction.")

                # Explanations
                if not fcr_improvement_df.empty or not churn_improvement_df.empty:
                    st.subheader("Improvement Explanations")
                    for _, row in fcr_improvement_df.iterrows():
                        metric, change, units, importance = row['Metric'], float(row['Suggested Change']), row['Units'], float(row['Difference'])
                        direction = "increase" if change > 0 else "decrease"
                        st.write(f"- To improve FCR, consider {direction}ing {metric} by {abs(change):.2f} {units}. (Difference: {importance:.2f})")
                    
                    for _, row in churn_improvement_df.iterrows():
                        metric, change, units, importance = row['Metric'], float(row['Suggested Change']), row['Units'], float(row['Difference'])
                        direction = "increase" if change > 0 else "decrease"
                        st.write(f"- To reduce Churn, consider {direction}ing {metric} by {abs(change):.2f} {units}. (Difference: {importance:.2f})")

                # Fine print explanation
                st.caption("These suggestions are based on benchmarking against industry standards. The 'Difference' score indicates how much each metric deviates from the benchmark. Use these as general guidance and consider the practical implications of each change in your specific context.")

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
