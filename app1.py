import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pingouin as pg

# Function to process data
def process_data(data):
    columns_to_drop = ['Year', 'Industry']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_columns]
    return data

# Function to calculate partial correlations
def calculate_partial_correlations(data, target):
    partial_corrs = {}
    for col in data.columns:
        if col != target:
            try:
                partial_corr = pg.partial_corr(data=data, x=col, y=target, covar=list(data.columns.difference([col, target])))
                partial_corrs[col] = partial_corr['r'].values[0]
            except:
                partial_corrs[col] = 0
    return partial_corrs

# Function to perform multiple regression
def multiple_regression(data, target):
    X = data.drop(columns=[target])
    y = data[target]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

# Function to calculate rule-based improvements
def calculate_rule_based_improvements(data, target, rules):
    improvements = {
        "Metric": [],
        "Current Value": [],
        "Suggested Change": [],
        "New Value": [],
        "Importance": [],
        "Units": []
    }

    for rule in rules:
        feature = rule['metric']
        if feature in data.columns:
            current_value = data[feature].median()
            suggested_change = rule['change']
            new_value = current_value + suggested_change
            importance = rule['importance']
            units = rule['units']

            improvements["Metric"].append(feature)
            improvements["Current Value"].append(f"{current_value:.2f}")
            improvements["Suggested Change"].append(f"{suggested_change:+.2f}")
            improvements["New Value"].append(f"{new_value:.2f}")
            improvements["Importance"].append(f"{importance:.4f}")
            improvements["Units"].append(units)
    
    return pd.DataFrame(improvements).sort_values("Importance", ascending=False)

# Define rule-based recommendations based on domain knowledge
def get_rule_based_recommendations(target):
    if target == 'First Call Resolution (FCR %)':
        return [
            {'metric': 'Average Handling Time (AHT)', 'change': -30, 'importance': 0.8, 'units': 'sec'},
            {'metric': 'After Call Work (ACW)', 'change': -5, 'importance': 0.6, 'units': 'sec'},
            {'metric': 'Average Speed of Answer (ASA)', 'change': -3, 'importance': 0.4, 'units': 'sec'},
        ]
    elif target == 'Churn Rate (%)':
        return [
            {'metric': 'Customer Satisfaction (CSAT)', 'change': 5, 'importance': 0.8, 'units': '%'},
            {'metric': 'Average Speed of Answer (ASA)', 'change': -3, 'importance': 0.6, 'units': 'sec'},
            {'metric': 'After Call Work (ACW)', 'change': -5, 'importance': 0.4, 'units': 'sec'},
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

            # User input for desired improvement percentage
            desired_fcr_improvement = st.number_input("Desired Improvement in FCR (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            desired_churn_improvement = st.number_input("Desired Reduction in Churn (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            # Calculate partial correlations and multiple regression results
            st.subheader("Partial Correlation and Regression Analysis")
            target_variable = st.selectbox("Select Target Variable", options=["First Call Resolution (FCR %)", "Churn Rate (%)"])
            partial_correlations = calculate_partial_correlations(data, target_variable)
            regression_model = multiple_regression(data, target_variable)

            st.write("Partial Correlations:")
            st.write(partial_correlations)
            st.write("Multiple Regression Summary:")
            st.write(regression_model.summary())

            # Calculate and display rule-based improvements
            if st.button("Calculate Improvements"):
                with st.spinner("Calculating improvements... This may take a moment."):
                    st.subheader(f"Suggested Changes for {target_variable} Improvement")
                    rules = get_rule_based_recommendations(target_variable)
                    improvements_df = calculate_rule_based_improvements(data, target_variable, rules)
                    if not improvements_df.empty:
                        st.table(improvements_df)
                    else:
                        st.write("No significant changes suggested.")

                # Explanations
                if not improvements_df.empty:
                    st.subheader("Improvement Explanations")
                    for _, row in improvements_df.iterrows():
                        metric, change, units, importance = row['Metric'], float(row['Suggested Change']), row['Units'], float(row['Importance'])
                        direction = "increase" if change > 0 else "decrease"
                        st.write(f"- To improve {target_variable}, consider {direction}ing {metric} by {abs(change):.2f} {units}. (Importance: {importance:.4f})")

                # Fine print explanation
                st.caption("These suggestions are based on partial correlation, multiple regression, and rule-based recommendations. The 'Importance' score indicates the relative impact of each metric. Use these as general guidance and consider the practical implications of each change in your specific context.")

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
