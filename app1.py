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

# Function to run Monte Carlo simulations
def monte_carlo_simulation(data, target, num_simulations=10000):
    simulations = []
    for _ in range(num_simulations):
        simulated_data = data.copy()
        for column in data.columns:
            if column != target:
                change = np.random.normal(loc=0, scale=0.1 * data[column].std(), size=data.shape[0])
                simulated_data[column] += change
        if target == 'First Call Resolution (FCR %)':
            model = sm.OLS(simulated_data[target], sm.add_constant(simulated_data.drop(columns=[target]))).fit()
            predictions = model.predict(sm.add_constant(simulated_data.drop(columns=[target])))
        elif target == 'Churn Rate (%)':
            model = sm.Logit((simulated_data[target] > simulated_data[target].median()).astype(int), 
                              sm.add_constant(simulated_data.drop(columns=[target]))).fit()
            predictions = model.predict(sm.add_constant(simulated_data.drop(columns=[target])))
        simulations.append(predictions.median())
    return simulations

# Function to visualize Monte Carlo results
def visualize_monte_carlo(simulations, target):
    plt.figure(figsize=(10, 6))
    sns.histplot(simulations, kde=True, bins=50)
    plt.title(f'Monte Carlo Simulation Results for {target}')
    plt.xlabel(f'{target} (%)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

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

            # Monte Carlo Simulation for FCR and Churn Prediction
            st.subheader("Monte Carlo Simulation")
            target_variable = st.selectbox("Select Target Variable for Simulation", options=["First Call Resolution (FCR %)", "Churn Rate (%)"])
            num_simulations = st.slider("Number of Simulations", min_value=1000, max_value=20000, value=10000, step=1000)
            
            if st.button("Run Monte Carlo Simulation"):
                with st.spinner("Running Monte Carlo simulations..."):
                    simulations = monte_carlo_simulation(data, target_variable, num_simulations)
                    visualize_monte_carlo(simulations, target_variable)
                    
                    # Provide summary and insights
                    st.subheader("Simulation Summary and Insights")
                    st.write(f"The average simulated {target_variable} is {np.mean(simulations):.2f}%.")
                    st.write(f"The median simulated {target_variable} is {np.median(simulations):.2f}%.")
                    st.write(f"The 95% confidence interval for {target_variable} is ({np.percentile(simulations, 2.5):.2f}%, {np.percentile(simulations, 97.5):.2f}%).")
                    st.write("These results provide a range of potential outcomes based on the simulated scenarios. Use this information to guide decisions on performance improvements.")

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
