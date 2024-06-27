import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Function to process data
def process_data(data):
    columns_to_drop = ['Year']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_columns].fillna(0)  # Ensure no missing values
    return data

# Function to run Monte Carlo simulations
def monte_carlo_simulation(data, target, num_simulations=10000):
    simulations = []
    for _ in range(num_simulations):
        simulated_data = data.copy()
        for column in data.columns:
            if column != target:
                change = np.random.normal(loc=0, scale=0.1 * data[column].std(), size=data.shape[0])
                simulated_data[column] = np.maximum(0, simulated_data[column] + change)  # Ensure no negative values
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

# Example industry benchmark data
industry_benchmarks = {
    "Retail": {'FCR': 80, 'Churn': 10},
    "Healthcare": {'FCR': 85, 'Churn': 5},
    "Technology": {'FCR': 78, 'Churn': 12},
    "Finance": {'FCR': 82, 'Churn': 8}
}

# Sidebar for industry selection and current performance input
st.sidebar.header("Settings")

industry = st.sidebar.selectbox("Select Industry", options=list(industry_benchmarks.keys()))
benchmark_fcr = industry_benchmarks[industry]['FCR']
benchmark_churn = industry_benchmarks[industry]['Churn']

# Sidebar for current performance input
st.sidebar.header("Current Performance")
current_fcr = st.sidebar.number_input("Current FCR (%)", min_value=0.0, max_value=100.0, value=float(benchmark_fcr))
current_churn = st.sidebar.number_input("Current Churn Rate (%)", min_value=0.0, max_value=100.0, value=float(benchmark_churn))

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        data = process_data(data)
        
        # Calculate median for each metric
        medians = data.median()

        # Calculate correlations with FCR and Churn
        correlation_fcr = data.corr()['First Call Resolution (FCR %)'].sort_values(ascending=False)
        correlation_churn = data.corr()['Churn Rate (%)'].sort_values(ascending=False)

        # Main content area
        tab1, tab2 = st.tabs(["FCR and Churn Predictor", "Industry Trends"])

        with tab1:
            st.subheader("Performance Comparison")
            col1, col2 = st.columns(2)
            fcr_delta = current_fcr - benchmark_fcr
            churn_delta = current_churn - benchmark_churn
            with col1:
                st.metric("Your FCR", f"{current_fcr:.2f}%", f"{fcr_delta:.2f}% from industry median", delta_color="normal" if fcr_delta >= 0 else "inverse")
                st.metric("Industry FCR", f"{benchmark_fcr:.2f}%")
            with col2:
                st.metric("Your Churn Rate", f"{current_churn:.2f}%", f"{churn_delta:.2f}% from industry median", delta_color="inverse" if churn_delta >= 0 else "normal")
                st.metric("Industry Churn Rate", f"{benchmark_churn:.2f}%")

            # Monte Carlo Simulation for FCR and Churn Prediction
            st.subheader("Monte Carlo Simulation")
            target_variable = st.selectbox("Select Target Variable for Simulation", options=["First Call Resolution (FCR %)", "Churn Rate (%)"])
            num_simulations = st.slider("Number of Simulations", min_value=10, max_value=20000, value=10000, step=10)
            
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

            # Correlation tables
            st.subheader("Correlations with FCR and Churn")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Top 5 Correlations with FCR")
                st.dataframe(correlation_fcr.head(6).drop(labels=['First Call Resolution (FCR %)']).to_frame('Correlation'))
            with col2:
                st.write("Top 5 Correlations with Churn Rate")
                st.dataframe(correlation_churn.head(6).drop(labels=['Churn Rate (%)']).to_frame('Correlation'))

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
