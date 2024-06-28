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

# Function to calculate risk metrics
def calculate_risk_metrics(simulations, confidence_level=0.95):
    VaR = np.percentile(simulations, (1 - confidence_level) * 100)
    CVaR = np.mean([x for x in simulations if x <= VaR])
    return VaR, CVaR

# Function to visualize Monte Carlo results
def visualize_monte_carlo(simulations, target):
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))

    # Histogram
    sns.histplot(simulations, kde=True, bins=50, ax=ax[0, 0])
    ax[0, 0].set_title(f'Histogram of {target}')
    ax[0, 0].set_xlabel(f'{target} (%)')
    ax[0, 0].set_ylabel('Frequency')

    # Density Plot
    sns.kdeplot(simulations, ax=ax[0, 1])
    ax[0, 1].set_title(f'Density Plot of {target}')
    ax[0, 1].set_xlabel(f'{target} (%)')
    ax[0, 1].set_ylabel('Density')

    # Box Plot
    sns.boxplot(x=simulations, ax=ax[0, 2])
    ax[0, 2].set_title(f'Boxplot of {target}')
    ax[0, 2].set_xlabel(f'{target} (%)')

    # Cumulative Distribution Function (CDF)
    sns.ecdfplot(simulations, ax=ax[1, 0])
    ax[1, 0].set_title(f'Cumulative Distribution of {target}')
    ax[1, 0].set_xlabel(f'{target} (%)')
    ax[1, 0].set_ylabel('Cumulative Probability')

    # Line Plot of Simulated Paths (Sampled Paths)
    for i in range(10):
        sampled_path = np.random.choice(simulations, size=len(simulations))
        ax[1, 1].plot(sampled_path, alpha=0.3)
    ax[1, 1].set_title(f'Simulated Paths of {target}')
    ax[1, 1].set_xlabel('Simulation Index')
    ax[1, 1].set_ylabel(f'{target} (%)')

    st.pyplot(fig)

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

# Fetch data directly from GitHub
url = "https://raw.githubusercontent.com/skappal7/CCPAaaS/main/Call%20Center%20Data%202022%20-%202024.csv"
data = pd.read_csv(url)

# Filter data based on the selected industry
data = data[data['Industry'] == industry]
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
        st.metric("Your FCR", f"{current_fcr:.2f}%", f"{fcr_delta:.2f}% from industry median")
        st.metric("Industry FCR", f"{benchmark_fcr:.2f}%")
    with col2:
        # Custom display for churn rate
        st.write("Your Churn Rate")
        st.write(f"<span style='font-size:24px; font-weight:bold;'>{current_churn:.2f}%</span>", unsafe_allow_html=True)
        if churn_delta < 0:
            st.write(f"<span style='color:green;'>▼ {abs(churn_delta):.2f}% below industry median</span>", unsafe_allow_html=True)
        elif churn_delta > 0:
            st.write(f"<span style='color:red;'>▲ {churn_delta:.2f}% above industry median</span>", unsafe_allow_html=True)
        else:
            st.write("At industry median")
        st.metric("Industry Churn Rate", f"{benchmark_churn:.2f}%")

    # Information box about data source
    st.info("The underlying data has been curated from reports from platforms like Talkdesk, LiveAgent, and Sprinklr from years 2021 to 2024.")

    # Monte Carlo Simulation for FCR and Churn Prediction
    st.subheader("Monte Carlo Simulation")
    
    # 1. User Guidance
    with st.expander("What is Monte Carlo Simulation?"):
        st.write("""
        Monte Carlo simulation is a statistical technique that uses random sampling to obtain numerical results. 
        In this context, it helps predict possible outcomes for FCR and Churn Rate based on the input data and industry trends.
        The simulation runs thousands of scenarios to provide a range of possible outcomes and their probabilities.
        """)

    target_variable = st.selectbox("Select Target Variable for Simulation", options=["First Call Resolution (FCR %)", "Churn Rate (%)"])
    
    # 2. Interactivity
    num_simulations = st.slider("Number of Simulations", min_value=10, max_value=20000, value=10000, step=10)
    confidence_level = st.slider("Confidence Level for Risk Metrics", min_value=0.8, max_value=0.99, value=0.95, step=0.01)

    # 3. Scenario Analysis
    st.subheader("Scenario Analysis")
    scenario = st.radio("Choose a scenario:", ["Current", "Optimistic", "Pessimistic"])
    if scenario == "Optimistic":
        data = data * 1.1  # Improve all metrics by 10%
    elif scenario == "Pessimistic":
        data = data * 0.9  # Worsen all metrics by 10%

    # 4. Data Input Flexibility
    st.subheader("Custom Data Input")
    use_custom_data = st.checkbox("Use custom data")
    if use_custom_data:
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            data = process_data(data)

    if st.button("Run Monte Carlo Simulation"):
        # 9. Error Handling
        try:
            with st.spinner("Running Monte Carlo simulations..."):
                simulations = monte_carlo_simulation(data, target_variable, num_simulations)
                visualize_monte_carlo(simulations, target_variable)
                
                # Provide summary and insights
                st.subheader("Simulation Summary and Insights")
                mean_sim = np.mean(simulations)
                median_sim = np.median(simulations)
                std_sim = np.std(simulations)
                min_sim = np.min(simulations)
                max_sim = np.max(simulations)
                percentiles = np.percentile(simulations, [5, 25, 50, 75, 95])
                ci_low, ci_high = np.percentile(simulations, [2.5, 97.5])
                probability = np.mean(np.array(simulations) > current_fcr if target_variable == "First Call Resolution (FCR %)" else current_churn) * 100
                VaR, CVaR = calculate_risk_metrics(simulations, confidence_level)

                insights = [
                    f"The average simulated {target_variable} is {mean_sim:.2f}%.",
                    f"The median simulated {target_variable} is {median_sim:.2f}%.",
                    f"The standard deviation of the simulated {target_variable} is {std_sim:.2f}.",
                    f"The minimum simulated {target_variable} is {min_sim:.2f}%.",
                    f"The maximum simulated {target_variable} is {max_sim:.2f}%.",
                    f"The 95% confidence interval for {target_variable} is ({ci_low:.2f}%, {ci_high:.2f}%).",
                    f"The probability of achieving more than the current {target_variable} is {probability:.2f}%.",
                    f"The Value at Risk (VaR) at {confidence_level*100:.0f}% confidence level is {VaR:.2f}%.",
                    f"The Conditional Value at Risk (CVaR) at {confidence_level*100:.0f}% confidence level is {CVaR:.2f}%."
                ]

                st.markdown(
                    '<ul style="color: lightgrey; font-style: italic;">' +
                    ''.join([f'<li>{insight}</li>' for insight in insights]) +
                    '</ul>',
                    unsafe_allow_html=True
                )

                st.subheader(f"Percentiles of the Simulated {target_variable}")
                percentiles_df = pd.DataFrame({
                    'Percentile': ['5th', '25th', '50th', '75th', '95th'],
                    'Value (%)': [f"{p:.2f}%" for p in percentiles]
                })
                st.table(percentiles_df)

                # 5. Recommendations Engine
                st.subheader("Recommendations")
                if target_variable == "First Call Resolution (FCR %)":
                    if mean_sim < current_fcr:
                        st.warning("Your FCR is predicted to decrease. Consider the following actions:")
                        st.write("- Improve agent training programs")
                        st.write("- Enhance knowledge base and self-service options")
                        st.write("- Implement advanced call routing to match customers with the most suitable agents")
                    else:
                        st.success("Your FCR is predicted to improve. To maintain this trend:")
                        st.write("- Continue monitoring and optimizing agent performance")
                        st.write("- Regularly update your knowledge base")
                        st.write("- Collect and act on customer feedback")
                else:  # Churn Rate
                    if mean_sim > current_churn:
                        st.warning("Your Churn Rate is predicted to increase. Consider the following actions:")
                        st.write("- Implement a proactive customer retention program")
                        st.write("- Improve product/service quality based on customer feedback")
                        st.write("- Enhance customer support and engagement strategies")
                    else:
                        st.success("Your Churn Rate is predicted to decrease. To maintain this trend:")
                        st.write("- Continue focusing on customer satisfaction and loyalty programs")
                        st.write("- Regularly assess and improve your retention strategies")
                        st.write("- Monitor competitors and stay ahead in your offerings")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}. Please check your input data and try again.")

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
