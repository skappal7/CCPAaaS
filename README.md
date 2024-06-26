# Call Center Performance Predictor 🎯📊

![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/call-center-performance-predictor)
![GitHub contributors](https://img.shields.io/github/contributors/yourusername/call-center-performance-predictor)
![GitHub stars](https://img.shields.io/github/stars/yourusername/call-center-performance-predictor?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/call-center-performance-predictor?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/call-center-performance-predictor)
![GitHub license](https://img.shields.io/github/license/yourusername/call-center-performance-predictor)

This Streamlit app allows users to predict call center performance metrics such as First Call Resolution (FCR) and Churn Rate. Users can input their current performance metrics and select the metric they want to improve. The app provides predictions, feature importance, model accuracy, and detailed model evaluation using Shapash.

## Features ✨

- **Model and Metric Selection**: Choose between improving FCR or Churn and select the model type (Gradient Boosting or Random Forest).
- **Input Performance Metrics**: Input your current performance metrics to get predictions.
- **Predictions**: Get predicted values for FCR or Churn based on your inputs.
- **Feature Importance**: View the feature importance for the selected model.
- **Model Accuracy**: See the R-squared or accuracy score of the selected model.
- **Model Evaluation with Shapash**: Generate detailed model evaluation reports using Shapash, including feature importance and contribution plots.

## Models 🧠

The following models are used in this app:
- Gradient Boosting for Churn Prediction: `best_gb_churn_model.pkl`
- Gradient Boosting for FCR Prediction: `best_gb_fcr_model.pkl`
- Random Forest for Churn Prediction: `best_rf_churn_model.pkl`
- Random Forest for FCR Prediction: `best_rf_fcr_model.pkl`

## Requirements 🛠️

Ensure you have the following dependencies installed. You can install them using the `requirements.txt` file:


## Installation 📥

1. Clone the repository:
    ```bash
    git clone https://github.com/skappal7/CCPAaaS.git
    ```
2. Navigate to the project directory:
    ```bash
    cd CCPAaaS
    ```
3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage 🚀

1. Save the models (`best_gb_churn_model.pkl`, `best_gb_fcr_model.pkl`, `best_rf_churn_model.pkl`, `best_rf_fcr_model.pkl`) in the root directory of the project.
2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3. Open the provided local URL in your web browser to access the app.

## Contributing 🤝

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements 🙏

- [Streamlit](https://streamlit.io/)
- [Shapash](https://github.com/MAIF/shapash)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.ai/)
