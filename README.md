# 🌐 Network Traffic Prediction System

This repository contains a Streamlit application that simulates network traffic data and uses Long Short-Term Memory (LSTM) neural networks to predict various network parameters. This tool demonstrates how machine learning, specifically LSTMs, can be applied to network management for predictive analytics and proactive decision-making.

## Features

* **Synthetic Data Generation**: Create realistic-looking network traffic data with configurable parameters, noise, and time-dependent patterns.
* **LSTM Model Configuration**: Define the architecture and hyperparameters of the LSTM neural network.
* **Model Training**: Train the LSTM model on the generated synthetic data with real-time progress visualization and loss curve plotting.
* **Network Traffic Prediction**: Generate predictions for selected network parameters and compare them against actual values.
* **Performance Analysis**: Evaluate model performance using key metrics such as MSE, RMSE, MAE, and MAPE.
* **Advanced Analysis**: Conduct error analysis, visualize residual distributions, analyze rolling performance, and perform simulated what-if scenarios.

## Technologies Used

* **Streamlit**: For creating the interactive web application.
* **PyTorch**: For building and training the LSTM neural network.
* **Pandas & NumPy**: For data manipulation and numerical operations.
* **Plotly & Matplotlib**: For data visualization and interactive plots.
* **Scikit-learn**: For data preprocessing (MinMaxScaler).

## Getting Started

Follow these instructions to set up and run the application locally.

### Prerequisites

Ensure you have Python 3.8 or higher installed.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/network-traffic-prediction.git](https://github.com/your-username/network-traffic-prediction.git)
    cd network-traffic-prediction
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages:**

    ```bash
    pip install streamlit numpy pandas torch matplotlib plotly scikit-learn
    ```
    Alternatively, you can create a `requirements.txt` file with the following content:
    ```
    streamlit>=1.0.0
    numpy>=1.20.0
    pandas>=1.3.0
    torch>=1.9.0
    matplotlib>=3.4.0
    plotly>=5.0.0
    scikit-learn>=1.0.0
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

After installing the dependencies, run the Streamlit application using the command:

```bash
streamlit run PRO3.py
How to Use
The application is structured into several tabs to guide you through the process:

📊 Data Generation:

Configure the number of data points and network parameters to simulate.

Optionally add random noise to the data.

Click "Generate Data" to create the synthetic dataset.

Preview the generated data, its statistics, and time-series visualizations.

⚙️ Model Configuration:

Select the specific network parameter you want the LSTM model to predict.

Set the Sliding Window Size (sequence length for LSTM input).

Define the Test Data Size (%) for splitting the dataset.

Configure LSTM architecture parameters: Number of Hidden Units, Number of LSTM Layers, and Dropout Rate.

Set training parameters like Number of Training Epochs, Learning Rate, Optimizer, and Batch Size.

Click "Configure Model" to preprocess data and initialize the LSTM model.

🧠 Training:

Review the model architecture and training data summary.

Click "Train Model" to start the training process. You will see a progress bar and loss updates.

After training, view the Training Loss Curve.

(Note: In this prototype, the model is saved to session state. In a real application, you would save it to disk.)

📈 Prediction:

Click "Generate Predictions" to apply the trained model to the test data.

Visualize the Actual vs Predicted values for the selected network parameter.

Review the prediction accuracy metrics: MSE, RMSE, MAE, and MAPE.

Expand to "View Prediction Results Table" for a detailed comparison.

📝 Analysis:

Error Analysis: Visualize the prediction errors over time and their distribution.

Forecast Analysis: Examine residual statistics and rolling MSE to understand model performance stability.

Feature Importance Simulation: See a simulated bar chart of relative importance for different network parameters (demonstrative only).

What-If Analysis: Simulate the impact of a percentage change in a selected network parameter on the predicted values.

Project Structure
PRO3.py: The main Streamlit application file containing all the UI, data generation, model definition, training, prediction, and analysis logic.

Limitations and Future Enhancements
Synthetic Data: The current version uses synthetically generated data. For real-world applications, actual network traffic data would be required.

Model Saving: In this prototype, the trained model is only stored in the Streamlit session state. For persistence, it should be saved to and loaded from disk.

Advanced Features: Future enhancements could include:

Integration with real-time network monitoring tools.

Anomaly detection capabilities.

More sophisticated feature engineering.

Comparison with other time-series forecasting models (e.g., ARIMA, Prophet).

Hyperparameter tuning features.
