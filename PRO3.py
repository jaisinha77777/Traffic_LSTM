import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random

# Set page configuration
st.set_page_config(
    page_title="Network Traffic Prediction",
    page_icon="📊",
    layout="wide"
)

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to generate synthetic network traffic data
def generate_synthetic_data(num_samples, feature_count):
    # Define network parameters with realistic ranges
    network_params = {
        "packet_count": (100, 5000),  # packets per interval
        "throughput": (10, 1000),     # Mbps
        "latency": (5, 200),          # ms
        "packet_size": (64, 1500),    # bytes
        "packet_loss": (0, 5),        # percentage
        "jitter": (1, 50),            # ms
        "bandwidth_utilization": (10, 95), # percentage
        "connection_count": (10, 1000), # number of connections
        "retransmission_rate": (0, 10), # percentage
        "error_rate": (0, 3),         # percentage
        "cpu_utilization": (5, 95),   # percentage
        "memory_utilization": (10, 90), # percentage
        "queue_depth": (0, 100),      # packets in queue
        "dropped_packets": (0, 200),  # packets dropped
        "collision_rate": (0, 8)      # percentage
    }
    
    # Select parameters based on feature_count
    selected_params = list(network_params.keys())[:feature_count]
    
    # Create time index
    time_index = pd.date_range(start='2025-01-01', periods=num_samples, freq='1min')
    
    # Create empty dataframe
    df = pd.DataFrame(index=time_index)
    
    # Generate data for each parameter with some patterns and seasonality
    for param in selected_params:
        min_val, max_val = network_params[param]
        
        # Base signal
        base = np.random.uniform(min_val, max_val, num_samples)
        
        # Add time-dependent patterns
        time_pattern = np.sin(np.linspace(0, 4*np.pi, num_samples)) * (max_val - min_val) * 0.3
        
        # Add some random peaks (to simulate network events)
        peaks = np.zeros(num_samples)
        num_peaks = int(num_samples * 0.05)  # 5% of data points have peaks
        peak_indices = np.random.choice(range(num_samples), size=num_peaks, replace=False)
        peaks[peak_indices] = np.random.uniform(0.5, 1.0, size=num_peaks) * (max_val - min_val)
        
        # Combine signals
        signal = base + time_pattern + peaks
        
        # Make sure values are within range
        signal = np.clip(signal, min_val, max_val)
        
        # Add to dataframe
        df[param] = signal
    
    return df

# Function to create sliding windows from data
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Function to train LSTM model
def train_model(model, train_X, train_y, epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert numpy arrays to PyTorch tensors
    train_X = torch.FloatTensor(train_X).to(device)
    train_y = torch.FloatTensor(train_y).to(device)
    
    # Training loop
    losses = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Training Progress: {progress*100:.2f}% - Loss: {loss.item():.6f}")
        
    return model, losses

# Function to make predictions
def predict(model, test_X, device):
    model.eval()
    with torch.no_grad():
        test_X = torch.FloatTensor(test_X).to(device)
        predictions = model(test_X)
    return predictions.cpu().numpy()

# Function to evaluate model
def evaluate_model(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# App title
st.title('🌐 Network Traffic Prediction System')
st.markdown("""
This application simulates network traffic data and uses LSTM neural networks 
to predict various network parameters. Use the tabs below to navigate through 
different functionalities.
""")

# Create tabs
tabs = st.tabs(["📊 Data Generation", "⚙️ Model Configuration", "🧠 Training", "📈 Prediction", "📝 Analysis"])

# Global session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'training_losses' not in st.session_state:
    st.session_state.training_losses = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'feature_to_predict' not in st.session_state:
    st.session_state.feature_to_predict = None

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TAB 1: DATA GENERATION
with tabs[0]:
    st.header("Network Traffic Data Generation")
    
    with st.form("data_generation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            num_samples = st.slider("Number of Data Points", min_value=100, max_value=5000, value=1000, step=100)
            feature_count = st.slider("Number of Network Parameters", min_value=5, max_value=15, value=10, step=1)
        
        with col2:
            add_noise = st.checkbox("Add Random Noise", value=True)
            noise_level = st.slider("Noise Level (%)", min_value=0, max_value=20, value=5, step=1, disabled=not add_noise)
        
        generate_button = st.form_submit_button("Generate Data")
        
        if generate_button:
            with st.spinner("Generating synthetic network traffic data..."):
                # Generate data
                data = generate_synthetic_data(num_samples, feature_count)
                
                # Add noise if selected
                if add_noise:
                    for col in data.columns:
                        noise = np.random.normal(0, data[col].std() * noise_level/100, size=len(data))
                        data[col] = data[col] + noise
                        data[col] = np.clip(data[col], 0, None)  # Ensure no negative values
                
                st.session_state.data = data
                st.success(f"Successfully generated {num_samples} data points for {feature_count} network parameters!")
    
    if st.session_state.data is not None:
        st.subheader("Generated Data Preview")
        st.dataframe(st.session_state.data.head(10))
        
        st.subheader("Data Statistics")
        st.dataframe(st.session_state.data.describe())
        
        st.subheader("Data Visualization")
        
        # Allow user to select parameters to visualize
        selected_params = st.multiselect(
            "Select parameters to visualize",
            options=st.session_state.data.columns.tolist(),
            default=st.session_state.data.columns.tolist()[:3]
        )
        
        if selected_params:
            # Create time series plot
            fig = make_subplots(rows=len(selected_params), cols=1, 
                                subplot_titles=selected_params,
                                shared_xaxes=True,
                                vertical_spacing=0.05)
            
            for i, param in enumerate(selected_params):
                fig.add_trace(
                    go.Scatter(x=st.session_state.data.index, y=st.session_state.data[param], name=param),
                    row=i+1, col=1
                )
            
            fig.update_layout(height=250*len(selected_params), width=800, 
                             title_text="Network Parameters Over Time",
                             showlegend=False)
            
            st.plotly_chart(fig)

# TAB 2: MODEL CONFIGURATION
with tabs[1]:
    st.header("LSTM Model Configuration")
    
    if st.session_state.data is None:
        st.warning("Please generate data in the 'Data Generation' tab first.")
    else:
        with st.form("model_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Configuration")
                
                # Feature selection
                feature_to_predict = st.selectbox(
                    "Select Parameter to Predict", 
                    options=st.session_state.data.columns.tolist(),
                    index=0
                )
                
                window_size = st.slider("Sliding Window Size (time steps)", 
                                     min_value=5, max_value=120, value=30, step=5)
                
                test_size = st.slider("Test Data Size (%)", 
                                   min_value=10, max_value=40, value=20, step=5)
                
                st.subheader("Training Configuration")
                epochs = st.slider("Number of Training Epochs", 
                                min_value=10, max_value=500, value=100, step=10)
                
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                    value=0.001
                )
            
            with col2:
                st.subheader("LSTM Architecture")
                hidden_size = st.slider("Number of Hidden Units", 
                                     min_value=10, max_value=256, value=64, step=8)
                
                num_layers = st.slider("Number of LSTM Layers", 
                                    min_value=1, max_value=3, value=1, step=1)
                
                dropout = st.slider("Dropout Rate", 
                                 min_value=0.0, max_value=0.5, value=0.2, step=0.1)
                
                st.subheader("Additional Settings")
                optimizer_choice = st.selectbox(
                    "Optimizer",
                    options=["Adam", "SGD", "RMSprop"],
                    index=0
                )
                
                batch_size = st.select_slider(
                    "Batch Size",
                    options=[8, 16, 32, 64, 128, 256],
                    value=32
                )
            
            configure_button = st.form_submit_button("Configure Model")
            
            if configure_button:
                # Store configuration in session state
                st.session_state.feature_to_predict = feature_to_predict
                st.session_state.window_size = window_size
                st.session_state.test_size = test_size
                st.session_state.epochs = epochs
                st.session_state.learning_rate = learning_rate
                st.session_state.hidden_size = hidden_size
                st.session_state.num_layers = num_layers
                st.session_state.dropout = dropout
                st.session_state.optimizer_choice = optimizer_choice
                st.session_state.batch_size = batch_size
                
                # Preprocess the data
                data = st.session_state.data[feature_to_predict].values.reshape(-1, 1)
                
                # Scale the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                data_scaled = scaler.fit_transform(data)
                st.session_state.scaler = scaler
                
                # Create sequences
                X, y = create_sequences(data_scaled, window_size)
                
                # Split into train and test sets
                train_size = int(len(X) * (1 - test_size/100))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                # Store preprocessed data
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                # Initialize model
                model = LSTMModel(
                    input_size=1, 
                    hidden_size=hidden_size,
                    output_size=1,
                    num_layers=num_layers
                ).to(device)
                
                st.session_state.model = model
                st.success("Model configured successfully! You can now proceed to the Training tab.")

# TAB 3: TRAINING
with tabs[2]:
    st.header("Model Training")
    
    if st.session_state.model is None:
        st.warning("Please configure the model in the 'Model Configuration' tab first.")
    else:
        st.subheader("Train LSTM Model")
        
        # Display model architecture
        with st.expander("Model Architecture", expanded=False):
            st.code(str(st.session_state.model))
        
        # Display training data summary
        with st.expander("Training Data Summary", expanded=False):
            st.write(f"Training samples: {len(st.session_state.X_train)}")
            st.write(f"Testing samples: {len(st.session_state.X_test)}")
            st.write(f"Input shape: {st.session_state.X_train.shape}")
            st.write(f"Output shape: {st.session_state.y_train.shape}")
        
        train_button = st.button("Train Model")
        
        if train_button:
            with st.spinner("Training LSTM model..."):
                # Train model
                model, losses = train_model(
                    model=st.session_state.model,
                    train_X=st.session_state.X_train,
                    train_y=st.session_state.y_train,
                    epochs=st.session_state.epochs,
                    learning_rate=st.session_state.learning_rate,
                    device=device
                )
                
                st.session_state.model = model
                st.session_state.training_losses = losses
                
                st.success("Model training completed!")
        
        if st.session_state.training_losses is not None:
            st.subheader("Training Loss Curve")
            
            # Plot loss curve
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(st.session_state.training_losses)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.set_title('Training Loss Over Epochs')
            ax.grid(True)
            
            st.pyplot(fig)
            
            # Save model button
            if st.button("Save Trained Model"):
                # In a real application, you would save the model to a file
                st.success("Model saved successfully! (Note: In this prototype, models are stored in session state only)")

# TAB 4: PREDICTION
with tabs[3]:
    st.header("Network Traffic Prediction")
    
    if st.session_state.model is None or st.session_state.training_losses is None:
        st.warning("Please train the model in the 'Training' tab first.")
    else:
        st.subheader(f"Predict {st.session_state.feature_to_predict}")
        
        predict_button = st.button("Generate Predictions")
        
        if predict_button:
            with st.spinner("Generating predictions..."):
                # Make predictions
                test_predictions = predict(
                    model=st.session_state.model,
                    test_X=st.session_state.X_test,
                    device=device
                )
                
                # Inverse transform predictions to original scale
                test_predictions_rescaled = st.session_state.scaler.inverse_transform(test_predictions)
                
                # Inverse transform actual values to original scale
                y_test_rescaled = st.session_state.scaler.inverse_transform(st.session_state.y_test)
                
                # Store predictions
                st.session_state.predictions = test_predictions_rescaled
                st.session_state.actual = y_test_rescaled
                
                # Calculate forecast horizon
                forecast_start = len(st.session_state.data) - len(test_predictions_rescaled)
                forecast_end = len(st.session_state.data)
                
                # Create dataframe for visualization
                results_df = pd.DataFrame({
                    'Timestamp': st.session_state.data.index[forecast_start:forecast_end],
                    'Actual': y_test_rescaled.flatten(),
                    'Predicted': test_predictions_rescaled.flatten()
                })
                
                st.session_state.results_df = results_df
                
                st.success("Predictions generated successfully!")
        
        if 'results_df' in st.session_state:
            st.subheader("Prediction Results")
            
            # Plot predictions vs actual
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=st.session_state.results_df['Timestamp'],
                y=st.session_state.results_df['Actual'],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=st.session_state.results_df['Timestamp'],
                y=st.session_state.results_df['Predicted'],
                mode='lines',
                name='Predicted',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f'{st.session_state.feature_to_predict} - Actual vs Predicted',
                xaxis_title='Time',
                yaxis_title=st.session_state.feature_to_predict,
                legend=dict(x=0, y=1, traceorder='normal'),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction accuracy metrics
            metrics = evaluate_model(
                st.session_state.results_df['Actual'].values,
                st.session_state.results_df['Predicted'].values
            )
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", f"{metrics['MSE']:.4f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
            col3.metric("MAE", f"{metrics['MAE']:.4f}")
            col4.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            
            # Display results table
            with st.expander("View Prediction Results Table", expanded=False):
                st.dataframe(st.session_state.results_df)

# TAB 5: ANALYSIS
with tabs[4]:
    st.header("Advanced Analysis")
    
    if 'results_df' not in st.session_state:
        st.warning("Please generate predictions in the 'Prediction' tab first.")
    else:
        st.subheader("Error Analysis")
        
        # Calculate error
        error = st.session_state.results_df['Actual'] - st.session_state.results_df['Predicted']
        st.session_state.results_df['Error'] = error
        
        # Error distribution
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=["Error Over Time", "Error Distribution"],
                           vertical_spacing=0.15)
        
        # Error over time
        fig.add_trace(
            go.Scatter(x=st.session_state.results_df['Timestamp'], y=error, mode='lines', name='Error'),
            row=1, col=1
        )
        
        # Error distribution histogram
        fig.add_trace(
            go.Histogram(x=error, nbinsx=30, name='Error Distribution'),
            row=2, col=1
        )
        
        fig.update_layout(height=700, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast Analysis
        st.subheader("Forecast Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residual analysis
            residuals = st.session_state.results_df['Error']
            
            st.write("Residual Statistics:")
            residual_stats = {
                "Mean Error": residuals.mean(),
                "Std Dev": residuals.std(),
                "Min Error": residuals.min(),
                "Max Error": residuals.max(),
                "25th Percentile": residuals.quantile(0.25),
                "Median Error": residuals.median(),
                "75th Percentile": residuals.quantile(0.75)
            }
            
            st.dataframe(pd.DataFrame([residual_stats]).T.rename(columns={0: "Value"}))
        
        with col2:
            # Performance over time
            window_size = st.slider("Rolling Window Size", 5, 50, 10)
            
            rolling_mse = (st.session_state.results_df['Error'] ** 2).rolling(window=window_size).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.results_df['Timestamp'][window_size-1:],
                y=rolling_mse[window_size-1:],
                mode='lines',
                name=f'Rolling MSE (window={window_size})'
            ))
            
            fig.update_layout(
                title=f'Model Performance Over Time (Rolling MSE, window={window_size})',
                xaxis_title='Time',
                yaxis_title='MSE',
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance simulation (for demonstration)
        st.subheader("Feature Importance Simulation")
        st.info("Note: This is a simulated feature importance analysis for demonstration purposes.")
        
        # Simulated feature importance based on network parameters
        if st.session_state.data is not None:
            features = st.session_state.data.columns.tolist()
            importance = np.random.uniform(0, 1, size=len(features))
            importance = importance / importance.sum()  # Normalize
            
            # Sort by importance
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=feature_importance['Feature'],
                y=feature_importance['Importance'],
                marker_color='royalblue'
            ))
            
            fig.update_layout(
                title='Feature Importance (Simulated)',
                xaxis_title='Network Parameter',
                yaxis_title='Relative Importance',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # What-if analysis
        st.subheader("What-If Analysis")
        
        # Allow user to modify parameters and see potential impact
        with st.form("what_if_form"):
            st.write("Modify network parameters to see potential impact on predictions")
            
            parameter_to_modify = st.selectbox(
                "Select parameter to modify",
                options=["packet_count", "throughput", "latency"] if st.session_state.data is None else st.session_state.data.columns.tolist(),
                index=0
            )
            
            change_percentage = st.slider(
                "Change percentage",
                min_value=-50,
                max_value=50,
                value=10,
                step=5
            )
            
            what_if_button = st.form_submit_button("Run Analysis")
            
            if what_if_button:
                st.info(f"Simulating {change_percentage}% change in {parameter_to_modify}")
                
                # For demonstration, show simulated impact
                impact_factor = 1 + (change_percentage / 100)
                
                # Create modified predictions
                modified_predictions = st.session_state.results_df['Predicted'] * impact_factor
                
                # Plot comparison
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=st.session_state.results_df['Timestamp'],
                    y=st.session_state.results_df['Predicted'],
                    mode='lines',
                    name='Original Prediction',
                    line=dict(color='red')
                ))
                
                fig.add_trace(go.Scatter(
                    x=st.session_state.results_df['Timestamp'],
                    y=modified_predictions,
                    mode='lines',
                    name=f'Modified Prediction ({change_percentage}% change)',
                    line=dict(color='green')
                ))
                
                fig.update_layout(
                    title=f'What-If Analysis: Impact of {change_percentage}% change in {parameter_to_modify}',
                    xaxis_title='Time',
                    yaxis_title=st.session_state.feature_to_predict,
                    legend=dict(x=0, y=1),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
---
### About this Application

This Network Traffic Prediction System uses Long Short-Term Memory (LSTM) neural networks to analyze and predict 
network traffic parameters based on historical data. The application demonstrates how machine learning can be 
applied to network management for predictive analytics.

**Features:**
- Generate synthetic network traffic data with realistic patterns
- Configure and train LSTM models for time-series prediction
- Visualize predictions and analyze model performance
- Perform what-if analysis to simulate parameter changes

**Note:** This is a demonstration application using synthetic data. In a production environment, real network 
data would be used, and additional features like anomaly detection could be implemented.
""")