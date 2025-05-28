import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, RobustScaler
import plotly.graph_objects as go
from datetime import date

# Streamlit app title
st.title("AI Powered Market  Predictions: AIMP")

# Sidebar for user inputs
st.sidebar.header("Prediction Parameters")
sequence_length = st.sidebar.slider("Sequence Length", min_value=50, max_value=200, value=100, step=10)
future_steps = st.sidebar.slider("Future Steps to Predict", min_value=10, max_value=60, value=40, step=5)
model_choice = st.sidebar.selectbox("Select Model", ["LSTM", "GRU", "Ensemble"])

# Function to fetch XAU/USD data
@st.cache_data
def get_xauusd_data():
    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    ticker = "GC=F"  # XAU/USD (Gold Futures)
    try:
        data = yf.download(ticker, start=START, end=TODAY, progress=False)
        if data.empty:
            raise ValueError("No data retrieved from yfinance for ticker GC=F.")
        return data[['Close']]
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Preprocessing function
def preprocess_data(data, sequence_length):
    if data is None or data.empty:
        raise ValueError("Input data is empty or None.")
    
    robust_scaler = RobustScaler()
    data_robust = robust_scaler.fit_transform(data)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_robust)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    
    X, y = np.array(X), np.array(y)
    if X.size == 0 or y.size == 0:
        raise ValueError("Not enough data to create sequences with the specified sequence length.")
    return X, y, scaler, robust_scaler

# Prediction function for LSTM or GRU
def predict_future_single(model, input_data, scaler, robust_scaler, future_steps):
    predicted_values = []
    current_sequence = input_data[-1].reshape(1, input_data.shape[1], input_data.shape[2])
    
    for _ in range(future_steps):
        next_value = model.predict(current_sequence, verbose=0)
        predicted_values.append(next_value[0])
        next_value_reshaped = next_value.reshape((1, 1, input_data.shape[2]))
        current_sequence = np.concatenate((current_sequence[:, 1:, :], next_value_reshaped), axis=1)
    
    predicted_values = np.array(predicted_values)
    predicted_values_inversed = robust_scaler.inverse_transform(scaler.inverse_transform(predicted_values))
    
    last_sequence_inversed = robust_scaler.inverse_transform(scaler.inverse_transform(input_data[-1]))
    last_real_price = last_sequence_inversed[-1]
    prediction_offset = last_real_price - predicted_values_inversed[0]
    predicted_values_original = predicted_values_inversed + prediction_offset
    
    return predicted_values_original.flatten()

# Prediction function for Ensemble
def predict_future_ensemble(lstm_model, gru_model, meta_model, input_data, scaler, robust_scaler, future_steps):
    predicted_values = []
    current_sequence = input_data[-1].reshape(1, input_data.shape[1], input_data.shape[2])
    
    for _ in range(future_steps):
        lstm_pred = lstm_model.predict(current_sequence, verbose=0)
        gru_pred = gru_model.predict(current_sequence, verbose=0)
        stacked_input = np.column_stack((lstm_pred, gru_pred))
        next_value = meta_model.predict(stacked_input, verbose=0)
        predicted_values.append(next_value[0, 0])
        next_input = np.append(current_sequence[:, 1:, :], [[[next_value[0, 0]]]], axis=1)
        current_sequence = next_input
    
    predicted_values = np.array(predicted_values).reshape(-1, 1)
    predicted_values_inversed = robust_scaler.inverse_transform(scaler.inverse_transform(predicted_values))
    
    last_sequence_inversed = robust_scaler.inverse_transform(scaler.inverse_transform(input_data[-1]))
    last_real_price = last_sequence_inversed[-1]
    prediction_offset = last_real_price - predicted_values_inversed[0]
    predicted_values_original = predicted_values_inversed + prediction_offset
    
    return predicted_values_original.flatten()

# Main app logic
if st.button("Run Prediction"):
    with st.spinner("Fetching data and making predictions..."):
        # Load data
        data = get_xauusd_data()
        if data is None:
            st.stop()
        st.write("Data Loaded")
        
        # Preprocess data
        try:
            X, y, scaler, robust_scaler = preprocess_data(data, sequence_length)
            st.write("Data Preprocessed")
        except ValueError as e:
            st.error(f"Error preprocessing data: {str(e)}")
            st.stop()
        
        # Load models
        try:
            lstm_model = load_model("./ensembling/lstm_model.keras")
            gru_model = load_model("./ensembling/gru_model.keras")
            meta_model = load_model("./ensembling/final_mlp_model.keras")
            st.write("Models Loaded")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()
        
        # Make predictions based on model choice
        try:
            if model_choice == "LSTM":
                future_predictions = predict_future_single(lstm_model, X, scaler, robust_scaler, future_steps)
                title = "XAU/USD Future Price Predictions (LSTM Model)"
            elif model_choice == "GRU":
                future_predictions = predict_future_single(gru_model, X, scaler, robust_scaler, future_steps)
                title = "XAU/USD Future Price Predictions (GRU Model)"
            else:  # Ensemble
                future_predictions = predict_future_ensemble(lstm_model, gru_model, meta_model, X, scaler, robust_scaler, future_steps)
                title = "XAU/USD Future Price Predictions (Meta Learner Ensemble Model)"
            
            st.write("Future Predictions:", future_predictions.tolist())
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Plot historical prices
            historical_prices = robust_scaler.inverse_transform(scaler.inverse_transform(y)).flatten()
            historical_dates = data.index[-len(y):]
            fig.add_trace(go.Scatter(
                x=historical_dates,
                y=historical_prices,
                mode='lines',
                name='Historical Prices',
                line=dict(color='blue')
            ))
            
            # Plot future predictions
            future_index = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='D')[1:]
            fig.add_trace(go.Scatter(
                x=future_index,
                y=future_predictions,
                mode='lines',
                name='Future Predictions',
                line=dict(color='red')
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_white",
                hovermode="x unified"
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")