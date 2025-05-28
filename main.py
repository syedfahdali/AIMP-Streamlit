import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, RobustScaler
import plotly.graph_objects as go
from datetime import date
import tensorflow as tf

# Initialize session state for storing results
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = []

# Streamlit app title
st.title("AI Powered Markets Prediction: AIMP")

# Sidebar for user inputs
st.sidebar.header("Prediction Parameters")
ticker = st.sidebar.selectbox("Select Asset", ["GC=F (Gold)", "BTC-USD (Bitcoin)", "ETH-USD (Etherium)" ,"EURUSD=X (Euro/USD)","AAPL (Apple Inc.)", "GOOGL (Alphabet Inc.)", "MSFT (Microsoft Corp.)"])
sequence_length = st.sidebar.slider("Sequence Length", min_value=50, max_value=200, value=100, step=10)
future_steps = st.sidebar.slider("Future Steps to Predict", min_value=10, max_value=60, value=40, step=5)
model_choice = st.sidebar.selectbox("Select Model", ["LSTM", "GRU", "Ensemble"])

# Display TensorFlow version for debugging
st.sidebar.write(f"TensorFlow Version: {tf.__version__}")

# Function to fetch data for selected ticker
@st.cache_data
def get_data(ticker_symbol):
    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    try:
        data = yf.download(ticker_symbol, start=START, end=TODAY, progress=False)
        if data.empty:
            raise ValueError(f"No data retrieved from yfinance for ticker {ticker_symbol}.")
        return data[['Close']]
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {str(e)}")
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

# Function to create Plotly figure
def create_plot(historical_dates, historical_prices, future_index, future_predictions, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_prices,
        mode='lines',
        name='Historical Prices',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=future_index,
        y=future_predictions,
        mode='lines',
        name='Future Predictions',
        line=dict(color='red')
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

# Main app logic
if st.button("Run Prediction"):
    with st.spinner("Fetching data and making predictions..."):
        # Extract ticker symbol from selection
        ticker_symbol = ticker.split()[0]  # e.g., "GC=F" from "GC=F (Gold)"
        
        # Load data
        data = get_data(ticker_symbol)
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
            lstm_model = load_model("./ensembling/lstm_model.keras", compile=False)
            gru_model = load_model("./ensembling/gru_model.keras", compile=False)
            meta_model = load_model("./ensembling/final_mlp_model.keras", compile=False)
            st.write("Models Loaded")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.error("This may be due to a TensorFlow or protobuf version mismatch. Try installing 'protobuf==3.20.3' or re-saving the models with your current TensorFlow version.")
            st.stop()
        
        # Make predictions based on model choice
        try:
            if model_choice == "LSTM":
                future_predictions = predict_future_single(lstm_model, X, scaler, robust_scaler, future_steps)
                title = f"{ticker} Future Price Predictions (LSTM Model)"
            elif model_choice == "GRU":
                future_predictions = predict_future_single(gru_model, X, scaler, robust_scaler, future_steps)
                title = f"{ticker} Future Price Predictions (GRU Model)"
            else:  # Ensemble
                future_predictions = predict_future_ensemble(lstm_model, gru_model, meta_model, X, scaler, robust_scaler, future_steps)
                title = f"{ticker} Future Price Predictions (Meta Learner Ensemble Model)"
            
            st.write("Future Predictions:", future_predictions.tolist())
            
            # Create and display plot
            historical_prices = robust_scaler.inverse_transform(scaler.inverse_transform(y)).flatten()
            historical_dates = data.index[-len(y):]
            future_index = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='D')[1:]
            fig = create_plot(historical_dates, historical_prices, future_index, future_predictions, title)
            st.plotly_chart(fig, use_container_width=True, key=f"current_plot_{ticker_symbol}_{model_choice}")
            
            # Store result in session state
            st.session_state.prediction_results.append({
                'ticker': ticker,
                'model': model_choice,
                'predictions': future_predictions.tolist(),
                'fig': fig,
                'title': title
            })
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("Ensure the models are compatible with your TensorFlow version and that sufficient data is available.")

# Display stored prediction results
st.header("Previous Prediction Results")
if st.session_state.prediction_results:
    for idx, result in enumerate(st.session_state.prediction_results):
        st.subheader(f"Prediction {idx + 1}: {result['ticker']} - {result['model']}")
        st.write("Future Predictions:", result['predictions'])
        # Use unique key for each stored plot
        st.plotly_chart(result['fig'], use_container_width=True, key=f"stored_plot_{idx}_{result['ticker']}_{result['model']}")
else:
    st.write("No predictions yet. Run a prediction to see results here.")

# Button to clear results
if st.button("Clear Previous Results"):
    st.session_state.prediction_results = []
    st.rerun()