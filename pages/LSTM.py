import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
import os
from keras.initializers import Orthogonal # Import Orthogonal initializer

# Define start and end dates for yfinance data
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Function to load and preprocess data from yfinance
@st.cache_data
def load_and_preprocess_data(ticker, sequence_length=100):
    try:
        data = yf.download(ticker, start=START, end=TODAY)
        if data.empty:
            st.error(f"No data found for ticker: {ticker}. Please check the ticker symbol.")
            return None, None, None, None, None
        
        df = data.copy()
        if 'Close' not in df.columns:
            st.error("The fetched data must contain a 'Close' column.")
            return None, None, None, None, None

        close_prices = df['Close'].values.reshape(-1, 1)

        # First scale using RobustScaler to reduce outlier influence
        robust_scaler = RobustScaler()
        data_robust = robust_scaler.fit_transform(close_prices)
        
        # Then standardize using StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_robust)
        
        # Prepare sequences for prediction
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i])
        
        X, y = np.array(X), np.array(y)
        # Reshape for LSTM input (samples, timesteps, features)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1)) 

        return X, y, scaler, robust_scaler, df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None, None, None, None

def predict_future(model, input_data, scaler, robust_scaler, future_steps=10):
    predicted_values = []
    # Start with the last sequence in the input data
    current_sequence = input_data[-1].reshape(1, input_data.shape[1], input_data.shape[2])

    for _ in range(future_steps):
        next_value = model.predict(current_sequence, verbose=0)  # Predict next value
        predicted_values.append(next_value[0])         # Save predicted value
        
        # Update current_sequence by appending the prediction and dropping the oldest entry
        next_value_reshaped = next_value.reshape((1, 1, input_data.shape[2]))
        current_sequence = np.concatenate((current_sequence[:, 1:, :], next_value_reshaped), axis=1)

    # Convert predictions back to original scale by inverting both scalers in reverse order:
    # 1. Invert StandardScaler, then 2. Invert RobustScaler.
    predicted_values = np.array(predicted_values)
    predicted_values_inversed = robust_scaler.inverse_transform(scaler.inverse_transform(predicted_values))

    # Adjust predictions so that the first prediction attaches to the last real price
    # First, obtain the last known price in the original scale:
    last_sequence_inversed = robust_scaler.inverse_transform(scaler.inverse_transform(input_data[-1]))
    last_real_price = last_sequence_inversed[-1]
    
    # Compute the offset between the last real price and the first prediction
    prediction_offset = last_real_price - predicted_values_inversed[0]
    predicted_values_original = predicted_values_inversed + prediction_offset

    return predicted_values_original

st.title("LSTM Model for Stock Price Prediction")
st.write("Select a ticker to get predictions using the LSTM model.")

# Ticker selection
selected_ticker = st.selectbox(
    "Choose a stock ticker:",
    ("BTC-USD", "GC=F", "AAPL", "MSFT", "GOOG", "EURUSD=X") # Prioritize trained tickers
)
st.info("Note: This model was primarily trained on BTC-USD and GC=F (Gold Futures). Predictions for other tickers might not be accurate.")

if selected_ticker:
    X, y, scaler, robust_scaler, original_df = load_and_preprocess_data(selected_ticker)

    if X is not None:
        st.subheader(f"Original Data for {selected_ticker} (Head)")
        st.write(original_df.head())

        st.subheader(f"Generating Future Predictions with LSTM Model for {selected_ticker}...")
        try:
            # Load the LSTM model
            lstm_model_path = "models/lstm_model.keras" # Assuming model is in 'models' directory
            if not os.path.exists(lstm_model_path):
                lstm_model_path = "ensembling/lstm_model.keras" # Check ensembling directory too
            
            if os.path.exists(lstm_model_path):
                model = load_model(lstm_model_path)
                
                future_steps = st.slider("Number of future steps to predict:", 10, 90, 30)
                future_predictions = predict_future(model, X, scaler, robust_scaler, future_steps)

                st.subheader("Future Prediction Results")

                fig = plt.figure(figsize=(12, 6))
                
                # Invert both scalers for historical y-values to get original prices
                historical_prices = robust_scaler.inverse_transform(scaler.inverse_transform(y))
                plt.plot(original_df.index[-len(y):], historical_prices, label="Historical Prices")
                
                # Create a date range for the future predictions, starting right after the last available date
                future_index = pd.date_range(start=original_df.index[-1], periods=future_steps + 1, freq='D')[1:]
                plt.plot(future_index, future_predictions, label="Future Predictions", color="red")
                
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.title(f"LSTM Model: Historical and Future Price Predictions for {selected_ticker}")
                plt.legend()
                st.pyplot(fig)

                # Displaying the last few historical prices and the first few future predictions
                st.subheader("Last Historical Prices and First Future Predictions")
                last_historical_date = original_df.index[-1].strftime('%Y-%m-%d')
                st.write(f"Last Historical Price ({last_historical_date}): {historical_prices[-1][0]:.2f}")
                
                for i, pred_val in enumerate(future_predictions.flatten()):
                    st.write(f"Prediction for {future_index[i].strftime('%Y-%m-%d')}: {pred_val:.2f}")

            else:
                st.error(f"LSTM model not found at {lstm_model_path}. Please ensure 'lstm_model.keras' is in the 'models' or 'ensembling' directory.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.info("Please select a ticker to proceed with LSTM prediction.")
