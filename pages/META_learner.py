import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
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
def load_and_preprocess_data(ticker, look_back=100):
    try:
        data = yf.download(ticker, start=START, end=TODAY)
        if data.empty:
            st.error(f"No data found for ticker: {ticker}. Please check the ticker symbol.")
            return None, None, None, None
        
        df = data.copy()
        if 'Close' not in df.columns:
            st.error("The fetched data must contain a 'Close' column.")
            return None, None, None, None

        close_prices = df['Close'].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        X = []
        y = []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        # For LSTM/GRU, reshape to (samples, timesteps, features)
        # For meta-learner, the input will be (samples, 2)
        X_reshaped_for_rnn = np.reshape(X, (X.shape[0], X.shape[1], 1)) 

        return X_reshaped_for_rnn, y, scaler, df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None, None, None

st.title("Meta-Learner (Ensembling) Model for Stock Price Prediction")
st.write("Select a ticker to get predictions using the Meta-Learner model.")

# Ticker selection
selected_ticker = st.selectbox(
    "Choose a stock ticker:",
    ("AAPL", "MSFT", "GOOG", "BTC-USD", "GC=F", "EURUSD=X") # Added more reliable tickers
)
st.info("Note: Some tickers like 'BTC-USD', 'GC=F', 'EURUSD=X' might have issues fetching data via yfinance due to delisting or timezone errors. Please try 'AAPL', 'MSFT', or 'GOOG' if you encounter issues.")

if selected_ticker:
    X_test_rnn, y_test, scaler, original_df = load_and_preprocess_data(selected_ticker)

    if X_test_rnn is not None:
        st.subheader(f"Original Data for {selected_ticker} (Head)")
        st.write(original_df.head())

        st.subheader(f"Making Predictions with Meta-Learner Model for {selected_ticker}...")
        try:
            # Load base models (LSTM and GRU)
            lstm_model_path = "models/lstm_model.keras"
            if not os.path.exists(lstm_model_path):
                lstm_model_path = "ensembling/lstm_model.keras"
            
            gru_model_path = "models/gru_model.keras"
            if not os.path.exists(gru_model_path):
                gru_model_path = "ensembling/gru_model.keras"

            meta_model_path = "ensembling/final_mlp_model.keras" # Prioritize MLP meta-learner
            if not os.path.exists(meta_model_path):
                meta_model_path = "ensembling/final_attention_model.keras" # Fallback to attention model

            if os.path.exists(lstm_model_path) and os.path.exists(gru_model_path) and os.path.exists(meta_model_path):
                lstm_model = load_model(lstm_model_path)
                gru_model = load_model(gru_model_path)
                meta_model = load_model(meta_model_path)

                # Generate predictions from base models
                lstm_preds_scaled = lstm_model.predict(X_test_rnn)
                gru_preds_scaled = gru_model.predict(X_test_rnn)

                # Stack predictions for meta-learner input
                stacked_preds_scaled = np.column_stack((lstm_preds_scaled, gru_preds_scaled))

                # Make final prediction with meta-learner
                final_predictions_scaled = meta_model.predict(stacked_preds_scaled)
                final_predictions = scaler.inverse_transform(final_predictions_scaled)
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

                st.subheader("Prediction Results")

                fig = plt.figure(figsize=(12, 6))
                plt.plot(y_test_actual, 'b', label='Original Price')
                plt.plot(final_predictions, 'g', label='Meta-Learner Predicted Price')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.title(f'Meta-Learner Model: Original vs. Predicted Price for {selected_ticker}')
                plt.legend()
                st.pyplot(fig)

                from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

                mse = mean_squared_error(y_test_actual, final_predictions)
                mae = mean_absolute_error(y_test_actual, final_predictions)
                mape = mean_absolute_percentage_error(y_test_actual, final_predictions)

                st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
                st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
                st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.4f}")

            else:
                st.error("One or more models (LSTM, GRU, or Meta-Learner) not found. Please ensure 'lstm_model.keras', 'gru_model.keras', and 'final_mlp_model.keras' (or 'final_attention_model.keras') are in the 'models' or 'ensembling' directory.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.info("Please select a ticker to proceed with Meta-Learner prediction.")
