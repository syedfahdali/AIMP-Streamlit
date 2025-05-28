import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
import plotly.graph_objects as go

st.title("GRU Prediction")

# Load Data
STOCK_LIST = "EQUITY_L.csv"

@st.cache_data
def load_stock_list(file):
    df = pd.read_csv(file)
    return df['SYMBOL'].dropna().tolist()

stock_symbols = load_stock_list(STOCK_LIST)

selected_stock = st.selectbox("Select Stock", stock_symbols)

@st.cache_data
def load_data(ticker):
    try:
        START = "2010-01-01"
        TODAY = "2022-12-31"
        df = yf.download(ticker, START, TODAY)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

df = load_data(selected_stock)

# Check if data is empty
if df.empty:
    st.error("Failed to retrieve data. Please select a different stock.")
    st.stop()

# Preprocessing
train = pd.DataFrame(df[0:int(len(df)*0.70)])
test = pd.DataFrame(df[int(len(df)*0.70): int(len(df))])
train_close = train.iloc[:, 4:5].values
test_close = test.iloc[:, 4:5].values

minmax_scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = minmax_scaler.fit_transform(train_close)

x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# GRU Model
gru_model = Sequential()
gru_model.add(GRU(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(60, return_sequences=True))
gru_model.add(Dropout(0.3))
gru_model.add(GRU(80, return_sequences=True))
gru_model.add(Dropout(0.4))
gru_model.add(GRU(120))
gru_model.add(Dropout(0.5))
gru_model.add(Dense(units=1))

gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_model.fit(x_train, y_train, epochs = 5, batch_size=32) # Reduced epochs for faster execution

# Preparing Test Data
past_100_days = pd.DataFrame(train_close[-100:])
test_df = pd.DataFrame(test_close)
final_df = pd.concat([past_100_days, test_df], ignore_index=True)
input_data = minmax_scaler.transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Making Predictions
gru_pred = gru_model.predict(x_test)
gru_pred = minmax_scaler.inverse_transform(gru_pred)
y_test = minmax_scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test.flatten(), mode='lines', name='Original'))
fig.add_trace(go.Scatter(x=np.arange(len(gru_pred)), y=gru_pred.flatten(), mode='lines', name='GRU Prediction'))
fig.update_layout(title='GRU Prediction vs Original', xaxis_title='Time', yaxis_title='Price')
st.plotly_chart(fig)

# Model evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

mse = mean_squared_error(y_test, gru_pred)
mae = mean_absolute_error(y_test, gru_pred)
rmse = root_mean_squared_error(y_test, gru_pred)
mape = mean_absolute_percentage_error(y_test, gru_pred)

st.write('Mean squared error on test set: {:.2f}'.format(mse))
st.write('Mean absolute error on test set: {:.2f}'.format(mae))
st.write('Root mean squared error on test set: {:.2f}'.format(rmse))
st.write('Mean absolute percentage error on test set: {:.2f}'.format(mape))
