import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split

def load_and_process_data(
    company: str,
    start_date: str,
    end_date: str,
    features: list = ['Open', 'High', 'Low', 'Close', 'Volume'],
    handle_nan: str = 'drop',
    split_method: str = 'ratio',
    train_ratio: float = 0.8,
    scale_data: bool = True,
    save_local: bool = True,
    local_path: str = "data.csv"
):

# Check if the data is already stored locally
    if os.path.exists(local_path):
        df = pd.read_csv(local_path, index_col=0, parse_dates=True)
    else:
        # Download data from Yahoo Finance
        df = yf.download(company, start=start_date, end=end_date)
        if save_local:
            df.to_csv(local_path)

# Select relevant features
    df = df[features]
    
# Handle missing values
    if handle_nan == 'drop':
        df.dropna(inplace=True)
    elif handle_nan == 'fill':
        df.fillna(method='ffill', inplace=True)

# Define target variable (Close price) and separate features
    target = df['Close'].values.reshape(-1, 1)
    features = df.drop(columns=['Close']).values

# Scale features if required
    scalers = {}
    if scale_data:
        feature_scaler = MinMaxScaler()
        features = feature_scaler.fit_transform(features)
        scalers['features'] = feature_scaler
        
        target_scaler = MinMaxScaler()
        target = target_scaler.fit_transform(target)
        scalers['target'] = target_scaler
    
# Split the data using different methods
    if split_method == 'ratio':
        X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=train_ratio, shuffle=False)
    elif split_method == 'random':
        X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=train_ratio, shuffle=True)
    else:
        # Split by date
        split_date = dt.datetime.strptime(start_date, "%Y-%m-%d") + (dt.datetime.strptime(end_date, "%Y-%m-%d") - dt.datetime.strptime(start_date, "%Y-%m-%d")) * train_ratio
        df_train = df[df.index < split_date]
        df_test = df[df.index >= split_date]
        X_train, y_train = df_train.drop(columns=['Close']).values, df_train['Close'].values.reshape(-1, 1)
        X_test, y_test = df_test.drop(columns=['Close']).values, df_test['Close'].values.reshape(-1, 1)
    
    return X_train, X_test, y_train, y_test, scalers if scale_data else None

# Load and process the stock data
company = 'META'
start_date = "2012-01-01"
end_date = "2020-01-01"
X_train, X_test, y_train, y_test, scalers = load_and_process_data(company, start_date, end_date)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=25, batch_size=32)

# Test the Model Accuracy on Existing Data

# Load test data and process
test_start = "2020-01-01"
test_end = dt.datetime.now().strftime("%Y-%m-%d")
X_train, X_test, y_train, y_test, _ = load_and_process_data(company, test_start, test_end)

# Predict stock prices using the trained model
predicted_prices = model.predict(X_test)
predicted_prices = scalers['target'].inverse_transform(predicted_prices)

# Load actual stock prices for comparison
test_data = yf.download(company, start=test_start, end=test_end)
actual_price = test_data['Close'].values

# Plot actual vs predicted prices
plt.plot(actual_price, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

# Predict the next day's stock price
real_data = [X_test[-1]]  # Use the last test data sample
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], X_test.shape[2]))

prediction = model.predict(real_data)  # Make prediction
prediction = scalers['target'].inverse_transform(prediction)
print(f"Prediction: {prediction}")
