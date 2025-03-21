import os
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Function to load and process data (multivariate support already present)
def load_and_process_data(company, start_date, end_date, features=['Open', 'High', 'Low', 'Close','Adj Close','Volume'],
                          handle_nan='drop', scale_data=True, save_local=True, local_path='data.csv', scalers=None):
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        df.rename(columns={'Price': 'Date'}, inplace=True)
        df = df.iloc[1:].reset_index(drop=True)
        df.set_index('Date', inplace=True)
    else:
        df = yf.download(company, start=start_date, end_date=end_date, actions=False, auto_adjust=False)
        if save_local:
            df.to_csv(local_path)
    
    df = df[features]
    if df.empty:
        raise ValueError("Error: Stock data is empty.")
    
    if handle_nan == 'drop':
        df.dropna(inplace=True)
    elif handle_nan == 'fill':
        df.fillna(method='ffill', inplace=True)
    
    df = df.astype(float)
    
    features_data = df.values
    target = df['Close'].values.reshape(-1, 1)
    
    if scale_data:
        if scalers is None:
            feature_scaler = MinMaxScaler()
            features_data = feature_scaler.fit_transform(features_data)
            target_scaler = MinMaxScaler()
            target = target_scaler.fit_transform(target)
            scalers = {'features': feature_scaler, 'target': target_scaler}
        else:
            features_data = scalers['features'].transform(features_data)
            target = scalers['target'].transform(target)
    
    return features_data, target, scalers, df

# Prepare sequences for multistep prediction
def create_sequences(features, target, seq_length, steps_ahead):
    X, y = [], []
    for i in range(len(features) - seq_length - steps_ahead + 1):
        X.append(features[i:i + seq_length])  # Input sequence
        y.append(target[i + seq_length:i + seq_length + steps_ahead].flatten())  # Target sequence (k steps), flattened to (steps_ahead,)
    return np.array(X), np.array(y)

# Updated model creation function (supports multistep output)
def create_dl_model(layer_type='LSTM', num_layers=3, layer_sizes=[50, 50, 50], 
                    input_shape=(60, 5), steps_ahead=1, dropout_rate=0.2):
    model = Sequential()
    layer_dict = {'LSTM': LSTM}
    if layer_type not in layer_dict:
        raise ValueError("Invalid layer type. Choose 'LSTM' for now.")
    Layer = layer_dict[layer_type]
    
    model.add(Layer(units=layer_sizes[0], return_sequences=(num_layers > 1), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    for i in range(1, num_layers - 1):
        model.add(Layer(units=layer_sizes[i], return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    if num_layers > 1:
        model.add(Layer(units=layer_sizes[-1]))
        model.add(Dropout(dropout_rate))
    
    # Output layer for multistep prediction (steps_ahead predictions)
    model.add(Dense(units=steps_ahead))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Combined multivariate, multistep prediction function
def train_and_predict_multivariate_multistep(company, start_date, end_date, test_start, test_end, 
                                             seq_length=60, steps_ahead=5, layer_type='LSTM', 
                                             num_layers=3, layer_sizes=[50, 50, 50], epochs=25, batch_size=32):
    # Load training data
    features_data, target, scalers, df = load_and_process_data(company, start_date, end_date, scale_data=True)
    
    # Create sequences
    X, y = create_sequences(features_data, target, seq_length, steps_ahead)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
    
    # Create and train the model
    model = create_dl_model(layer_type=layer_type, num_layers=num_layers, layer_sizes=layer_sizes, 
                           input_shape=(seq_length, features_data.shape[1]), steps_ahead=steps_ahead)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Load test data using training scalers
    features_test, target_test, _, df_test = load_and_process_data(company, test_start, test_end, scale_data=True, scalers=scalers)
    X_test, y_test_full = create_sequences(features_test, target_test, seq_length, steps_ahead)
    
    # Predict
    predicted_prices = model.predict(X_test)  # (n_test, steps_ahead)
    
    # Inverse transform actual prices
    y_test_reshaped = y_test_full.reshape(-1, 1)  # (n_test * steps_ahead, 1)
    actual_prices = scalers['target'].inverse_transform(y_test_reshaped)
    actual_prices = actual_prices.reshape(X_test.shape[0], steps_ahead)  # back to (n_test, steps_ahead)
    
    # Inverse transform predicted prices
    predicted_prices_reshaped = predicted_prices.reshape(-1, 1)  # (n_test * steps_ahead, 1)
    predicted_prices_inv = scalers['target'].inverse_transform(predicted_prices_reshaped)
    predicted_prices_inv = predicted_prices_inv.reshape(X_test.shape[0], steps_ahead)  # back to (n_test, steps_ahead)
    
    # Plot results for each step ahead
    for step in range(steps_ahead):
        plt.figure(figsize=(10, 6))
        plt.plot(actual_prices[:, step], color="black", label=f"Actual {company} Price (Day {step+1})")
        plt.plot(predicted_prices_inv[:, step], color="green", label=f"Predicted {company} Price (Day {step+1})")
        plt.title(f"{company} Share Price Prediction (Day {step+1} Ahead)")
        plt.xlabel('Time')
        plt.ylabel(f'{company} Share Price')
        plt.legend()
        plt.show()
    
    # Last prediction
    last_sequence = np.array([features_test[-seq_length:]])
    prediction = model.predict(last_sequence)  # (1, steps_ahead)
    prediction_reshaped = prediction.reshape(-1, 1)  # (steps_ahead, 1)
    prediction_inv = scalers['target'].inverse_transform(prediction_reshaped)
    prediction_inv = prediction_inv.reshape(1, steps_ahead)  # (1, steps_ahead)
    print(f"Prediction for next {steps_ahead} days: {prediction_inv.flatten()}")

# Example usage
company = 'META' #input("Enter stock symbol (e.g., META): ")
start_date = '2012-01-01' #input("Enter start date (YYYY-MM-DD): ")
end_date = '2022-01-01'  #input("Enter end date (YYYY-MM-DD): ")
test_start = '2022-01-01'  #input("Enter test start date (YYYY-MM-DD): ")
test_end = '2024-01-01'  #input("Enter test end date (YYYY-MM-DD): ")
train_and_predict_multivariate_multistep(company, start_date, end_date, test_start, test_end, steps_ahead=5)