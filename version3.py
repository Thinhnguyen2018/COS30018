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
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_process_data(company, start_date, end_date, features=['Open', 'High', 'Low', 'Close', 'Volume'],
                           handle_nan='drop', train_ratio=0.8, scale_data=True, save_local=True, local_path='data.csv'):
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        df.rename(columns={'Price': 'Date'}, inplace=True)
        df = df.iloc[1:].reset_index(drop=True)
        df.set_index('Date', inplace=True)
    else:
        df = yf.download(company, start=start_date, end=end_date)
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
    
    target = df['Close'].values.reshape(-1, 1)
    features = df.drop(columns=['Close']).values
    
    scalers = {}
    if scale_data:
        feature_scaler = MinMaxScaler()
        features = feature_scaler.fit_transform(features)
        scalers['features'] = feature_scaler
        
        target_scaler = MinMaxScaler()
        target = target_scaler.fit_transform(target)
        scalers['target'] = target_scaler
    
    features = np.reshape(features, (features.shape[0], features.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=train_ratio, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scalers, df

def create_dl_model(layer_type='LSTM', num_layers=3, layer_sizes=[50, 50, 50], input_shape=(60, 1), dropout_rate=0.2):
    model = Sequential()
    layer_dict = {'LSTM': LSTM, 'RNN': SimpleRNN, 'GRU': GRU}
    if layer_type not in layer_dict:
        raise ValueError("Invalid layer type. Choose from 'LSTM', 'RNN', or 'GRU'.")
    Layer = layer_dict[layer_type]
    
    model.add(Layer(units=layer_sizes[0], return_sequences=(num_layers > 1), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    for i in range(1, num_layers - 1):
        model.add(Layer(units=layer_sizes[i], return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    if num_layers > 1:
        model.add(Layer(units=layer_sizes[-1]))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_predict(company, start_date, end_date, test_start, test_end, layer_type='LSTM', num_layers=3, layer_sizes=[50, 50, 50], epochs=25, batch_size=32):
    X_train, X_test, y_train, y_test, scalers, df = load_and_process_data(company, start_date, end_date)
    model = create_dl_model(layer_type=layer_type, num_layers=num_layers, layer_sizes=layer_sizes, input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    X_train, X_test, y_train, y_test, scalers, df_test = load_and_process_data(company, test_start, test_end)
    predicted_prices = model.predict(X_test)
    predicted_prices = scalers['target'].inverse_transform(predicted_prices)
    
    test_data = yf.download(company, start=test_start, end=test_end)
    actual_price = test_data['Close'].values
    
    plt.plot(actual_price, color="black", label=f"Actual {company} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
    plt.title(f"{company} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()
    
    real_data = np.array([X_test[-1]])
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], X_test.shape[2]))
    prediction = model.predict(real_data)
    prediction = scalers['target'].inverse_transform(prediction)
    print(f"Prediction: {prediction}")
    plot_candlestick_chart(df_test, n=1)
    plot_boxplot_chart(df_test, n=30, step=10)

def plot_candlestick_chart(data, n=1):
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    if n > 1:
        data = data.resample(f'{n}D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    mpf.plot(data, type='candle', style='charles', volume=True, title=f'Candlestick Chart ({n}-day)', ylabel='Price ($)', ylabel_lower='Volume')
    plt.show()

def plot_boxplot_chart(data, n=30, step=10):
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data = data.tail(300)
    rolling_data = [data['Close'][i:i+n].values for i in range(0, len(data)-n+1, step)]
    rolling_dates = [data.index[i].strftime('%Y-%m-%d') for i in range(0, len(data)-n+1, step)]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=rolling_data)
    plt.xticks(ticks=range(len(rolling_dates)), labels=rolling_dates, rotation=45)
    plt.xlabel("Trading Window")
    plt.ylabel("Closing Price")
    plt.title(f"Boxplot of Stock Prices (Rolling {n} Days, Step {step})")
    plt.show()

company = input("Enter stock symbol (e.g., META): ")
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")
test_start = input("Enter test start date (YYYY-MM-DD): ")
test_end = input("Enter test end date (YYYY-MM-DD): ")
train_and_predict(company, start_date, end_date, test_start, test_end)
