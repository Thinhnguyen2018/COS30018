import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import io
import base64

# Function to load and process data
def load_and_process_data(company, start_date, end_date, features=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
                          handle_nan='drop', scale_data=True, save_local=True, local_path='data.csv', scalers=None):
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        df.rename(columns={'Price': 'Date'}, inplace=True)
        df = df.iloc[1:].reset_index(drop=True)
        df.set_index('Date', inplace=True)
    else:
        df = yf.download(company, start=start_date, end=end_date, actions=False, auto_adjust=False)
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
    feature_data = df.values
    if scale_data:
        if scalers is None:
            feature_scaler = MinMaxScaler()
            feature_data = feature_scaler.fit_transform(feature_data)
            target_scaler = MinMaxScaler()
            target = target_scaler.fit_transform(target)
            scalers = {'features': feature_scaler, 'target': target_scaler}
        else:
            feature_data = scalers['features'].transform(feature_data)
            target = scalers['target'].transform(target)

    return feature_data, target, scalers, df

# Function to create sequences
def create_sequences(features, target, seq_length, steps_ahead):
    X, y = [], []
    for i in range(len(features) - seq_length - steps_ahead + 1):
        X.append(features[i:i + seq_length])
        y.append(target[i + seq_length:i + seq_length + steps_ahead].flatten())
    return np.array(X), np.array(y)

# Model creation function
def create_dl_model(layer_type='LSTM', num_layers=3, layer_sizes=[20, 20, 20], input_shape=(60, 5),
                    steps_ahead=1, dropout_rate=0.2):
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

    model.add(Dense(units=steps_ahead))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Combined function for training and prediction with ensemble
def train_and_predict(company, start_date, end_date, test_start, test_end, seq_length=60, steps_ahead=1,
                      layer_type='LSTM', num_layers=3, layer_sizes=[20, 20, 20], epochs=5, batch_size=32,
                      use_sarima=False):
    # Load and prepare training data
    features, target, scalers, df = load_and_process_data(company, start_date, end_date, scale_data=True)
    X, y = create_sequences(features, target, seq_length, steps_ahead)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

    # Fit ARIMA/SARIMA on training 'Close' prices
    seasonal = use_sarima
    model_arima = auto_arima(df['Close'], seasonal=seasonal, m=5 if seasonal else 1, trace=True,
                             suppress_warnings=True)

    # Create and train DL model
    model_dl = create_dl_model(layer_type=layer_type, num_layers=num_layers, layer_sizes=layer_sizes,
                               input_shape=(seq_length, features.shape[1]), steps_ahead=steps_ahead)
    model_dl.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Adjust test_end to not exceed current date (April 3, 2025)
    current_date = '2025-04-03'
    test_end = min(test_end, current_date)

    # Load and prepare test data
    features_test, target_test, _, df_test = load_and_process_data(company, test_start, test_end, scale_data=True,
                                                                   scalers=scalers)
    if df_test.empty:
        raise ValueError(f"Test data is empty for {company} from {test_start} to {test_end}.")

    test_dates = pd.to_datetime(df_test.index)
    X_test, y_test_full = create_sequences(features_test, target_test, seq_length, steps_ahead)

    # Forecast with ARIMA/SARIMA for test period
    arima_forecasts = model_arima.predict(n_periods=len(test_dates))
    arima_forecasts = pd.Series(arima_forecasts, index=test_dates)

    # Predict with DL model
    predicted_prices = model_dl.predict(X_test)

    # Inverse transform predictions
    y_test_reshaped = y_test_full.reshape(-1, 1)
    actual_prices = scalers['target'].inverse_transform(y_test_reshaped).reshape(X_test.shape[0], steps_ahead)
    predicted_prices_reshaped = predicted_prices.reshape(-1, 1)
    predicted_prices_inv = scalers['target'].inverse_transform(predicted_prices_reshaped).reshape(X_test.shape[0], steps_ahead)

    # Create ARIMA/SARIMA predictions for each sequence with NaN fallback
    y_arima = np.zeros_like(predicted_prices_inv)
    for i in range(len(X_test)):
        for j in range(steps_ahead):
            pred_date = test_dates[i + seq_length + j]
            forecast_value = arima_forecasts.get(pred_date, np.nan)
            y_arima[i, j] = forecast_value if not pd.isna(forecast_value) else actual_prices[i, j]

    # Ensemble prediction
    ensemble_predictions = (predicted_prices_inv + y_arima) / 2

    # Generate plots as base64 strings
    plot_images = []
    for step in range(steps_ahead):
        plt.figure(figsize=(12, 6))
        plt.plot(actual_prices[:, step], color="black", label=f"Actual {company} Price (Day {step+1})")
        plt.plot(predicted_prices_inv[:, step], color="green", label=f"DL Predicted (Day {step+1})")
        plt.plot(y_arima[:, step], color="red", label=f"ARIMA Predicted (Day {step+1})")
        plt.plot(ensemble_predictions[:, step], color="blue", label=f"Ensemble Predicted (Day {step+1})")
        plt.title(f"{company} Share Price Prediction (Day {step+1} Ahead) - Ensemble")
        plt.xlabel('Time')
        plt.ylabel(f'{company} Share Price')
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_images.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        plt.close()

    # Candlestick chart
    df_test.index = pd.to_datetime(df_test.index)
    buf = io.BytesIO()
    mpf.plot(df_test, type='candle', style='charles', volume=True,
             title=f'Candlestick Chart (1-day)', ylabel='Price ($)',
             ylabel_lower='Volume', savefig=buf)
    buf.seek(0)
    plot_images.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

    # Boxplot
    n, step = 30, 10
    data = df_test.copy()
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
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_images.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
    plt.close()

    # Calculate MSE
    mse_results = {}
    for step in range(steps_ahead):
        mse_dl = mean_squared_error(actual_prices[:, step], predicted_prices_inv[:, step])
        mse_arima = mean_squared_error(actual_prices[:, step], y_arima[:, step])
        mse_ensemble = mean_squared_error(actual_prices[:, step], ensemble_predictions[:, step])
        mse_results[step] = {'DL': mse_dl, 'ARIMA': mse_arima, 'Ensemble': mse_ensemble}

    return mse_results, plot_images