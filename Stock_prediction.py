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
    company: str, #Stock ticker symbol 
    start_date: str, #Start date range for downloading stock data 
    end_date: str, #End date range for downloading stock data 
    features: list = ['Open', 'High', 'Low', 'Close', 'Volume'], #List of columns to be used as input features
    handle_nan: str = 'drop', #Determines how to handle missing values
    split_method: str = 'ratio', #Ratio for splitting training and test data
    train_ratio: float = 0.8,
    scale_data: bool = True, #If true, scales the data using MinMaxScaler
    save_local: bool = True, #Saves the dataset locally if True
    local_path: str = "data.csv" #Path where the data is saved
):
    """
    Loads and processes stock data with multiple features.
    """

    #Check if the data exist in the local files
    #If the data is not exist, download the stock datae from Yahoo Finance and saves it locally. 
    if os.path.exists(local_path):
        df = pd.read_csv(local_path, index_col=0, parse_dates=True)
    else:
        df = yf.download(company, start=start_date, end=end_date)
        if save_local:
            df.to_csv(local_path)
    
    #Selects only relevant stock price features
    df = df[features]
    
    #If the dataset is empty, an error will raised
    if df.empty:
        raise ValueError("Error: Stock data is empty. Check the ticker symbol and date range.")
    
    #If the data is missing it will either drop or fill
    if handle_nan == 'drop':
        df.dropna(inplace=True)
    elif handle_nan == 'fill':
        df.fillna(method='ffill', inplace=True)
    
    #Target variable is the closing price => Dependent variable 
    #Feature include Open, High, Low and Volume => Independent variable
    target = df['Close'].values.reshape(-1, 1)
    features = df.drop(columns=['Close']).values
    
    #Normalize the feature values between 0 and 1 to improve neural network performance
    scalers = {}
    if scale_data:
        feature_scaler = MinMaxScaler()
        features = feature_scaler.fit_transform(features)
        scalers['features'] = feature_scaler
        
        target_scaler = MinMaxScaler()
        target = target_scaler.fit_transform(target)
        scalers['target'] = target_scaler
    
    #Reshape the feature set into 3D format for LSTM input
    if features.shape[0] > 0 and features.shape[1] > 0:
        features = np.reshape(features, (features.shape[0], features.shape[1], 1))
    else:
        raise ValueError("Error: No data available after preprocessing.")
    
    #Splits the dataset into training (80%) and testing (20%) sets. 
    if features.shape[0] > 1:
        X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=train_ratio, shuffle=False)
    else:
        raise ValueError("Not enough data to split. Check the dataset size.")
    
    return X_train, X_test, y_train, y_test, scalers if scale_data else None


#Load and processes dataset for training
company = 'META'
start_date = "2012-01-01"
end_date = "2020-01-01"
X_train, X_test, y_train, y_test, scalers = load_and_process_data(company, start_date, end_date)

if X_train.shape[0] == 0:
    raise ValueError("Error: X_train is empty after processing. Please check the dataset.")

#Creating the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the Model
model.fit(X_train, y_train, epochs=25, batch_size=32)

#Testing the model
#Downloads new test data from 2020 till today
test_start = "2020-01-01"
test_end = dt.datetime.now().strftime("%Y-%m-%d")
X_train, X_test, y_train, y_test, _ = load_and_process_data(company, test_start, test_end)

if X_test.shape[0] == 0:
    raise ValueError("Error: X_test is empty after processing. Please check the dataset.")

#Making a predictions
#Predicts stock prices using the trained model and
#Convers scaled prediction back to original values
predicted_prices = model.predict(X_test)
predicted_prices = scalers['target'].inverse_transform(predicted_prices)

#Plotting actual prices vs predicted prices
test_data = yf.download(company, start=test_start, end=test_end)
actual_price = test_data['Close'].values

plt.plot(actual_price, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

#Using last available data point to predict the next day's stock price
real_data = [X_test[-1]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], X_test.shape[2]))

prediction = model.predict(real_data)
prediction = scalers['target'].inverse_transform(prediction)
print(f"Prediction: {prediction}")
