import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

def plot_candlestick_chart(data, n=1):
    """
    Plots a candlestick chart for the given stock market data.
    Each candlestick can represent the data for 'n' trading days.
    
    Parameters:
    - data: DataFrame containing 'Date', 'Open', 'High', 'Low', 'Close' columns.
    - n: Number of trading days to aggregate per candlestick (default is 1).
    """
    
    #Converting the Date column in the data.csv to a datetime format to allow time-based operations
    data['Date'] = pd.to_datetime(data['Date'])  
    data.set_index('Date', inplace=True)

    #If n>1, aggregate data over 'n' trading days
    if n > 1:
        data = data.resample(f'{n}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    
    #Generate and display the candlestick chart 
    mpf.plot( 
                data,
                type='candle', 
                style='charles', 
                volume=True, 
                title=f'Candlestick Chart ({n}-day)', 
                ylabel='Price ($) ', 
                ylabel_lower='Volume')


#Reading the data from the data.csv locally
df = pd.read_csv("data.csv")
plot_candlestick_chart(df, n=30)

