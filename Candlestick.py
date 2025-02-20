import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

data_df = pd.read_csv("data.csv", index_col=0, parse_dates=True)
data_df = data_df.reset_index()

data_df=data_df[-30:]

green_df = data_df[data_df.Close > data_df.Open].copy()
green_df["Height"] = green_df["Close"] - green_df["Open"]

red_df = data_df[data_df.Close < data_df.Open].copy()
red_df["Height"] = red_df["Open"] - red_df["Close"]

fig = plt.figure(figsize=(15,7))

plt.style.use("fivethirtyeight")

#Grey Lines
plt.vlines(x=green_df["Date"], ymin=green_df["Low"], ymax=green_df["High"],color="green")
plt.vlines(x=red_df["Date"], ymin=red_df["Low"], ymax=red_df["High"],color="orangered")

#Green Candles
plt.bar(x=green_df["Date"], height=green_df["Height"], bottom=green_df["Open"], color="green")

#Red Candles
plt.bar(x=red_df["Date"], height=red_df["Height"], bottom=red_df["Close"], color="orangered")

plt.yticks(range(180,230,10), ["{} $".format(v) for v in range(180,230,10)])

plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.title("META Stock Prices")


plt.show()