## Loading the Data
import pandas as pd
import numpy as np
import talib as ta # in order to run this, you need to install talib see https://github.com/TA-Lib/ta-lib-python
from eval_algo import eval_actions

# Read the training and testing data from CSV files
train_df = pd.read_csv('train_data_50.csv')

# change date into datetime objects
train_df['Date'] = pd.to_datetime(train_df['Date'])

# set indexes 
train_df.set_index(["Ticker", "Date"], inplace=True)

print(train_df)

tickers = sorted(train_df.index.get_level_values('Ticker').unique())

open_prices = []

for ticker in tickers:
    stock_close_data = train_df.loc[ticker]["Open"]
    open_prices.append(stock_close_data.values)

open_prices = np.stack(open_prices)
print(open_prices.shape)
print(open_prices)

trades = np.zeros_like(open_prices)
print(trades)

# NOTE: that we don't use future price data to make trades in the past!
# This is to prevent "look-ahead bias" which is a common mistake in algorithmic trading, and will result in a score of 0 in the competition
# TODO: have a way to track your positions (short/long positions), your cash balance, adn your portfolio value
# recall the restrictions that you cannot buy a share if you don't have enough cash
# and you can't have your debt exceed the value of your cash and long positions
# "debt" in our case is really just the negative value of our short positions


for stock in range(len(open_prices)): 
    fast_sma = ta.SMA(open_prices[stock], timeperiod=5)
    slow_sma = ta.SMA(open_prices[stock], timeperiod=40)

    for day in range(1, len(open_prices[0])-1):
        
        # Buy: fast SMA crosses above slow SMA
        if fast_sma[day] > slow_sma[day] and fast_sma[day-1] <= slow_sma[day-1]:
            # we are trading the next day's open price
            trades[stock][day+1] = 1      
        # Sell/short: fast SMA crosses below slow SMA
        elif fast_sma[day] < slow_sma[day] and fast_sma[day-1] >= slow_sma[day-1]:
            # we are trading the next day's open price
            trades[stock][day+1] = -1
        # else do nothing
        else:
            trades[stock][day+1] = 0
print(trades)

portfolio_value, sharpe_ratio = eval_actions(trades, open_prices, cash=25000, verbose=True)
print(f"\nPortfolio value: {portfolio_value}")
print(f"Sharpe ratio: {sharpe_ratio}")
