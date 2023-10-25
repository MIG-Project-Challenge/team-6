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
fast_period = 14
slow_period = 29
signal_period = 9
for stock in range(len(open_prices)):
    # Calculate MACD
    macd, signal, _ = ta.MACD(open_prices[stock], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    
    # Initialize buy and sell signals for the current stock
    buy_signal = [False] * len(macd)
    sell_signal = [False] * len(macd)
    fast_sma = ta.SMA(open_prices[stock], timeperiod=5)
    slow_sma = ta.SMA(open_prices[stock], timeperiod=40)
    # Determine buy and sell signals
    for i in range(1, len(macd)):
        if macd[i] > signal[i] and macd[i - 1] <= signal[i - 1]:
            buy_signal[i] = True
        elif macd[i] < signal[i] and macd[i - 1] >= signal[i - 1]:
            sell_signal[i] = True

    # Apply buy and sell signals to the trades list
    for day in range(1, len(open_prices[stock])-1):
        if buy_signal[day]:
            trades[stock][day+1] = 1  # Buy signal
        elif sell_signal[day]:
            trades[stock][day+1] = -1  # Sell signal
        else:
            trades[stock][day+1] = 0 # No signal
# for stock in range(len(open_prices)): 
#     fast_sma = ta.SMA(open_prices[stock], timeperiod=5)
#     slow_sma = ta.SMA(open_prices[stock], timeperiod=40)
#     momentum = ta.MOM(open_prices[stock], timeperiod=10)
#     mean = np.mean(open_prices[stock])

#     for day in range(1, len(open_prices[0])-1):
        
#         # Buy: price is below the mean and momentum is positive
#         if open_prices[stock][day] < mean and momentum[day] > 0:
#             # we are trading the next day's open price
#             trades[stock][day+1] = 1

#         # Sell/short: price is above the mean and momentum is negative
#         elif open_prices[stock][day] > mean and momentum[day] < 0:
#             # we are trading the next day's open price
#             trades[stock][day+1] = -1
#         # else do nothing
#         else:
#             trades[stock][day+1] = 0

# for stock in range(len(open_prices)): 
#     rsi = ta.RSI(open_prices[stock], timeperiod=14)
#     # print(rsi)

#     for day in range(1, len(open_prices[0])-1):
        
#         # Buy: RSI crosses above 30
#         if rsi[day] > 30 and rsi[day-1] <= 30:
#             # we are trading the next day's open price
#             trades[stock][day+1] = 1      
#         # Sell/short: RSI crosses above 70
#         elif rsi[day] > 70 and rsi[day-1] <= 70:
#             # we are trading the next day's open price
#             trades[stock][day+1] = -1
#         # else do nothing
#         else:
#             trades[stock][day+1] = 0
# for stock in range(len(open_prices)): 
#     fast_sma = ta.SMA(open_prices[stock], timeperiod=5)
#     slow_sma = ta.SMA(open_prices[stock], timeperiod=40)

#     for day in range(1, len(open_prices[0])-1):

#         # In your trading loop
#         if fast_sma[day] > slow_sma[day]:
#             stop_price = open_prices[stock][day] - stop_loss[day]
#             take_profit_price = open_prices[stock][day] + take_profit[day]
        
#         # Buy: fast SMA crosses above slow SMA
#         if fast_sma[day] > slow_sma[day] and fast_sma[day-1] <= slow_sma[day-1]:
#             # we are trading the next day's open price
#             trades[stock][day+1] = 1
        
#         # Sell/short: fast SMA crosses below slow SMA
#         elif fast_sma[day] < slow_sma[day] and fast_sma[day-1] >= slow_sma[day-1]:
#             # we are trading the next day's open price
#             trades[stock][day+1] = -1
#         # else do nothing
#         else:
#             trades[stock][day+1] = 0
# with np.printoptions(threshold=np.inf):
#     print(trades)

# for stock in range(len(open_prices)): 
#     macd, signal, _ = ta.MACD(open_prices[stock], fastperiod=12, slowperiod=26, signalperiod=9)

#     for day in range(1, len(open_prices[0]) - 1):
#         # Buy: MACD crosses above its signal line
#         if macd[day] > signal[day] and macd[day - 1] <= signal[day - 1]:
#             # we are trading the next day's open price
#             trades[stock][day + 1] = 1

#         # Sell/short: MACD crosses below its signal line
#         elif macd[day] < signal[day] and macd[day - 1] >= signal[day - 1]:
#             # we are trading the next day's open price
#             trades[stock][day + 1] = -1
#         # else do nothing
#         else:
#             trades[stock][day + 1] = 0

print(trades)
portfolio_value, sharpe_ratio = eval_actions(trades, open_prices, cash=25000, verbose=True)
print(f"\nPortfolio value: {portfolio_value}")
print(f"Sharpe ratio: {sharpe_ratio}")
