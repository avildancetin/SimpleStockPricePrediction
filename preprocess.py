import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_stock_data(ticker_symbol, start_date, end_date):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # drop the rows with NaN values
    stock_data = stock_data.dropna()

    stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
    stock_data['Moving_Avg_5'] = stock_data['Adj Close'].rolling(
        window=5).mean()
    stock_data['Moving_Avg_10'] = stock_data['Adj Close'].rolling(
        window=10).mean()

    stock_data = stock_data.dropna()

    # For each row in the DataFrame, the 'Next_Close' column will contain
    # the adjusted closing price of the day after the current row.
    # The last row of the DataFrame will have a NaN (Not a Number) value in
    # the 'Next_Close' column because there is no data available for the next day.
    stock_data['Next_Close'] = stock_data['Adj Close'].shift(-1)

    stock_data = stock_data.dropna()

    return stock_data


ticker_symbol = 'AMZN'
start_date = '2023-01-01'
end_date = '2024-01-01'
stock_data = preprocess_stock_data(ticker_symbol, start_date, end_date)
print(stock_data.head())
print(stock_data.tail())

x = stock_data[['Daily_Return', 'Moving_Avg_5', 'Moving_Avg_10']]  # features
y = stock_data['Next_Close']  # target variable

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

print(f'Training set shape - X: {x_train.shape}, y: {y_train.shape}')
print(f'Testing set shape - X: {x_test.shape}, y: {y_test.shape}')
