import yfinance as yf
# I have chosen Amazon.com, Inc. (AMZN)
ticker_symbol = 'AMZN'
start_date = '2023-01-01'
end_date = '2024-01-01'

stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# print(stock_data)
print(stock_data.head())
print(stock_data.tail())
# print(stock_data.info())
# print(stock_data.describe())
# print(stock_data.columns)
print(stock_data.shape)
# print(stock_data.index)
