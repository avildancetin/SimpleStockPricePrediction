# Simple Stock Price Prediction with Linear Regression

This project uses a linear regression model to predict the next day's adjusted closing price of a stock based on its historical data, daily returns, and moving averages.

## Dependencies

Python
yfinance
pandas
numpy
scikit-learn
matplotlib
## Installation

Bash
pip install yfinance 
pip install pandas
pip install numpy
pip install scikit-learn 
pip install matplotlib
## Usage

Run the script:

Bash
python stock_prediction.py
Use code with caution. Learn more
Customize parameters (optional):

ticker_symbol: The stock ticker symbol to analyze (default: 'AMZN')
start_date: The start date of the data to fetch (default: '2023-01-01')
end_date: The end date of the data to fetch (default: '2024-01-01')
future_date: The date for which to predict the closing price (default: '2024-01-06')
## Output

Data exploration: Prints the first and last few rows of the preprocessed stock data.
Model training: Trains a linear regression model and reports its performance metrics (Mean Squared Error).
Prediction: Predicts the next day's adjusted closing price for the specified future date.
Visualization: Plots actual, predicted, and future predicted closing prices against the 5-day moving average.
## Limitations

Stock prices are inherently unpredictable and influenced by numerous factors. This model is a simplified approach for prediction.
Linear regression might not capture complex non-linear relationships in stock price movements.
Consider exploring other prediction techniques like time series analysis or machine learning models for potentially better results.
