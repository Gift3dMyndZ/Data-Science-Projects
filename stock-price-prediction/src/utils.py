import yfinance as yf
import pandas as pd

def download_stock_data(ticker: str, start='2015-01-01', end='2023-12-31'):
    """
    Download stock price data from Yahoo Finance.
    Returns a DataFrame with the 'Close' column as the main focus.
    """
    print(f"⬇️ Downloading {ticker} data from Yahoo Finance...")
    data = yf.download(ticker, start=start, end=end)
    data = data[['Close']]
    data.dropna(inplace=True)
    print(f"✅ Successfully loaded {len(data)} records.")
    return data