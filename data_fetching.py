import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df