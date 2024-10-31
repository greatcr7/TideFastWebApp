from datetime import datetime, timedelta
from dotenv import load_dotenv
import tushare as ts
import os
import yfinance as yf
import streamlit as st

load_dotenv()
tushare_token = os.getenv('TUSHARE_TOKEN')
pro = ts.pro_api(token=tushare_token)

def convert_date_string(date_str):
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

def calculate_derivatives(df, column):
    """
    Calculate first and second derivatives of a given column.
    """
    df[f'{column}_d1'] = df[column].diff()
    df[f'{column}_d2'] = df[f'{column}_d1'].diff()
    return df


@st.cache_data
def get_stock_prices(ticker, start_date=None, end_date=None):
    """
    Fetch stock price data using Tushare and YFinance.

    :param ticker: Stock ticker symbol
    :param start_date: Start date for data retrieval (optional)
    :param end_date: End date for data retrieval (optional)
    :return: DataFrame with stock price data
    """

    def determine_market(ticker):
        if ticker.endswith('.HK'):
            return 'hk'
        elif ticker.endswith(('.SH', '.SZ', '.BJ')):
            return 'china'
        else:
            return 'us'
        
    market = determine_market(ticker)

    # Handle date logic
    if market in ["us", "hk"]:
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)
            start_date = start_date_dt.strftime('%Y-%m-%d')
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
    else:  # china
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
        
        if start_date is None:
            start_date_dt = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)
            start_date = start_date_dt.strftime('%Y%m%d')
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')

    shared_columns = ['date', 'open', 'high', 'low', 'close', 'change', 'pct_change', 'volume']

    # Fetch data using Tushare or YFinance
    if market == "china":
        try:
            data = pro.daily(ts_code=ticker, start_date=start_date, end_date=end_date)
            data = data.copy().rename(columns={
                    'trade_date': 'date',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'change': 'change',
                    'pct_chg': 'pct_change',
                    'vol': 'volume'
                })
            data = data[shared_columns]
            data['date'] = data['date'].astype(str).apply(convert_date_string)
            data = data.iloc[::-1].reset_index(drop=True)
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    elif market in ["hk", "us"]:
        try:
            data = yf.download(ticker, start=start_date, end=end_date).reset_index()
            data['change'] = data['Close'].diff()
            data['pct_change'] = data['Close'].pct_change() * 100
            data = data.copy().rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'change': 'change',
                    'pct_change': 'pct_change'
                })
            data = data[shared_columns]
            data["date"] = data["date"].astype("str")
            data = data.fillna(0)
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
        
    # Round price columns to 2 decimal places
    price_columns = ['open', 'high', 'low', 'close', 'change']
    data[price_columns] = data[price_columns].round(2)
    data['pct_change'] = data['pct_change'].round(2)

    return data


