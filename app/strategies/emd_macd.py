from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from data.stock import get_stock_prices  # Ensure this custom module is available
import pytz

# ---------------------------
# EMD-MACD Analysis Function
# ---------------------------

def emd_macd_strategy(ticker):
    st.markdown(f"# 📈 Ehlers’ MACD均线 (EMD-MACD) for {ticker.upper()}")

    # Sidebar for user inputs specific to EMD-MACD Analysis
    st.sidebar.header("📊 指标参数 (Indicator Parameters)")

    # Function to convert period to start and end dates
    def convert_period_to_dates(period):
        # Define Beijing timezone
        beijing_tz = pytz.timezone('Asia/Shanghai')  # Beijing shares the same timezone as Shanghai

        # Get current time in Beijing
        end_date = datetime.now(beijing_tz)

        # Calculate start date based on the selected period
        if period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=365*2)
        elif period == "5y":
            start_date = end_date - timedelta(days=365*5)
        elif period == "10y":
            start_date = end_date - timedelta(days=365*10)
        else:
            start_date = end_date

        # Convert to 'yyyy-mm-dd' format
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    # User input function with additional EMD-MACD parameters
    def user_input_features():
        period = st.sidebar.selectbox(
            "📅 时间跨度 (Time Period)",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            help="选择分析的时间跨度。"
        )
        short_window = st.sidebar.number_input(
            "🔢 短期EMA窗口 (Short EMA Window)",
            min_value=1,
            max_value=100,
            value=12,
            help="短期EMA的窗口期，通常设为12。"
        )
        long_window = st.sidebar.number_input(
            "🔢 长期EMA窗口 (Long EMA Window)",
            min_value=1,
            max_value=200,
            value=26,
            help="长期EMA的窗口期，通常设为26。"
        )
        signal_window = st.sidebar.number_input(
            "🔢 信号线窗口 (Signal Line Window)",
            min_value=1,
            max_value=100,
            value=9,
            help="信号线的窗口期，通常设为9。"
        )
        smoothing = st.sidebar.number_input(
            "🔢 平滑参数 (Smoothing Parameter)",
            min_value=1,
            max_value=100,
            value=3,
            help="用于Savitzky-Golay滤波器的平滑参数，通常设为3。"
        )

        # Convert period to start and end dates
        start_date, end_date = convert_period_to_dates(period)

        return (
            start_date, end_date, short_window, long_window,
            signal_window, smoothing
        )

    # Getting user input
    (
        start_date, end_date, short_window, long_window,
        signal_window, smoothing
    ) = user_input_features()

    # Step 1: Fetch Historical Data using custom get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("❌ 未获取到数据。请检查股票代码并重试。")
        st.stop()

    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Step 2: Calculate Ehlers’ Modified MACD (EMD-MACD)
    def calculate_emd_macd(df, short_window=12, long_window=26, signal_window=9, smoothing=3):
        """
        Calculate Ehlers’ Modified MACD using zero-lag EMA and Savitzky-Golay filter for smoothing.
        """
        # Zero-lag EMA implementation (simple approximation)
        def zero_lag_ema(series, span):
            ema = series.ewm(span=span, adjust=False).mean()
            lag = (series - ema).shift(1)
            return ema + lag

        # Calculate zero-lag EMAs
        df['EMA_short'] = zero_lag_ema(df['close'], span=short_window)
        df['EMA_long'] = zero_lag_ema(df['close'], span=long_window)

        # MACD line
        df['MACD'] = df['EMA_short'] - df['EMA_long']

        # Apply Savitzky-Golay filter for smoothing the MACD line
        df['MACD_smooth'] = savgol_filter(df['MACD'].fillna(0), window_length=smoothing*2+1 if smoothing*2+1 < len(df) else len(df)//2*2+1, polyorder=2)

        # Signal line
        df['Signal'] = df['MACD_smooth'].ewm(span=signal_window, adjust=False).mean()

        # Histogram
        df['Histogram'] = df['MACD_smooth'] - df['Signal']

        return df

    df = calculate_emd_macd(df, short_window, long_window, signal_window, smoothing)

    # Step 3: Identify MACD Crossovers
    def identify_crossovers(df):
        """
        Identify bullish and bearish crossovers in the EMD-MACD.
        """
        df['Crossover'] = np.where(df['MACD_smooth'] > df['Signal'], 1, 0)
        df['Crossover_Signal'] = df['Crossover'].diff()

        bullish_crossovers = df[df['Crossover_Signal'] == 1]
        bearish_crossovers = df[df['Crossover_Signal'] == -1]

        return bullish_crossovers, bearish_crossovers

    bullish_crossovers, bearish_crossovers = identify_crossovers(df)

    # Step 4: Determine Market Trend Based on EMD-MACD
    def determine_trend(df):
        """
        Determine the current market trend based on EMD-MACD.
        """
        latest_histogram = df['Histogram'].iloc[-1]
        if latest_histogram > 0:
            trend = "上升趋势 (Uptrend)"
        elif latest_histogram < 0:
            trend = "下降趋势 (Downtrend)"
        else:
            trend = "震荡区间 (Sideways)"
        latest_price = df['close'].iloc[-1]
        return trend, latest_price

    trend, current_price = determine_trend(df)

    # Step 5: Plot Using Plotly
    def plot_emd_macd(df, bullish_crossovers, bearish_crossovers, ticker,
                     short_window=12, long_window=26, signal_window=9):
        """
        Plot the price data and EMD-MACD using Plotly.
        """
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{ticker.upper()} 的股价 (Price)', 'Ehlers’ Modified MACD (EMD-MACD)'),
            row_width=[0.2, 0.7]
        )

        # Candlestick for Price
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # EMD-MACD Line
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['MACD_smooth'],
                line=dict(color='blue', width=2),
                name='EMD-MACD'
            ),
            row=2, col=1
        )

        # Signal Line
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['Signal'],
                line=dict(color='orange', width=2),
                name='Signal Line'
            ),
            row=2, col=1
        )

        # Histogram
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['Histogram'],
                marker_color=np.where(df['Histogram'] >= 0, 'green', 'red'),
                name='Histogram'
            ),
            row=2, col=1
        )

        # Highlight Bullish Crossovers
        fig.add_trace(
            go.Scatter(
                x=bullish_crossovers['date'],
                y=bullish_crossovers['MACD_smooth'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name='Bullish Crossover'
            ),
            row=2, col=1
        )

        # Highlight Bearish Crossovers
        fig.add_trace(
            go.Scatter(
                x=bearish_crossovers['date'],
                y=bearish_crossovers['MACD_smooth'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='Bearish Crossover'
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f'Ehlers’ Modified MACD (EMD-MACD) for {ticker.upper()}',
            yaxis_title='Price',
            template='plotly_dark',
            showlegend=True,
            height=1000
        )

        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

        return fig

    fig = plot_emd_macd(
        df, bullish_crossovers, bearish_crossovers, ticker,
        short_window, long_window, signal_window
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Display Data Table
    with st.expander("📊 查看原始数据 (View Raw Data)"):
        st.dataframe(df)