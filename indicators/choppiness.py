from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from data.stock import get_stock_prices  # Ensure this module is available
from ta.trend import EMAIndicator
import pytz
import numpy as np

# ---------------------------
# Choppiness Index Analysis Function
# ---------------------------

def choppiness_analysis(ticker):
    st.markdown(f"# 📈 波动指数 (CI) for {ticker.upper()}")

    # Sidebar for user inputs specific to Choppiness Index Analysis
    st.sidebar.header("指标参数")

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
        else:
            start_date = end_date

        # Convert to 'YYYY-MM-DD' format
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    # User input function
    def user_input_features():
        period = st.sidebar.selectbox(
            "时间跨度 (Time Period)", 
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], 
            index=3
        )
        return convert_period_to_dates(period)

    # Getting user input
    start_date, end_date = user_input_features()

    # Step 1: Fetch Historical Data using custom get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("未获取到数据。请检查股票代码并重试。 (No data fetched. Please check the ticker symbol and try again.)")
        st.stop()

    # Step 2: User Inputs for Choppiness Index Parameters
    st.sidebar.header("波动指数参数")

    ci_period = st.sidebar.number_input(
        "周期 (Period)", 
        min_value=1, 
        max_value=100, 
        value=14,  # Common default
        step=1,
        help="用于计算 CI 的周期。推荐值：14。 (The period over which CI is calculated. Recommended value: 14.)"
    )

    # Additional Parameters for EMA
    st.sidebar.header("其他参数")

    ema_short_window = st.sidebar.number_input(
        "EMA 短期窗口 (EMA Short Window)", 
        min_value=10, 
        max_value=100, 
        value=50,  # Common default
        step=5,
        help="短期指数移动平均线（EMA）的窗口大小。推荐值：50。 (Short-term EMA window size. Recommended value: 50.)"
    )

    ema_long_window = st.sidebar.number_input(
        "EMA 长期窗口 (EMA Long Window)", 
        min_value=100, 
        max_value=300, 
        value=200,  # Common default
        step=10,
        help="长期指数移动平均线（EMA）的窗口大小。推荐值：200。 (Long-term EMA window size. Recommended value: 200.)"
    )

    # Plotting Options
    st.sidebar.header("绘图选项")
    show_ema = st.sidebar.checkbox("显示 EMA (Show EMAs)", value=True)
    show_ci = st.sidebar.checkbox("显示 CI (Show Choppiness Index)", value=True)

    # Define a custom function to calculate Choppiness Index (CI)
    def choppiness_index(df, window=14):
        """
        Calculate the Choppiness Index (CI) for a given DataFrame of stock data.
        
        Parameters:
        df (DataFrame): DataFrame containing 'high', 'low', and 'close' price data.
        window (int): Lookback period for the Choppiness Index calculation.
        
        Returns:
        Series: Choppiness Index values.
        """
        # Calculate the True Range (High - Low)
        true_range = df['high'] - df['low']
        
        # Sum of True Range over the window
        sum_true_range = true_range.rolling(window=window).sum()
        
        # Highest High and Lowest Low over the window
        highest_high = df['high'].rolling(window=window).max()
        lowest_low = df['low'].rolling(window=window).min()
        
        # Calculate Choppiness Index (CI)
        ci = 100 * np.log10(sum_true_range / (highest_high - lowest_low)) / np.log10(window)
        
        return ci

    # Example usage
    # Assuming `df` is your DataFrame containing stock data with 'high', 'low', 'close' columns
    df['CI'] = choppiness_index(df, window=ci_period)

    # Calculate EMAs using ta
    ema_short_indicator = EMAIndicator(close=df['close'], window=ema_short_window)
    ema_long_indicator = EMAIndicator(close=df['close'], window=ema_long_window)
    df['EMA_Short'] = ema_short_indicator.ema_indicator()
    df['EMA_Long'] = ema_long_indicator.ema_indicator()

    # Step 3: Plot Using Plotly
    def plot_ci(df, ticker, show_ema=True, show_ci=True):
        """
        Plot the Choppiness Index along with price data and EMAs using Plotly.
        """
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, 
            subplot_titles=(f'{ticker}的股价和移动平均线 (Price and EMAs)', 'Choppiness Index (CI)'),
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

        # EMAs
        if show_ema:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['EMA_Short'], 
                    line=dict(color='blue', width=1), 
                    name=f'EMA{ema_short_window}'
                ), 
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['EMA_Long'], 
                    line=dict(color='purple', width=1), 
                    name=f'EMA{ema_long_window}'
                ), 
                row=1, col=1
            )

        # CI Line
        if show_ci:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['CI'], 
                    line=dict(color='orange', width=2), 
                    name='CI'
                ), 
                row=2, col=1
            )
            # Add key levels for CI (e.g., 61.8, 38.2)
            fig.add_hline(
                y=61.8, 
                line=dict(color='gray', dash='dash'), 
                row=2, col=1, annotation_text="Choppy Market"
            )
            fig.add_hline(
                y=38.2, 
                line=dict(color='gray', dash='dash'), 
                row=2, col=1, annotation_text="Trending Market"
            )

        # Update Layout
        fig.update_layout(
            title=f"{ticker} 的 Choppiness Index (CI) 分析",
            yaxis_title='价格 (Price)',
            xaxis_title='',
            template='plotly_dark',
            showlegend=True,
            height=900
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    fig = plot_ci(df, ticker, show_ema=show_ema, show_ci=show_ci)
    st.plotly_chart(fig, use_container_width=True)

    # Step 4: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation(df, current_price):
        """
        Provide a detailed, actionable interpretation based on Choppiness Index in both English and Chinese.
        """
        latest_ci = df['CI'].iloc[-1]
        interpretation_en = f"###### Choppiness Index: {latest_ci:.2f}\n\n"
        interpretation_cn = f"###### Choppiness Index (CI)：{latest_ci:.2f}\n\n"

        # Interpretation based on CI value
        if latest_ci > 61.8:
            interpretation_en += "- **Choppy Market**: The market is likely range-bound with no clear trend.\n"
            interpretation_cn += "- **市场震荡**：市场可能处于震荡状态，没有明确趋势。\n"
        elif latest_ci < 38.2:
            interpretation_en += "- **Trending Market**: The market is showing a clear trend, ideal for trend-following strategies.\n"
            interpretation_cn += "- **趋势市场**：市场呈现明显趋势，适合跟随趋势的交易策略。\n"
        else:
            interpretation_en += "- **Neutral Market**: The market is not showing significant choppiness or trend.\n"
            interpretation_cn += "- **中性市场**：市场未显示显著的震荡或趋势。\n"

        return interpretation_en, interpretation_cn

    interpret_en, interpret_cn = detailed_interpretation(df, df['close'].iloc[-1])

    # Display Interpretations
    st.markdown("##### 📄 指标解读 (Indicator Interpretation)")

    # Tabs for English and Chinese
    tab1, tab2 = st.tabs(["中文", "English"])

    with tab1:
        st.markdown(interpret_cn)

    with tab2:
        st.markdown(interpret_en)

    # Optional: Display Data Table
    with st.expander("📊 查看原始数据 (View Raw Data)"):
        st.dataframe(df)
