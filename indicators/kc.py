from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf  # Using yfinance for data retrieval
from ta.volatility import KeltnerChannel

from data.stock import get_stock_prices  # Placeholder function for stock data retrieval

# ---------------------------
# Keltner Channel Analysis Function
# ---------------------------

def keltner_channel_analysis(ticker):
    st.markdown(f"# 📊 肯特纳通道 (KC) for {ticker.upper()}")

    # Sidebar for user inputs specific to Keltner Channels
    st.sidebar.header("指标参数")

    # Function to convert period to start and end dates
    def convert_period_to_dates(period):
        # Get the current date
        end_date = datetime.now()
        
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
            start_date = end_date - timedelta(days=365 * 2)
        else:
            start_date = end_date

        # Convert to 'YYYY-MM-DD' format
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    # User input for the time period
    def user_input_features():
        period = st.sidebar.selectbox(
            "时间跨度 (Time Period)", 
            options=["1mo", "3mo", "6mo", "1y", "2y"], 
            index=3
        )
        return convert_period_to_dates(period)

    # Getting user input
    start_date, end_date = user_input_features()

    # Step 1: Fetch Historical Data using get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("未获取到数据。请检查股票代码并重试。 (No data fetched. Please check the ticker symbol and try again.)")
        st.stop()

    # Step 2: User Inputs for Keltner Channel Parameters
    st.sidebar.header("肯特纳通道参数")

    # Keltner Channel Parameters
    kc_window = st.sidebar.number_input(
        "窗口周期 (KC Window)", 
        min_value=10, 
        max_value=100, 
        value=20,  # Default value
        step=1,
        help="Keltner 通道的窗口周期。推荐值：20。 (Window size for Keltner Channels. Recommended value: 20.)"
    )
    atr_window = st.sidebar.number_input(
        "ATR 周期 (ATR Window)", 
        min_value=5, 
        max_value=50, 
        value=10,  # Default value
        step=1,
        help="用于计算真实波幅的 ATR 窗口周期。推荐值：10。 (Window size for Average True Range. Recommended value: 10.)"
    )
    atr_multiplier = st.sidebar.number_input(
        "ATR 倍数 (ATR Multiplier)", 
        min_value=1.0, 
        max_value=5.0, 
        value=2.0,  # Default value
        step=0.1,
        help="ATR 的倍数决定通道的宽度。推荐值：2.0。 (ATR multiplier for channel width. Recommended value: 2.0.)"
    )

    # Step 3: Calculate Keltner Channels using ta library
    kc_indicator = KeltnerChannel(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=kc_window, 
        window_atr=atr_window, 
        original_version=False
    )
    df['KC_Center'] = kc_indicator.keltner_channel_mband()
    df['KC_Upper'] = kc_indicator.keltner_channel_hband()
    df['KC_Lower'] = kc_indicator.keltner_channel_lband()

    # Step 4: Plot Keltner Channels with Price using Plotly
    def plot_keltner_channels(df, ticker):
        fig = make_subplots(
            rows=1, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, 
            subplot_titles=(f'{ticker} 的股价与 Keltner 通道 (Price and Keltner Channels)'),
            row_width=[0.2]
        )

        # Candlestick for Price
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='价格 (Price)'
            ), 
            row=1, col=1
        )

        # Keltner Channels
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['KC_Center'], 
                line=dict(color='blue', width=1), 
                name='KC 中线 (KC Center)'
            ), 
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['KC_Upper'], 
                line=dict(color='green', width=1, dash='dot'), 
                name='KC 上轨 (KC Upper)'
            ), 
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['KC_Lower'], 
                line=dict(color='red', width=1, dash='dot'), 
                name='KC 下轨 (KC Lower)'
            ), 
            row=1, col=1
        )

        fig.update_layout(
            title=f"{ticker} 的 Keltner Channel 分析",
            yaxis_title='价格 (Price)',
            xaxis_title='日期 (Date)',
            template='plotly_dark',
            showlegend=True,
            height=800
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    fig = plot_keltner_channels(df, ticker)
    st.plotly_chart(fig, use_container_width=True)

    # Step 5: Detailed Interpretation of Keltner Channels
    def detailed_interpretation(df):
        latest_price = df['close'].iloc[-1]
        kc_upper = df['KC_Upper'].iloc[-1]
        kc_lower = df['KC_Lower'].iloc[-1]

        interpretation_en = f"""
        ### Interpretation for {ticker.upper()}
        
        - **Current Price**: {latest_price:.2f}
        - **Keltner Channel Upper Band**: {kc_upper:.2f}
        - **Keltner Channel Lower Band**: {kc_lower:.2f}
        
        #### Key Insights:
        - If the price is touching or crossing above the upper band, this may indicate that the stock is **overbought**, potentially signaling a reversal or a continuation of the uptrend.
        - If the price is touching or crossing below the lower band, this may indicate that the stock is **oversold**, potentially signaling a reversal or a continuation of the downtrend.
        - Price staying within the bands typically indicates a **neutral** market with less volatility.
        """

        interpretation_cn = f"""
        ### {ticker.upper()} 的解读

        - **当前价格**：{latest_price:.2f}
        - **Keltner 通道上轨**：{kc_upper:.2f}
        - **Keltner 通道下轨**：{kc_lower:.2f}

        #### 关键见解：
        - 如果价格触及或突破上轨，这可能表明股票 **超买**，可能暗示反转或上涨趋势的延续。
        - 如果价格触及或跌破下轨，这可能表明股票 **超卖**，可能暗示反转或下跌趋势的延续。
        - 价格在通道内波动通常表示 **中性** 市场，波动性较小。
        """

        return interpretation_en, interpretation_cn

    # Display the interpretation
    st.markdown("##### 📄 指标解读")

    # Tabs for English and Chinese interpretations
    tab1, tab2 = st.tabs(["中文", "English"])

    interpret_en, interpret_cn = detailed_interpretation(df)

    with tab1:
        st.markdown(interpret_cn)

    with tab2:
        st.markdown(interpret_en)

    # Optional: Display Raw Data
    with st.expander("📊 查看原始数据 (View Raw Data)"):
        st.dataframe(df)
