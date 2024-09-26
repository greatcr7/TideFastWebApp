from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from data.stock import get_stock_prices  # Ensure this module is available
from ta.volatility import BollingerBands
import pytz

# ---------------------------
# Bollinger Bands Analysis Function
# ---------------------------

def bollinger_band_analysis(ticker):
    st.markdown(f"# 📈 布林带 (Bollinger Bands)")
    
    # Sidebar for user inputs specific to Bollinger Bands Analysis
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
        elif period == "10y":
            start_date = end_date - timedelta(days=365*10)
        else:
            start_date = end_date
        
        # Convert to 'YYYY-MM-DD' format
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    # User input function
    def user_input_features():
        period = st.sidebar.selectbox("时间跨度 (Time Period)", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=3)
        return convert_period_to_dates(period)
    
    # Getting user input
    start_date, end_date = user_input_features()
    
    # Step 1: Fetch Historical Data using custom get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)
    
    if df is None or df.empty:
        st.error("未获取到数据。请检查股票代码并重试。 (No data fetched. Please check the ticker symbol and try again.)")
        st.stop()
    
    # Step 2: User Inputs for Bollinger Bands Parameters
    st.sidebar.header("布林带参数")
    
    bb_window = st.sidebar.number_input(
        "移动窗口 (Moving Window)", 
        min_value=5, 
        max_value=100, 
        value=20,  # Common default
        step=1,
        help="布林带的移动窗口期。推荐值：20。"
    )
    
    bb_std = st.sidebar.number_input(
        "标准差倍数 (Standard Deviation Multiplier)", 
        min_value=1.0, 
        max_value=5.0, 
        value=2.0,  # Common default
        step=0.1,
        help="用于计算上下布林带的标准差倍数。推荐值：2.0。"
    )
    
    # Plotting Options
    st.sidebar.header("绘图选项 (Plotting Options)")
    show_bbands = st.sidebar.checkbox("显示布林带 (Show Bollinger Bands)", value=True)
    show_ma = st.sidebar.checkbox("显示移动平均线 (Show Moving Average)", value=True)
    
    # Calculate Bollinger Bands using ta
    bb_indicator = BollingerBands(close=df['close'], window=bb_window, window_dev=bb_std)
    df['BB_High'] = bb_indicator.bollinger_hband()
    df['BB_Low'] = bb_indicator.bollinger_lband()
    df['BB_Middle'] = bb_indicator.bollinger_mavg()
    
    # Identify Signals
    def identify_bollinger_signals(df):
        buy_signals = []
        sell_signals = []
        
        for i in range(1, len(df)):
            # Buy Signal: Price crosses above the lower band
            if (df['close'].iloc[i-1] < df['BB_Low'].iloc[i-1]) and (df['close'].iloc[i] > df['BB_Low'].iloc[i]):
                buy_signals.append({
                    'Date': df['date'].iloc[i],
                    'Price': df['close'].iloc[i],
                    'BB_Low': df['BB_Low'].iloc[i]
                })
            # Sell Signal: Price crosses below the upper band
            if (df['close'].iloc[i-1] > df['BB_High'].iloc[i-1]) and (df['close'].iloc[i] < df['BB_High'].iloc[i]):
                sell_signals.append({
                    'Date': df['date'].iloc[i],
                    'Price': df['close'].iloc[i],
                    'BB_High': df['BB_High'].iloc[i]
                })
        
        return buy_signals, sell_signals
    
    buy_signals, sell_signals = identify_bollinger_signals(df)
    
    # Determine Trend Based on Bollinger Bands
    def determine_trend_bbands(df):
        latest_close = df['close'].iloc[-1]
        latest_bb_middle = df['BB_Middle'].iloc[-1]
        if latest_close > latest_bb_middle:
            return "上升趋势 (Uptrend)", latest_close
        elif latest_close < latest_bb_middle:
            return "下降趋势 (Downtrend)", latest_close
        else:
            return "震荡区间 (Sideways)", latest_close
    
    trend, current_price = determine_trend_bbands(df)
    
    # Step 3: Plot Using Plotly
    def plot_bollinger_bands(df, buy_signals, sell_signals, ticker, show_bbands=True, show_ma=True):
        """
        Plot the Bollinger Bands along with price data using Plotly.
        """
        fig = make_subplots(
            rows=1, cols=1,
            shared_xaxes=True,
            subplot_titles=(f'{ticker}的股价与布林带',),
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
            )
        )
        
        # Bollinger Bands
        if show_bbands:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['BB_High'],
                    line=dict(color='rgba(255, 165, 0, 0.5)', width=3),  # Orange for upper band
                    name='布林带上轨 (BB Upper)'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['BB_Low'],
                    line=dict(color='rgba(255, 165, 0, 0.5)', width=3),  # Orange for lower band
                    name='布林带下轨 (BB Lower)'
                )
            )
            # Fill between upper and lower bands
            fig.add_traces([
                go.Scatter(
                    x=df['date'],
                    y=df['BB_High'],
                    line=dict(color='rgba(255, 165, 0, 0.5)', width=0),
                    hoverinfo='none',
                    showlegend=False
                ),
                go.Scatter(
                    x=df['date'],
                    y=df['BB_Low'],
                    line=dict(color='rgba(255, 165, 0, 0.5)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 165, 0, 0.1)',
                    hoverinfo='none',
                    showlegend=False
                )
            ])
        
        # Moving Average
        if show_ma:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['BB_Middle'],
                    line=dict(color='blue', width=1),
                    name=f'中轨移动平均 (MA{bb_window})'
                )
            )
        
        # Buy Signals
        buy_dates = [signal['Date'] for signal in buy_signals]
        buy_prices = [signal['Price'] for signal in buy_signals]
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=buy_dates,
                y=buy_prices,
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='买入信号 (Buy Signal)'
            )
        )
        
        # Sell Signals
        sell_dates = [signal['Date'] for signal in sell_signals]
        sell_prices = [signal['Price'] for signal in sell_signals]
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=sell_dates,
                y=sell_prices,
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='卖出信号 (Sell Signal)'
            )
        )
        
        # Update Layout
        fig.update_layout(
            title=f"{ticker} Bollinger Bands Analysis",
            yaxis_title='价格 (Price)',
            xaxis_title='日期 (Date)',
            template='plotly_dark',
            showlegend=True,
            height=800
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    fig = plot_bollinger_bands(df, buy_signals, sell_signals, ticker, show_bbands=show_bbands, show_ma=show_ma)
    st.plotly_chart(fig, use_container_width=True)
    
    # Step 4: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation_bb(buy_signals, sell_signals, current_price, trend):
        """
        Provide a detailed, actionable interpretation based on Bollinger Bands in both English and Chinese.
        """
        interpretation_en = ""
        interpretation_cn = ""
        
        # 1. Trend Analysis
        interpretation_en += f"###### Current Market Trend: {trend}\n\n"
        interpretation_en += f"**Current Price**: ${current_price:.2f}\n\n"
        
        interpretation_cn += f"###### 当前市场趋势：{trend}\n\n"
        interpretation_cn += f"**当前价格**：${current_price:.2f}\n\n"
        
        # 2. Signals Analysis
        if buy_signals or sell_signals:
            interpretation_en += "###### Trading Signals Detected:\n"
            interpretation_cn += "###### 检测到的交易信号：\n"
            if buy_signals:
                for signal in buy_signals:
                    interpretation_en += f"- **Buy Signal** on {signal['Date']}: Price crossed above the lower Bollinger Band at ${signal['Price']:.2f}.\n"
                    interpretation_cn += f"- **买入信号** 于 {signal['Date']}：价格上穿布林带下轨，价格为 ${signal['Price']:.2f}。\n"
            if sell_signals:
                for signal in sell_signals:
                    interpretation_en += f"- **Sell Signal** on {signal['Date']}: Price crossed below the upper Bollinger Band at ${signal['Price']:.2f}.\n"
                    interpretation_cn += f"- **卖出信号** 于 {signal['Date']}：价格下穿布林带上轨，价格为 ${signal['Price']:.2f}。\n"
            interpretation_en += "\n"
            interpretation_cn += "\n"
        else:
            interpretation_en += "###### No Trading Signals Detected.\n\n"
            interpretation_cn += "###### 未检测到交易信号。\n\n"
        
        # 3. Price Position Analysis
        interpretation_en += "###### Price Position Relative to Bollinger Bands:\n"
        interpretation_cn += "###### 当前价格相对于布林带的位置：\n"
        if trend == "上升趋势 (Uptrend)":
            interpretation_en += "- The current price is **above** the middle Bollinger Band, indicating an upward momentum.\n"
            interpretation_cn += "- 当前价格 **高于** 布林带中轨，表明上升动能。\n"
        elif trend == "下降趋势 (Downtrend)":
            interpretation_en += "- The current price is **below** the middle Bollinger Band, indicating a downward momentum.\n"
            interpretation_cn += "- 当前价格 **低于** 布林带中轨，表明下降动能。\n"
        else:
            interpretation_en += "- The current price is **around** the middle Bollinger Band, indicating a sideways or consolidating market.\n"
            interpretation_cn += "- 当前价格 **接近** 布林带中轨，表明横盘或整合市场。\n"
        interpretation_en += "\n"
        interpretation_cn += "\n"
        
        # 4. Actionable Recommendations
        interpretation_en += "###### Actionable Recommendations:\n"
        interpretation_cn += "###### 可操作的建议：\n"
        
        # Buy Signals
        if buy_signals:
            interpretation_en += "- **Buying Opportunity**: Consider buying when the price crosses above the lower Bollinger Band, indicating potential upward movement.\n"
            interpretation_cn += "- **买入机会**：当价格上穿布林带下轨，表明可能的上升走势时，考虑买入。\n"
        
        # Sell Signals
        if sell_signals:
            interpretation_en += "- **Selling Opportunity**: Consider selling when the price crosses below the upper Bollinger Band, indicating potential downward movement.\n"
            interpretation_cn += "- **卖出机会**：当价格下穿布林带上轨，表明可能的下降走势时，考虑卖出。\n"
        
        # No Signals
        if not buy_signals and not sell_signals:
            interpretation_en += "- **Hold Position**: In the absence of clear signals, consider holding your current position and monitor the price movements.\n"
            interpretation_cn += "- **持有仓位**：在缺乏明确信号的情况下，考虑持有当前仓位并监控价格走势。\n"
        
        # Risk Management
        interpretation_en += "\n###### Risk Management:\n"
        interpretation_cn += "\n###### 风险管理：\n"
        interpretation_en += "- **Stop-Loss**: Place stop-loss orders just outside the Bollinger Bands to limit potential losses.\n"
        interpretation_cn += "- **止损**：在布林带之外设置止损订单以限制潜在损失。\n"
        interpretation_en += "- **Take-Profit**: Set target levels based on historical support/resistance or use trailing stops to secure profits.\n"
        interpretation_cn += "- **止盈**：根据历史支撑/阻力位设置目标水平或使用移动止盈以确保利润。\n"
        
        # Market Conditions
        interpretation_en += "\n###### Optimal Market Conditions for Applying This Strategy:\n"
        interpretation_cn += "\n###### 应用此策略的最佳市场条件：\n"
        interpretation_en += "- **Trending Markets**: Effective in markets showing clear upward or downward trends where prices interact with Bollinger Bands.\n"
        interpretation_cn += "- **趋势市场**：在价格与布林带交互显示明显上升或下降趋势的市场中有效。\n"
        interpretation_en += "- **Volatile Markets**: Suitable for volatile markets where price frequently touches or breaks the Bollinger Bands.\n"
        interpretation_cn += "- **高波动市场**：适用于价格经常触及或突破布林带的高波动市场。\n"
        interpretation_en += "- **Avoid in Stable Markets**: In stable or low volatility markets, Bollinger Bands may produce fewer actionable signals.\n"
        interpretation_cn += "- **避免在稳定市场**：在稳定或低波动市场中，布林带可能产生较少的可操作信号。\n"
        
        return interpretation_en, interpretation_cn
    
    interpret_en, interpret_cn = detailed_interpretation_bb(
        buy_signals, sell_signals, current_price, trend
    )
    
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

# ---------------------------
# Main Streamlit App
# ---------------------------

def main():
    st.title("📊 技术分析工具 (Technical Analysis Tools)")
    
    # User inputs for ticker
    ticker = st.text_input("请输入股票代码 (Enter Stock Ticker)", value="AAPL")
    
    if ticker:
        bollinger_band_analysis(ticker.upper())

if __name__ == "__main__":
    main()