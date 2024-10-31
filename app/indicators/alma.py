from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import pandas as pd
from data.stock import get_stock_prices  # Assumed custom module
import pytz
import numpy as np

# ---------------------------
# Arnaud Legoux Moving Average (ALMA) Analysis Function
# ---------------------------

def alma_analysis(ticker):
    st.markdown(f"# 📈 ALMA 移动平均数 for {ticker.upper()}")

    # Sidebar for user inputs specific to ALMA Analysis
    st.sidebar.header("📊 指标参数")

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

    # User input function with additional ALMA parameters
    def user_input_features():
        period = st.sidebar.selectbox(
            "📅 时间跨度 (Time Period)",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            help="选择分析的时间跨度。"
        )
        alma_window = st.sidebar.number_input(
            "🔢 ALMA 窗口 (ALMA Window)",
            min_value=1,
            max_value=100,
            value=14,
            help="ALMA计算的窗口期，通常设为14。"
        )
        alma_offset = st.sidebar.number_input(
            "🔢 ALMA 偏移 (ALMA Offset)",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.01,
            help="ALMA计算的偏移，通常设为0.85。"
        )
        alma_sigma = st.sidebar.number_input(
            "🔢 ALMA Sigma",
            min_value=0.1,
            max_value=10.0,
            value=6.0,
            step=0.1,
            help="ALMA计算的Sigma值，通常设为6.0。"
        )
        ema50_period = st.sidebar.number_input(
            "📊 EMA50 周期 (EMA50 Period)",
            min_value=1,
            max_value=200,
            value=50,
            help="计算50期指数移动平均线的周期，通常设为50。"
        )
        ema200_period = st.sidebar.number_input(
            "📊 EMA200 周期 (EMA200 Period)",
            min_value=1,
            max_value=500,
            value=200,
            help="计算200期指数移动平均线的周期，通常设为200。"
        )
        peaks_prominence = st.sidebar.number_input(
            "🔝 峰值显著性 (Peak Prominence)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="峰值检测时的显著性要求，通常设为1.0。"
        )

        # Convert period to start and end dates
        start_date, end_date = convert_period_to_dates(period)

        return (
            start_date, end_date, alma_window, alma_offset,
            alma_sigma, ema50_period, ema200_period,
            peaks_prominence
        )

    # Getting user input
    (
        start_date, end_date, alma_window, alma_offset,
        alma_sigma, ema50_period, ema200_period,
        peaks_prominence
    ) = user_input_features()

    # Step 1: Fetch Historical Data using custom get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("❌ 未获取到数据。请检查股票代码并重试。")
        st.stop()

    # Step 2: Calculate Arnaud Legoux Moving Average (ALMA)
    def calculate_alma(df, window=14, offset=0.85, sigma=6.0):
        """
        Calculate Arnaud Legoux Moving Average (ALMA).
        """
        df = df.copy()
        def alma(series, window, offset, sigma):
            m = offset * (window - 1)
            s = window / sigma
            weights = np.exp(-((np.arange(window) - m) ** 2) / (2 * s ** 2))
            weights /= weights.sum()
            return np.convolve(series, weights, mode='same')

        df['ALMA'] = alma(df['close'].values, window, offset, sigma)
        return df

    df = calculate_alma(df, window=alma_window, offset=alma_offset, sigma=alma_sigma)

    # Step 3: Identify ALMA Crossovers
    def identify_alma_signals(df, peaks_prominence=1.0):
        """
        Identify potential buy and sell signals based on ALMA crossovers with price.
        """
        buy_signals = []
        sell_signals = []

        # Buy Signal: Price crosses above ALMA
        crossover_buy = df['close'] > df['ALMA']
        crossover_buy_shift = df['close'].shift(1) <= df['ALMA'].shift(1)
        buy_indices = df.index[crossover_buy & crossover_buy_shift]

        for idx in buy_indices:
            buy_signals.append({
                'Date': df['date'].iloc[idx],
                'Price': df['close'].iloc[idx],
                'ALMA': df['ALMA'].iloc[idx]
            })

        # Sell Signal: Price crosses below ALMA
        crossover_sell = df['close'] < df['ALMA']
        crossover_sell_shift = df['close'].shift(1) >= df['ALMA'].shift(1)
        sell_indices = df.index[crossover_sell & crossover_sell_shift]

        for idx in sell_indices:
            sell_signals.append({
                'Date': df['date'].iloc[idx],
                'Price': df['close'].iloc[idx],
                'ALMA': df['ALMA'].iloc[idx]
            })

        return buy_signals, sell_signals

    buy_signals, sell_signals = identify_alma_signals(df, peaks_prominence=peaks_prominence)

    # Step 4: Identify Confluence with Exponential Moving Averages (EMA)
    def find_confluence(df, ema50_period=50, ema200_period=200):
        """
        Identify if ALMA aligns with other moving averages.
        """
        # Calculate EMAs
        df['EMA50'] = df['close'].ewm(span=ema50_period, adjust=False).mean()
        df['EMA200'] = df['close'].ewm(span=ema200_period, adjust=False).mean()

        latest_alma = df['ALMA'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]
        latest_price = df['close'].iloc[-1]

        confluence_levels = {}

        # Define confluence based on ALMA alignment with EMAs
        if latest_alma > latest_ema50 and latest_alma > latest_ema200:
            confluence_levels['Bullish Confluence'] = {
                'ALMA': latest_alma,
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }
        elif latest_alma < latest_ema50 and latest_alma < latest_ema200:
            confluence_levels['Bearish Confluence'] = {
                'ALMA': latest_alma,
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }

        return confluence_levels, df

    confluences, df = find_confluence(df, ema50_period=ema50_period, ema200_period=ema200_period)

    # Step 5: Determine Market Trend Based on ALMA and EMAs
    def determine_trend(df, confluences):
        """
        Determine the current market trend based on ALMA and EMAs.
        """
        latest_alma = df['ALMA'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]
        latest_price = df['close'].iloc[-1]

        if latest_alma > latest_ema50 and latest_alma > latest_ema200:
            trend = "上升趋势 (Uptrend)"
        elif latest_alma < latest_ema50 and latest_alma < latest_ema200:
            trend = "下降趋势 (Downtrend)"
        else:
            trend = "震荡区间 (Sideways)"

        return trend, latest_price

    trend, current_price = determine_trend(df, confluences)

    # Step 6: Plot Using Plotly
    def plot_alma(df, buy_signals, sell_signals, confluences, ticker,
                 alma_window=14, alma_offset=0.85, alma_sigma=6.0,
                 ema50_period=50, ema200_period=200):
        """
        Plot the ALMA along with price data and EMAs using Plotly.
        """
        fig = make_subplots(
            rows=1, cols=1,
            shared_xaxes=True,
            subplot_titles=(f'{ticker.upper()} 的股价和 Arnaud Legoux 移动平均线 (ALMA)'),
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
                name='Price'
            ),
            row=1, col=1
        )

        # EMAs
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['EMA50'],
                line=dict(color='blue', width=1),
                name=f'EMA{ema50_period}'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['EMA200'],
                line=dict(color='purple', width=1),
                name=f'EMA{ema200_period}'
            ),
            row=1, col=1
        )

        # ALMA Line
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['ALMA'],
                line=dict(color='orange', width=2),
                name='ALMA'
            ),
            row=1, col=1
        )

        # Highlight Buy Signals
        for signal in buy_signals:
            fig.add_annotation(
                x=signal['Date'], y=signal['Price'],
                text="Buy",
                showarrow=True,
                arrowhead=1,
                ax=0, ay=-40,
                arrowcolor='green',
                row=1, col=1
            )

        # Highlight Sell Signals
        for signal in sell_signals:
            fig.add_annotation(
                x=signal['Date'], y=signal['Price'],
                text="Sell",
                showarrow=True,
                arrowhead=1,
                ax=0, ay=40,
                arrowcolor='red',
                row=1, col=1
            )

        # Highlight Confluence Zones
        for key, value in confluences.items():
            if key == 'Bullish Confluence':
                color = 'green'
            elif key == 'Bearish Confluence':
                color = 'red'
            else:
                color = 'yellow'
            fig.add_vline(
                x=value['Date'] if 'Date' in value else df['date'].iloc[-1],
                line=dict(color=color, dash='dot'),
                row=1, col=1
            )

        fig.update_layout(
            title=f'Arnaud Legoux Moving Average (ALMA) 分析 for {ticker.upper()}',
            yaxis_title='Price',
            xaxis_title='',
            template='plotly_dark',
            showlegend=True,
            height=800
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    fig = plot_alma(
        df, buy_signals, sell_signals, confluences, ticker,
        alma_window=alma_window, alma_offset=alma_offset, alma_sigma=alma_sigma,
        ema50_period=ema50_period, ema200_period=ema200_period
    )
    st.plotly_chart(fig, use_container_width=True)

    # Step 7: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation(
        buy_signals, sell_signals, confluences,
        current_price, trend, alma_window, alma_offset, alma_sigma
    ):
        """
        Provide a detailed, actionable interpretation based on ALMA and crossovers in both English and Chinese.
        """
        interpretation_en = ""
        interpretation_cn = ""

        # 1. Trend Analysis
        interpretation_en += f"###### Current Market Trend: {trend}\n\n"
        interpretation_en += f"**Current Price**: {current_price:.2f}\n\n"

        interpretation_cn += f"###### 当前市场趋势：{trend}\n\n"
        interpretation_cn += f"**当前价格**：{current_price:.2f}\n\n"

        # 2. Confluence Analysis
        if confluences:
            interpretation_en += "###### Confluence Zones Detected:\n"
            interpretation_cn += "###### 检测到的共振区：\n"
            for key, indicators in confluences.items():
                if key == 'Bullish Confluence':
                    interpretation_en += (
                        f"- **Bullish Confluence**: ALMA is above EMA{ema50_period} ({indicators['EMA50']:.2f}) and EMA{ema200_period} ({indicators['EMA200']:.2f}), indicating strong bullish momentum.\n"
                    )
                    interpretation_cn += (
                        f"- **看涨共振区**：ALMA 高于 EMA{ema50_period} ({indicators['EMA50']:.2f}) 和 EMA{ema200_period} ({indicators['EMA200']:.2f})，表明强劲的看涨动能。\n"
                    )
                elif key == 'Bearish Confluence':
                    interpretation_en += (
                        f"- **Bearish Confluence**: ALMA is below EMA{ema50_period} ({indicators['EMA50']:.2f}) and EMA{ema200_period} ({indicators['EMA200']:.2f}), indicating strong bearish momentum.\n"
                    )
                    interpretation_cn += (
                        f"- **看跌共振区**：ALMA 低于 EMA{ema50_period} ({indicators['EMA50']:.2f}) 和 EMA{ema200_period} ({indicators['EMA200']:.2f})，表明强劲的看跌动能。\n"
                    )
            interpretation_en += "\n"
            interpretation_cn += "\n"
        else:
            interpretation_en += "###### No Confluence Zones Detected.\n\n"
            interpretation_cn += "###### 未检测到共振区。\n\n"

        # 3. Price Position Analysis
        interpretation_en += "###### Price Position Relative to ALMA and EMAs:\n"
        interpretation_cn += "###### 当前价格相对于 ALMA 和 EMA 的位置：\n"
        if trend == "上升趋势 (Uptrend)":
            interpretation_en += f"- The current price is **above** EMA{ema50_period} and EMA{ema200_period}, and above ALMA, indicating a potential **buy** signal.\n"
            interpretation_cn += f"- 当前价格 **高于** EMA{ema50_period} 和 EMA{ema200_period}，且高于 ALMA，表明可能的 **买入** 信号。\n"
        elif trend == "下降趋势 (Downtrend)":
            interpretation_en += f"- The current price is **below** EMA{ema50_period} and EMA{ema200_period}, and below ALMA, indicating a potential **sell** signal.\n"
            interpretation_cn += f"- 当前价格 **低于** EMA{ema50_period} 和 EMA{ema200_period}，且低于 ALMA，表明可能的 **卖出** 信号。\n"
        else:
            interpretation_en += f"- The current price is **between** EMA{ema50_period} and EMA{ema200_period}, with no clear ALMA signal, indicating a sideways or consolidating market.\n"
            interpretation_cn += f"- 当前价格 **位于** EMA{ema50_period} 和 EMA{ema200_period} 之间，且无明显的 ALMA 信号，表明横盘或整合市场。\n"
        interpretation_en += "\n"
        interpretation_cn += "\n"

        # 4. Actionable Recommendations
        interpretation_en += "###### Actionable Recommendations:\n"
        interpretation_cn += "###### 可操作的建议：\n"

        # Buy Signals
        if buy_signals:
            interpretation_en += (
                f"- **Buying Opportunity**: {len(buy_signals)} buy signal(s) detected based on ALMA crossover. Consider buying when the price crosses above ALMA (Window: {alma_window}, Offset: {alma_offset}, Sigma: {alma_sigma}).\n"
            )
            interpretation_cn += (
                f"- **买入机会**：检测到 {len(buy_signals)} 个基于 ALMA 交叉的买入信号。考虑在价格突破 ALMA（窗口期：{alma_window}，偏移：{alma_offset}，Sigma：{alma_sigma}）时买入。\n"
            )

        # Sell Signals
        if sell_signals:
            interpretation_en += (
                f"- **Selling Opportunity**: {len(sell_signals)} sell signal(s) detected based on ALMA crossover. Consider selling when the price crosses below ALMA (Window: {alma_window}, Offset: {alma_offset}, Sigma: {alma_sigma}).\n"
            )
            interpretation_cn += (
                f"- **卖出机会**：检测到 {len(sell_signals)} 个基于 ALMA 交叉的卖出信号。考虑在价格突破 ALMA（窗口期：{alma_window}，偏移：{alma_offset}，Sigma：{alma_sigma}）时卖出。\n"
            )

        # Confluence Zones
        if confluences:
            interpretation_en += "\n- **Confluence Zones**: Trades near these areas have a higher probability of success due to the alignment of ALMA with EMAs.\n"
            interpretation_cn += "\n- **共振区**：由于 ALMA 与 EMA 对齐，接近这些区域的交易成功概率更高。\n"

        # Risk Management
        interpretation_en += "\n###### Risk Management:\n"
        interpretation_cn += "\n###### 风险管理：\n"
        interpretation_en += f"- **Stop-Loss**: Place stop-loss orders just beyond ALMA to manage risk.\n"
        interpretation_cn += f"- **止损**：在 ALMA 之外稍微放置止损订单以管理风险。\n"
        interpretation_en += f"- **Take-Profit**: Set target levels based on recent support/resistance levels or use a trailing stop to lock in profits.\n"
        interpretation_cn += f"- **止盈**：根据近期的支撑/阻力位设置目标水平或使用移动止盈以锁定利润。\n"

        # Market Conditions
        interpretation_en += "\n###### Optimal Market Conditions for Applying This Strategy:\n"
        interpretation_cn += "\n###### 应用此策略的最佳市场条件：\n"
        interpretation_en += "- **Trending Markets**: Most effective in clear uptrends or downtrends where ALMA and EMAs confirm the direction.\n"
        interpretation_cn += "- **趋势市场**：在 ALMA 和 EMA 确认方向的明显上升或下降趋势中最为有效。\n"
        interpretation_en += "- **High Volatility**: ALMA is responsive to price changes, making it suitable for volatile markets.\n"
        interpretation_cn += "- **高波动性**：ALMA 对价格变化反应灵敏，适用于波动较大的市场。\n"
        interpretation_en += "- **Avoid in Sideways/Noisy Markets**: ALMA may produce false signals in choppy or non-trending markets.\n"
        interpretation_cn += "- **避免在横盘/嘈杂市场**：在波动剧烈或无趋势的市场中，ALMA 可能产生虚假信号。\n"

        return interpretation_en, interpretation_cn

    interpret_en, interpret_cn = detailed_interpretation(
        buy_signals, sell_signals, confluences,
        current_price, trend, alma_window, alma_offset, alma_sigma
    )

    # Display Interpretations
    st.markdown("##### 📄 指标解读")

    # Tabs for English and Chinese
    tab1, tab2 = st.tabs(["🇨🇳 中文", "🇺🇸 English"])

    with tab1:
        st.markdown(interpret_cn)

    with tab2:
        st.markdown(interpret_en)

    # Optional: Display Data Table
    with st.expander("📊 查看原始数据 (View Raw Data)"):
        st.dataframe(df)

