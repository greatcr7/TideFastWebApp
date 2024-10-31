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
# Schaff Trend Cycle (STC) Analysis Function
# ---------------------------

def stc_analysis(ticker):
    st.markdown(f"# 📈 沙夫趋势周期 (STC) for {ticker.upper()}")

    # Sidebar for user inputs specific to STC Analysis
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

    # User input function with additional STC parameters
    def user_input_features():
        period = st.sidebar.selectbox(
            "📅 时间跨度 (Time Period)",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            help="选择分析的时间跨度。"
        )
        stc_fast_length = st.sidebar.number_input(
            "🔢 STC 快速周期 (STC Fast Length)",
            min_value=1,
            max_value=100,
            value=23,
            help="STC快速周期，通常设为23。"
        )
        stc_slow_length = st.sidebar.number_input(
            "🔢 STC 慢速周期 (STC Slow Length)",
            min_value=1,
            max_value=100,
            value=50,
            help="STC慢速周期，通常设为50。"
        )
        stc_cycle_length = st.sidebar.number_input(
            "🔢 STC 循环周期 (STC Cycle Length)",
            min_value=1,
            max_value=100,
            value=10,
            help="STC循环周期，通常设为10。"
        )
        stc_overbought = st.sidebar.number_input(
            "📈 STC 超买水平 (STC Overbought Level)",
            min_value=50,
            max_value=100,
            value=75,
            help="STC指标的超买水平，通常设为75。"
        )
        stc_oversold = st.sidebar.number_input(
            "📉 STC 超卖水平 (STC Oversold Level)",
            min_value=0,
            max_value=50,
            value=25,
            help="STC指标的超卖水平，通常设为25。"
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
            start_date, end_date, stc_fast_length, stc_slow_length,
            stc_cycle_length, stc_overbought, stc_oversold,
            ema50_period, ema200_period,
            peaks_prominence
        )

    # Getting user input
    (
        start_date, end_date, stc_fast_length, stc_slow_length,
        stc_cycle_length, stc_overbought, stc_oversold,
        ema50_period, ema200_period,
        peaks_prominence
    ) = user_input_features()

    # Step 1: Fetch Historical Data using custom get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("❌ 未获取到数据。请检查股票代码并重试。")
        st.stop()

    # Step 2: Calculate Schaff Trend Cycle (STC)
    def calculate_stc(df, fast_length=23, slow_length=50, cycle_length=10):
        """
        Calculate Schaff Trend Cycle (STC).
        """
        df = df.copy()
        
        # Calculate MACD
        df['MACD'] = df['close'].ewm(span=fast_length, adjust=False).mean() - df['close'].ewm(span=slow_length, adjust=False).mean()
        df['Signal_Line'] = df['MACD'].ewm(span=cycle_length, adjust=False).mean()
        
        # Calculate MACD Histogram
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        # Normalize the MACD Histogram
        df['MACD_Hist_Scaled'] = (df['MACD_Hist'] - df['MACD_Hist'].min()) / (df['MACD_Hist'].max() - df['MACD_Hist'].min())
        df['MACD_Hist_Scaled'] = df['MACD_Hist_Scaled'] * 100
        
        # Calculate STC using fast and slow cycle lengths
        df['STC'] = df['MACD_Hist_Scaled'].rolling(window=cycle_length).mean()
        
        return df

    df = calculate_stc(df, fast_length=stc_fast_length, slow_length=stc_slow_length, cycle_length=stc_cycle_length)

    # Step 3: Identify STC Signals
    def identify_stc_signals(df, stc_overbought=75, stc_oversold=25, peaks_prominence=1.0):
        """
        Identify potential buy and sell signals based on STC crossovers.
        """
        buy_signals = []
        sell_signals = []

        # Buy Signal: STC crosses above oversold level
        crossover_buy = df['STC'] > stc_oversold
        crossover_buy_shift = df['STC'].shift(1) <= stc_oversold
        buy_indices = df.index[crossover_buy & crossover_buy_shift]

        for idx in buy_indices:
            buy_signals.append({
                'Date': df['date'].iloc[idx],
                'Price': df['close'].iloc[idx],
                'STC': df['STC'].iloc[idx]
            })

        # Sell Signal: STC crosses below overbought level
        crossover_sell = df['STC'] < stc_overbought
        crossover_sell_shift = df['STC'].shift(1) >= stc_overbought
        sell_indices = df.index[crossover_sell & crossover_sell_shift]

        for idx in sell_indices:
            sell_signals.append({
                'Date': df['date'].iloc[idx],
                'Price': df['close'].iloc[idx],
                'STC': df['STC'].iloc[idx]
            })

        return buy_signals, sell_signals

    buy_signals, sell_signals = identify_stc_signals(df, stc_overbought=stc_overbought, stc_oversold=stc_oversold, peaks_prominence=peaks_prominence)

    # Step 4: Identify Confluence with Exponential Moving Averages (EMA)
    def find_confluence(df, ema50_period=50, ema200_period=200):
        """
        Identify if STC aligns with other moving averages.
        """
        # Calculate EMAs
        df['EMA50'] = df['close'].ewm(span=ema50_period, adjust=False).mean()
        df['EMA200'] = df['close'].ewm(span=ema200_period, adjust=False).mean()

        latest_stc = df['STC'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]
        latest_price = df['close'].iloc[-1]

        confluence_levels = {}

        # Define confluence based on STC crossing over oversold or overbought and EMA alignment
        if latest_stc > stc_overbought and latest_price > latest_ema50 and latest_price > latest_ema200:
            confluence_levels['Bullish Confluence'] = {
                'STC': latest_stc,
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }
        elif latest_stc < stc_oversold and latest_price < latest_ema50 and latest_price < latest_ema200:
            confluence_levels['Bearish Confluence'] = {
                'STC': latest_stc,
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }

        return confluence_levels, df

    confluences, df = find_confluence(df, ema50_period=ema50_period, ema200_period=ema200_period)

    # Step 5: Determine Market Trend Based on STC and EMAs
    def determine_trend(df, confluences, stc_overbought=75, stc_oversold=25):
        """
        Determine the current market trend based on STC and EMAs.
        """
        latest_stc = df['STC'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]
        latest_price = df['close'].iloc[-1]

        if latest_stc > stc_overbought and latest_price > latest_ema50 and latest_price > latest_ema200:
            trend = "上升趋势 (Uptrend)"
        elif latest_stc < stc_oversold and latest_price < latest_ema50 and latest_price < latest_ema200:
            trend = "下降趋势 (Downtrend)"
        else:
            trend = "震荡区间 (Sideways)"

        return trend, latest_price

    trend, current_price = determine_trend(df, confluences, stc_overbought=stc_overbought, stc_oversold=stc_oversold)

    # Step 6: Plot Using Plotly
    def plot_stc(df, buy_signals, sell_signals, confluences, ticker,
                stc_fast_length=23, stc_slow_length=50, stc_cycle_length=10,
                stc_overbought=75, stc_oversold=25,
                ema50_period=50, ema200_period=200):
        """
        Plot the STC along with price data and EMAs using Plotly.
        """
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{ticker.upper()} 的股价和移动平均线 (Price and EMAs)', 'Schaff Trend Cycle (STC)'),
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

        # STC
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['STC'],
                line=dict(color='orange', width=2),
                name='STC'
            ),
            row=2, col=1
        )

        # Overbought and Oversold lines
        fig.add_hline(
            y=stc_overbought, line=dict(color='red', dash='dash'),
            row=2, col=1
        )
        fig.add_hline(
            y=stc_oversold, line=dict(color='green', dash='dash'),
            row=2, col=1
        )
        fig.add_hline(
            y=50, line=dict(color='gray', dash='dash'),
            row=2, col=1
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
            fig.add_annotation(
                x=signal['Date'], y=signal['STC'],
                text="Buy",
                showarrow=True,
                arrowhead=1,
                ax=0, ay=-40,
                arrowcolor='green',
                row=2, col=1
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
            fig.add_annotation(
                x=signal['Date'], y=signal['STC'],
                text="Sell",
                showarrow=True,
                arrowhead=1,
                ax=0, ay=40,
                arrowcolor='red',
                row=2, col=1
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
            fig.add_vline(
                x=value['Date'] if 'Date' in value else df['date'].iloc[-1],
                line=dict(color=color, dash='dot'),
                row=2, col=1
            )

        fig.update_layout(
            title=f'Schaff Trend Cycle (STC) 分析 for {ticker.upper()}',
            yaxis_title='Price',
            xaxis_title='',
            template='plotly_dark',
            showlegend=True,
            height=800
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    fig = plot_stc(
        df, buy_signals, sell_signals, confluences, ticker,
        stc_fast_length=stc_fast_length, stc_slow_length=stc_slow_length, stc_cycle_length=stc_cycle_length,
        stc_overbought=stc_overbought, stc_oversold=stc_oversold,
        ema50_period=ema50_period, ema200_period=ema200_period
    )
    st.plotly_chart(fig, use_container_width=True)

    # Step 7: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation(
        buy_signals, sell_signals, confluences,
        current_price, trend, stc_fast_length, stc_slow_length, stc_cycle_length
    ):
        """
        Provide a detailed, actionable interpretation based on STC and crossovers in both English and Chinese.
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
                        f"- **Bullish Confluence**: STC is above {stc_overbought} ({indicators['STC']:.2f}), "
                        f"and the price is above both EMA{ema50_period} ({indicators['EMA50']:.2f}) and EMA{ema200_period} ({indicators['EMA200']:.2f}).\n"
                    )
                    interpretation_cn += (
                        f"- **看涨共振区**：STC 高于 {stc_overbought} ({indicators['STC']:.2f})，"
                        f"价格高于 EMA{ema50_period} ({indicators['EMA50']:.2f}) 和 EMA{ema200_period} ({indicators['EMA200']:.2f})。\n"
                    )
                elif key == 'Bearish Confluence':
                    interpretation_en += (
                        f"- **Bearish Confluence**: STC is below {stc_oversold} ({indicators['STC']:.2f}), "
                        f"and the price is below both EMA{ema50_period} ({indicators['EMA50']:.2f}) and EMA{ema200_period} ({indicators['EMA200']:.2f}).\n"
                    )
                    interpretation_cn += (
                        f"- **看跌共振区**：STC 低于 {stc_oversold} ({indicators['STC']:.2f})，"
                        f"价格低于 EMA{ema50_period} ({indicators['EMA50']:.2f}) 和 EMA{ema200_period} ({indicators['EMA200']:.2f})。\n"
                    )
            interpretation_en += "\n"
            interpretation_cn += "\n"
        else:
            interpretation_en += "###### No Confluence Zones Detected.\n\n"
            interpretation_cn += "###### 未检测到共振区。\n\n"

        # 3. Price Position Analysis
        interpretation_en += "###### Price Position Relative to STC and EMAs:\n"
        interpretation_cn += "###### 当前价格相对于 STC 和 EMA 的位置：\n"
        if trend == "上升趋势 (Uptrend)":
            interpretation_en += f"- The current price is **above** EMA{ema50_period} and EMA{ema200_period}, and STC is above {stc_overbought}, indicating a potential **buy** signal.\n"
            interpretation_cn += f"- 当前价格 **高于** EMA{ema50_period} 和 EMA{ema200_period}，且 STC 高于 {stc_overbought}，表明可能的 **买入** 信号。\n"
        elif trend == "下降趋势 (Downtrend)":
            interpretation_en += f"- The current price is **below** EMA{ema50_period} and EMA{ema200_period}, and STC is below {stc_oversold}, indicating a potential **sell** signal.\n"
            interpretation_cn += f"- 当前价格 **低于** EMA{ema50_period} 和 EMA{ema200_period}，且 STC 低于 {stc_oversold}，表明可能的 **卖出** 信号。\n"
        else:
            interpretation_en += f"- The current price is **between** EMA{ema50_period} and EMA{ema200_period}, with STC around 50, indicating a sideways or consolidating market.\n"
            interpretation_cn += f"- 当前价格 **位于** EMA{ema50_period} 和 EMA{ema200_period} 之间，且 STC 约为50，表明横盘或整合市场。\n"
        interpretation_en += "\n"
        interpretation_cn += "\n"

        # 4. Actionable Recommendations
        interpretation_en += "###### Actionable Recommendations:\n"
        interpretation_cn += "###### 可操作的建议：\n"

        # Buy Signals
        if buy_signals:
            interpretation_en += (
                f"- **Buying Opportunity**: {len(buy_signals)} buy signal(s) detected based on STC crossing above {stc_oversold}. "
                f"Consider buying when the price crosses above STC (Fast Length: {stc_fast_length}, Slow Length: {stc_slow_length}, Cycle Length: {stc_cycle_length}).\n"
            )
            interpretation_cn += (
                f"- **买入机会**：检测到 {len(buy_signals)} 个基于 STC 超卖水平突破的买入信号。"
                f"考虑在价格突破 STC（快速周期：{stc_fast_length}，慢速周期：{stc_slow_length}，循环周期：{stc_cycle_length}）时买入。\n"
            )

        # Sell Signals
        if sell_signals:
            interpretation_en += (
                f"- **Selling Opportunity**: {len(sell_signals)} sell signal(s) detected based on STC crossing below {stc_overbought}. "
                f"Consider selling when the price crosses below STC (Fast Length: {stc_fast_length}, Slow Length: {stc_slow_length}, Cycle Length: {stc_cycle_length}).\n"
            )
            interpretation_cn += (
                f"- **卖出机会**：检测到 {len(sell_signals)} 个基于 STC 超买水平突破的卖出信号。"
                f"考虑在价格突破 STC（快速周期：{stc_fast_length}，慢速周期：{stc_slow_length}，循环周期：{stc_cycle_length}）时卖出。\n"
            )

        # Confluence Zones
        if confluences:
            interpretation_en += "\n- **Confluence Zones**: Trades near these areas have a higher probability of success due to the alignment of STC with EMAs.\n"
            interpretation_cn += "\n- **共振区**：由于 STC 与 EMA 对齐，接近这些区域的交易成功概率更高。\n"

        # Risk Management
        interpretation_en += "\n###### Risk Management:\n"
        interpretation_cn += "\n###### 风险管理：\n"
        interpretation_en += f"- **Stop-Loss**: Place stop-loss orders just beyond STC to manage risk.\n"
        interpretation_cn += f"- **止损**：在 STC 之外稍微放置止损订单以管理风险。\n"
        interpretation_en += f"- **Take-Profit**: Set target levels based on recent support/resistance levels or use a trailing stop to lock in profits.\n"
        interpretation_cn += f"- **止盈**：根据近期的支撑/阻力位设置目标水平或使用移动止盈以锁定利润。\n"

        # Market Conditions
        interpretation_en += "\n###### Optimal Market Conditions for Applying This Strategy:\n"
        interpretation_cn += "\n###### 应用此策略的最佳市场条件：\n"
        interpretation_en += "- **Trending Markets**: Most effective in clear uptrends or downtrends where STC and EMAs confirm the direction.\n"
        interpretation_cn += "- **趋势市场**：在 STC 和 EMA 确认方向的明显上升或下降趋势中最为有效。\n"
        interpretation_en += "- **High Volatility**: STC is responsive to price changes, making it suitable for volatile markets.\n"
        interpretation_cn += "- **高波动性**：STC 对价格变化反应灵敏，适用于波动较大的市场。\n"
        interpretation_en += "- **Avoid in Sideways/Noisy Markets**: STC may produce false signals in choppy or non-trending markets.\n"
        interpretation_cn += "- **避免在横盘/嘈杂市场**：在波动剧烈或无趋势的市场中，STC 可能产生虚假信号。\n"

        return interpretation_en, interpretation_cn

    interpret_en, interpret_cn = detailed_interpretation(
        buy_signals, sell_signals, confluences,
        current_price, trend, stc_fast_length, stc_slow_length, stc_cycle_length
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

