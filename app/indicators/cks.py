from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import pandas as pd
from data.stock import get_stock_prices
import pytz

# ---------------------------
# Chande Kroll Stop (CKS) Analysis Function
# ---------------------------

def cks_analysis(ticker):
    st.markdown(f"# 📈 CKS止损 for {ticker.upper()}")

    # Sidebar for user inputs specific to CKS Analysis
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

    # User input function with additional CKS parameters
    def user_input_features():
        period = st.sidebar.selectbox(
            "📅 时间跨度 (Time Period)",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            help="选择分析的时间跨度。"
        )
        cks_window = st.sidebar.number_input(
            "🔢 CKS 窗口 (CKS Window)",
            min_value=1,
            max_value=100,
            value=14,
            help="CKS计算的窗口期，通常设为14。"
        )
        cks_multiplier = st.sidebar.number_input(
            "🔢 CKS 乘数 (CKS Multiplier)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="CKS计算的乘数，通常设为2.0。"
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
            start_date, end_date, cks_window, cks_multiplier,
            ema50_period, ema200_period, peaks_prominence
        )

    # Getting user input
    (
        start_date, end_date, cks_window, cks_multiplier,
        ema50_period, ema200_period, peaks_prominence
    ) = user_input_features()

    # Step 1: Fetch Historical Data using custom get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("❌ 未获取到数据。请检查股票代码并重试。")
        st.stop()

    # Step 2: Calculate Chande Kroll Stop (CKS)
    def calculate_cks(df, window=14, multiplier=2.0):
        """
        Calculate Chande Kroll Stop (CKS).
        """
        df = df.copy()
        df['ATR'] = df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min()
        df['ATR'] = df['ATR'].fillna(method='backfill')

        # Calculate CKS for long positions
        df['CKS_Long'] = df['close'] - (multiplier * df['ATR'])

        # Calculate CKS for short positions
        df['CKS_Short'] = df['close'] + (multiplier * df['ATR'])

        return df

    df = calculate_cks(df, window=cks_window, multiplier=cks_multiplier)

    # Step 3: Identify CKS Crossovers
    def identify_cks_signals(df, peaks_prominence=1.0):
        """
        Identify potential buy and sell signals based on CKS crossovers.
        """
        buy_signals = []
        sell_signals = []

        # Buy Signal: Price crosses above CKS_Long
        crossover_buy = df['close'] > df['CKS_Long']
        crossover_buy_shift = df['close'].shift(1) <= df['CKS_Long'].shift(1)
        buy_indices = df.index[crossover_buy & crossover_buy_shift]

        for idx in buy_indices:
            buy_signals.append({
                'Date': df['date'].iloc[idx],
                'Price': df['close'].iloc[idx],
                'CKS_Long': df['CKS_Long'].iloc[idx]
            })

        # Sell Signal: Price crosses below CKS_Short
        crossover_sell = df['close'] < df['CKS_Short']
        crossover_sell_shift = df['close'].shift(1) >= df['CKS_Short'].shift(1)
        sell_indices = df.index[crossover_sell & crossover_sell_shift]

        for idx in sell_indices:
            sell_signals.append({
                'Date': df['date'].iloc[idx],
                'Price': df['close'].iloc[idx],
                'CKS_Short': df['CKS_Short'].iloc[idx]
            })

        return buy_signals, sell_signals

    buy_signals, sell_signals = identify_cks_signals(df, peaks_prominence=peaks_prominence)

    # Step 4: Identify Confluence with Exponential Moving Averages (EMA)
    def find_confluence(df, ema50_period=50, ema200_period=200, cks_threshold=0):
        """
        Identify if CKS aligns with other moving averages.
        """
        # Calculate EMAs
        df['EMA50'] = df['close'].ewm(span=ema50_period, adjust=False).mean()
        df['EMA200'] = df['close'].ewm(span=ema200_period, adjust=False).mean()

        latest_cks_long = df['CKS_Long'].iloc[-1]
        latest_cks_short = df['CKS_Short'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]
        latest_price = df['close'].iloc[-1]

        confluence_levels = {}

        # Define confluence based on CKS crossovers and EMA alignment
        if latest_price > latest_ema50 and latest_price > latest_ema200:
            confluence_levels['Bullish Confluence'] = {
                'CKS_Long': latest_cks_long,
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }
        elif latest_price < latest_ema50 and latest_price < latest_ema200:
            confluence_levels['Bearish Confluence'] = {
                'CKS_Short': latest_cks_short,
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }

        return confluence_levels, df

    confluences, df = find_confluence(df, ema50_period=ema50_period, ema200_period=ema200_period)

    # Step 5: Determine Market Trend Based on CKS and EMAs
    def determine_trend(df, confluences, cks_threshold=0):
        """
        Determine the current market trend based on CKS and EMAs.
        """
        latest_price = df['close'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]
        latest_cks_long = df['CKS_Long'].iloc[-1]
        latest_cks_short = df['CKS_Short'].iloc[-1]

        if latest_price > latest_ema50 and latest_price > latest_ema200:
            trend = "上升趋势 (Uptrend)"
        elif latest_price < latest_ema50 and latest_price < latest_ema200:
            trend = "下降趋势 (Downtrend)"
        else:
            trend = "震荡区间 (Sideways)"

        return trend, latest_price

    trend, current_price = determine_trend(df, confluences)

    # Step 6: Plot Using Plotly
    def plot_cks(df, buy_signals, sell_signals, confluences, ticker,
                cks_window=14, cks_multiplier=2.0, ema50_period=50, ema200_period=200):
        """
        Plot the CKS along with price data and EMAs using Plotly.
        """
        fig = make_subplots(
            rows=1, cols=1,
            shared_xaxes=True,
            subplot_titles=(f'{ticker.upper()} 的股价和 Chande Kroll Stop (CKS)'),
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

        # CKS Lines
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['CKS_Long'],
                line=dict(color='green', width=1, dash='dash'),
                name='CKS Long'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['CKS_Short'],
                line=dict(color='red', width=1, dash='dash'),
                name='CKS Short'
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
            title=f'Chande Kroll Stop (CKS) 分析 for {ticker.upper()}',
            yaxis_title='Price',
            xaxis_title='',
            template='plotly_dark',
            showlegend=True,
            height=800
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    fig = plot_cks(
        df, buy_signals, sell_signals, confluences, ticker,
        cks_window=cks_window, cks_multiplier=cks_multiplier,
        ema50_period=ema50_period, ema200_period=ema200_period
    )
    st.plotly_chart(fig, use_container_width=True)

    # Step 7: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation(
        buy_signals, sell_signals, confluences,
        current_price, trend, cks_window, cks_multiplier
    ):
        """
        Provide a detailed, actionable interpretation based on CKS and divergences in both English and Chinese.
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
                        f"- **Bullish Confluence**: Price is above EMA{ema50_period} ({indicators['EMA50']:.2f}) and EMA{ema200_period} ({indicators['EMA200']:.2f}), indicating strong bullish momentum.\n"
                    )
                    interpretation_cn += (
                        f"- **看涨共振区**：价格高于 EMA{ema50_period} ({indicators['EMA50']:.2f}) 和 EMA{ema200_period} ({indicators['EMA200']:.2f})，表明强劲的看涨动能。\n"
                    )
                elif key == 'Bearish Confluence':
                    interpretation_en += (
                        f"- **Bearish Confluence**: Price is below EMA{ema50_period} ({indicators['EMA50']:.2f}) and EMA{ema200_period} ({indicators['EMA200']:.2f}), indicating strong bearish momentum.\n"
                    )
                    interpretation_cn += (
                        f"- **看跌共振区**：价格低于 EMA{ema50_period} ({indicators['EMA50']:.2f}) 和 EMA{ema200_period} ({indicators['EMA200']:.2f})，表明强劲的看跌动能。\n"
                    )
            interpretation_en += "\n"
            interpretation_cn += "\n"
        else:
            interpretation_en += "###### No Confluence Zones Detected.\n\n"
            interpretation_cn += "###### 未检测到共振区。\n\n"

        # 3. Price Position Analysis
        interpretation_en += "###### Price Position Relative to CKS and EMAs:\n"
        interpretation_cn += "###### 当前价格相对于 CKS 和 EMA 的位置：\n"
        if trend == "上升趋势 (Uptrend)":
            interpretation_en += f"- The current price is **above** EMA{ema50_period} and EMA{ema200_period}, and above CKS_Long, indicating a potential **buy** signal.\n"
            interpretation_cn += f"- 当前价格 **高于** EMA{ema50_period} 和 EMA{ema200_period}，且高于 CKS_Long，表明可能的 **买入** 信号。\n"
        elif trend == "下降趋势 (Downtrend)":
            interpretation_en += f"- The current price is **below** EMA{ema50_period} and EMA{ema200_period}, and below CKS_Short, indicating a potential **sell** signal.\n"
            interpretation_cn += f"- 当前价格 **低于** EMA{ema50_period} 和 EMA{ema200_period}，且低于 CKS_Short，表明可能的 **卖出** 信号。\n"
        else:
            interpretation_en += f"- The current price is **between** EMA{ema50_period} and EMA{ema200_period}, with no clear CKS signal, indicating a sideways or consolidating market.\n"
            interpretation_cn += f"- 当前价格 **位于** EMA{ema50_period} 和 EMA{ema200_period} 之间，且无明显的 CKS 信号，表明横盘或整合市场。\n"
        interpretation_en += "\n"
        interpretation_cn += "\n"

        # 4. Actionable Recommendations
        interpretation_en += "###### Actionable Recommendations:\n"
        interpretation_cn += "###### 可操作的建议：\n"

        # Buy Signals
        if buy_signals:
            interpretation_en += (
                f"- **Buying Opportunity**: {len(buy_signals)} buy signal(s) detected based on CKS_Long crossover. Consider buying when the price crosses above CKS_Long ({cks_multiplier} * ATR{cks_window}).\n"
            )
            interpretation_cn += (
                f"- **买入机会**：检测到 {len(buy_signals)} 个基于 CKS_Long 的买入信号。考虑在价格突破 CKS_Long ({cks_multiplier} * ATR{cks_window}) 时买入。\n"
            )

        # Sell Signals
        if sell_signals:
            interpretation_en += (
                f"- **Selling Opportunity**: {len(sell_signals)} sell signal(s) detected based on CKS_Short crossover. Consider selling when the price crosses below CKS_Short ({cks_multiplier} * ATR{cks_window}).\n"
            )
            interpretation_cn += (
                f"- **卖出机会**：检测到 {len(sell_signals)} 个基于 CKS_Short 的卖出信号。考虑在价格突破 CKS_Short ({cks_multiplier} * ATR{cks_window}) 时卖出。\n"
            )

        # Confluence Zones
        if confluences:
            interpretation_en += "\n- **Confluence Zones**: Trades near these areas have a higher probability of success due to the alignment of CKS with EMAs.\n"
            interpretation_cn += "\n- **共振区**：由于 CKS 与 EMA 对齐，接近这些区域的交易成功概率更高。\n"

        # Risk Management
        interpretation_en += "\n###### Risk Management:\n"
        interpretation_cn += "\n###### 风险管理：\n"
        interpretation_en += f"- **Stop-Loss**: Place stop-loss orders just beyond CKS_Long or CKS_Short to manage risk.\n"
        interpretation_cn += f"- **止损**：在 CKS_Long 或 CKS_Short 之外稍微放置止损订单以管理风险。\n"
        interpretation_en += f"- **Take-Profit**: Set target levels based on recent support/resistance levels or use a trailing stop to lock in profits.\n"
        interpretation_cn += f"- **止盈**：根据近期的支撑/阻力位设置目标水平或使用移动止盈以锁定利润。\n"

        # Market Conditions
        interpretation_en += "\n###### Optimal Market Conditions for Applying This Strategy:\n"
        interpretation_cn += "\n###### 应用此策略的最佳市场条件：\n"
        interpretation_en += "- **Trending Markets**: Most effective in clear uptrends or downtrends where CKS and EMAs confirm the direction.\n"
        interpretation_cn += "- **趋势市场**：在 CKS 和 EMA 确认方向的明显上升或下降趋势中最为有效。\n"
        interpretation_en += "- **High Volatility**: CKS leverages ATR, making it suitable for volatile markets.\n"
        interpretation_cn += "- **高波动性**：CKS 利用 ATR，使其适用于波动较大的市场。\n"
        interpretation_en += "- **Avoid in Sideways/Noisy Markets**: CKS may produce false signals in choppy or non-trending markets.\n"
        interpretation_cn += "- **避免在横盘/嘈杂市场**：在波动剧烈或无趋势的市场中，CKS 可能产生虚假信号。\n"

        return interpretation_en, interpretation_cn

    interpret_en, interpret_cn = detailed_interpretation(
        buy_signals, sell_signals, confluences,
        current_price, trend, cks_window, cks_multiplier
    )

    # Display Interpretations
    st.markdown("##### 📄 指标解读 (Indicator Interpretation)")

    # Tabs for English and Chinese
    tab1, tab2 = st.tabs(["🇨🇳 中文", "🇺🇸 English"])

    with tab1:
        st.markdown(interpret_cn)

    with tab2:
        st.markdown(interpret_en)

    # Optional: Display Data Table
    with st.expander("📊 查看原始数据 (View Raw Data)"):
        st.dataframe(df)
