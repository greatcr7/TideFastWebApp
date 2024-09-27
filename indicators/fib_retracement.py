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
# Fibonacci Retracement Analysis Function
# ---------------------------

def fibonacci_retracement_analysis(ticker):
    st.markdown(f"# 📈 斐波那契回调线 for {ticker.upper()}")

    # Sidebar for user inputs specific to Fibonacci Retracement Analysis
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

    # User input function with additional Fibonacci Retracement parameters
    def user_input_features():
        period = st.sidebar.selectbox(
            "📅 时间跨度 (Time Period)",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            help="选择分析的时间跨度。"
        )
        num_peaks = st.sidebar.number_input(
            "🔢 峰值数量 (Number of Peaks)",
            min_value=1,
            max_value=10,
            value=1,
            help="选择用于绘制斐波那契回撤的峰值数量。"
        )
        peaks_prominence = st.sidebar.number_input(
            "🔝 峰值显著性 (Peak Prominence)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="峰值检测时的显著性要求，通常设为1.0。"
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

        # Convert period to start and end dates
        start_date, end_date = convert_period_to_dates(period)

        return (
            start_date, end_date, num_peaks, peaks_prominence,
            ema50_period, ema200_period
        )

    # Getting user input
    (
        start_date, end_date, num_peaks, peaks_prominence,
        ema50_period, ema200_period
    ) = user_input_features()

    # Step 1: Fetch Historical Data using custom get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("❌ 未获取到数据。请检查股票代码并重试。")
        st.stop()

    # Step 2: Identify Swing Highs and Lows
    def identify_swings(df, num_peaks=1, prominence=1.0):
        """
        Identify significant swing highs and lows using peak detection.
        """
        # Identify swing highs
        peaks, _ = find_peaks(df['high'], prominence=prominence)
        # Identify swing lows
        troughs, _ = find_peaks(-df['low'], prominence=prominence)

        # Combine peaks and troughs and sort by date
        swing_points = sorted(
            list(peaks) + list(troughs),
            key=lambda x: df['date'].iloc[x]
        )

        # Select the most recent 'num_peaks' swing highs and lows
        recent_peaks = sorted(peaks, key=lambda x: df['high'].iloc[x], reverse=True)[:num_peaks]
        recent_troughs = sorted(troughs, key=lambda x: df['low'].iloc[x])[:num_peaks]

        return recent_peaks, recent_troughs

    recent_peaks, recent_troughs = identify_swings(df, num_peaks=num_peaks, prominence=peaks_prominence)

    if not recent_peaks or not recent_troughs:
        st.error("❌ 未能检测到足够的峰值或谷值。请调整峰值数量或显著性。")
        st.stop()

    # For simplicity, take the highest peak and the lowest trough
    swing_high = recent_peaks[0]
    swing_low = recent_troughs[0]

    high_price = df['high'].iloc[swing_high]
    low_price = df['low'].iloc[swing_low]

    # Ensure swing_high occurs after swing_low for a valid retracement
    if swing_high < swing_low:
        swing_high = recent_peaks[1] if len(recent_peaks) > 1 else swing_high
        high_price = df['high'].iloc[swing_high]

    # Step 3: Calculate Fibonacci Retracement Levels
    def calculate_fibonacci_levels(high, low):
        """
        Calculate Fibonacci retracement levels based on high and low prices.
        """
        diff = high - low
        levels = {
            '0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '76.4%': high - 0.764 * diff,
            '100%': low
        }
        return levels

    fib_levels = calculate_fibonacci_levels(high_price, low_price)

    # Step 4: Identify Confluence with Exponential Moving Averages (EMA)
    def find_confluence(df, ema50_period=50, ema200_period=200):
        """
        Identify if price aligns with EMAs.
        """
        # Calculate EMAs
        df['EMA50'] = df['close'].ewm(span=ema50_period, adjust=False).mean()
        df['EMA200'] = df['close'].ewm(span=ema200_period, adjust=False).mean()

        latest_price = df['close'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]

        confluence_levels = {}

        # Define confluence based on EMA alignment
        if latest_price > latest_ema50 and latest_price > latest_ema200:
            confluence_levels['Bullish Confluence'] = {
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }
        elif latest_price < latest_ema50 and latest_price < latest_ema200:
            confluence_levels['Bearish Confluence'] = {
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }

        return confluence_levels, df

    confluences, df = find_confluence(df, ema50_period=ema50_period, ema200_period=ema200_period)

    # Step 5: Determine Market Trend Based on Price and EMAs
    def determine_trend(df, confluences):
        """
        Determine the current market trend based on price and EMAs.
        """
        latest_price = df['close'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]

        if latest_price > latest_ema50 and latest_price > latest_ema200:
            trend = "上升趋势 (Uptrend)"
        elif latest_price < latest_ema50 and latest_price < latest_ema200:
            trend = "下降趋势 (Downtrend)"
        else:
            trend = "震荡区间 (Sideways)"

        return trend, latest_price

    trend, current_price = determine_trend(df, confluences)

    # Step 6: Plot Using Plotly
    def plot_fibonacci_retracement(df, fib_levels, swing_high_idx, swing_low_idx, confluences, ticker,
                                   ema50_period=50, ema200_period=200):
        """
        Plot the price along with Fibonacci retracement levels and EMAs using Plotly.
        """
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                            subplot_titles=(f'{ticker.upper()} 的股价和斐波那契回撤 (Price and Fibonacci Retracement)'),
                            row_width=[0.2])

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

        # Fibonacci Retracement Levels
        for level, price in fib_levels.items():
            fig.add_hline(
                y=price, line=dict(color='grey', dash='dash'),
                annotation_text=level, annotation_position="top left",
                annotation=dict(font_size=10),
                row=1, col=1
            )

        # Highlight Confluence Zones
        for key, value in confluences.items():
            if key == 'Bullish Confluence':
                color = 'green'
                annotation_text = "Bullish Confluence"
            elif key == 'Bearish Confluence':
                color = 'red'
                annotation_text = "Bearish Confluence"
            else:
                color = 'yellow'
                annotation_text = "Confluence"

            fig.add_annotation(
                x=df['date'].iloc[-1],
                y=df['close'].iloc[-1],
                text=annotation_text,
                showarrow=True,
                arrowhead=1,
                ax=0, ay=-40 if key == 'Bullish Confluence' else 40,
                arrowcolor=color,
                row=1, col=1
            )

        fig.update_layout(
            title=f'Fibonacci Retracement 分析 for {ticker.upper()}',
            yaxis_title='Price',
            xaxis_title='',
            template='plotly_dark',
            showlegend=True,
            height=800
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    fig = plot_fibonacci_retracement(
        df, fib_levels, swing_high, swing_low, confluences, ticker,
        ema50_period=ema50_period, ema200_period=ema200_period
    )
    st.plotly_chart(fig, use_container_width=True)

    # Step 7: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation(
        fib_levels, confluences, current_price, trend,
        swing_high, swing_low
    ):
        """
        Provide a detailed, actionable interpretation based on Fibonacci Retracement in both English and Chinese.
        """
        interpretation_en = ""
        interpretation_cn = ""

        # 1. Trend Analysis
        interpretation_en += f"###### Current Market Trend: {trend}\n\n"
        interpretation_en += f"**Current Price**: {current_price:.2f}\n\n"

        interpretation_cn += f"###### 当前市场趋势：{trend}\n\n"
        interpretation_cn += f"**当前价格**：{current_price:.2f}\n\n"

        # 2. Fibonacci Retracement Levels
        interpretation_en += "###### Fibonacci Retracement Levels:\n"
        interpretation_cn += "###### 斐波那契回撤水平：\n"
        for level, price in fib_levels.items():
            interpretation_en += f"- **{level}**: {price:.2f}\n"
            interpretation_cn += f"- **{level}**：{price:.2f}\n"
        interpretation_en += "\n"
        interpretation_cn += "\n"

        # 3. Confluence Analysis
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

        # 4. Price Position Analysis
        interpretation_en += "###### Price Position Relative to Fibonacci Levels and EMAs:\n"
        interpretation_cn += "###### 当前价格相对于斐波那契水平和 EMA 的位置：\n"
        if trend == "上升趋势 (Uptrend)":
            interpretation_en += (
                f"- The current price is **above** EMA{ema50_period} and EMA{ema200_period}, and trading near the {fib_levels['38.2%']} level, indicating potential support.\n"
            )
            interpretation_cn += (
                f"- 当前价格 **高于** EMA{ema50_period} 和 EMA{ema200_period}，并接近 {fib_levels['38.2%']} 水平，表明潜在支撑。\n"
            )
        elif trend == "下降趋势 (Downtrend)":
            interpretation_en += (
                f"- The current price is **below** EMA{ema50_period} and EMA{ema200_period}, and trading near the {fib_levels['61.8%']} level, indicating potential resistance.\n"
            )
            interpretation_cn += (
                f"- 当前价格 **低于** EMA{ema50_period} 和 EMA{ema200_period}，并接近 {fib_levels['61.8%']} 水平，表明潜在阻力。\n"
            )
        else:
            interpretation_en += (
                f"- The current price is **between** EMA{ema50_period} and EMA{ema200_period}, and trading near the {fib_levels['50%']} level, indicating a consolidation phase.\n"
            )
            interpretation_cn += (
                f"- 当前价格 **位于** EMA{ema50_period} 和 EMA{ema200_period} 之间，并接近 {fib_levels['50%']} 水平，表明盘整阶段。\n"
            )
        interpretation_en += "\n"
        interpretation_cn += "\n"

        # 5. Actionable Recommendations
        interpretation_en += "###### Actionable Recommendations:\n"
        interpretation_cn += "###### 可操作的建议：\n"

        # Support and Resistance
        interpretation_en += (
            f"- **Support Level ({fib_levels['38.2%']}):** Consider buying if the price bounces off this level, especially if it aligns with **Bullish Confluence**.\n"
        )
        interpretation_cn += (
            f"- **支撑位 ({fib_levels['38.2%']}):** 如果价格在此水平反弹，特别是当它与 **看涨共振区** 对齐时，考虑买入。\n"
        )

        interpretation_en += (
            f"- **Resistance Level ({fib_levels['61.8%']}):** Consider selling if the price fails to break above this level, especially if it aligns with **Bearish Confluence**.\n"
        )
        interpretation_cn += (
            f"- **阻力位 ({fib_levels['61.8%']}):** 如果价格未能突破此水平，特别是当它与 **看跌共振区** 对齐时，考虑卖出。\n"
        )

        # Confluence Zones
        if confluences:
            interpretation_en += (
                f"- **Confluence Zones**: Trades near these areas have a higher probability of success due to the alignment of Fibonacci levels with EMAs.\n"
            )
            interpretation_cn += (
                f"- **共振区**：由于斐波那契水平与 EMA 对齐，接近这些区域的交易成功概率更高。\n"
            )

        # Breakout Scenarios
        interpretation_en += "\n###### Breakout Scenarios:\n"
        interpretation_cn += "\n###### 突破情景：\n"
        interpretation_en += (
            f"- **Bullish Breakout**: If the price breaks above the {fib_levels['23.6%']} level with increasing volume, consider **entering a long position**.\n"
        )
        interpretation_cn += (
            f"- **看涨突破**：如果价格在成交量增加的情况下突破 {fib_levels['23.6%']} 水平，考虑 **建立多头仓位**。\n"
        )
        interpretation_en += (
            f"- **Bearish Breakout**: If the price breaks below the {fib_levels['76.4%']} level with decreasing volume, consider **entering a short position**.\n"
        )
        interpretation_cn += (
            f"- **看跌突破**：如果价格在成交量减少的情况下突破 {fib_levels['76.4%']} 水平，考虑 **建立空头仓位**。\n"
        )

        # Risk Management
        interpretation_en += "\n###### Risk Management:\n"
        interpretation_cn += "\n###### 风险管理：\n"
        interpretation_en += f"- **Stop-Loss**: Place stop-loss orders just beyond the nearest Fibonacci level ({fib_levels['23.6%']} or {fib_levels['76.4%']}) to manage risk.\n"
        interpretation_cn += f"- **止损**：在最近的斐波那契水平（{fib_levels['23.6%']} 或 {fib_levels['76.4%']}）之外稍微放置止损订单以管理风险。\n"
        interpretation_en += f"- **Take-Profit**: Set target levels based on Fibonacci levels or recent support/resistance levels.\n"
        interpretation_cn += f"- **止盈**：根据斐波那契水平或近期的支撑/阻力位设置目标水平。\n"

        # Market Conditions
        interpretation_en += "\n###### Optimal Market Conditions for Applying This Strategy:\n"
        interpretation_cn += "\n###### 应用此策略的最佳市场条件：\n"
        interpretation_en += "- **Trending Markets**: Most effective in clear uptrends or downtrends where Fibonacci levels can act as support/resistance.\n"
        interpretation_cn += "- **趋势市场**：在斐波那契水平可以作为支撑/阻力的明显上升或下降趋势中最为有效。\n"
        interpretation_en += "- **High Volatility**: Fibonacci retracement levels are more reliable in volatile markets where significant price movements occur.\n"
        interpretation_cn += "- **高波动性**：在波动较大的市场中，斐波那契回撤水平更为可靠，因为会发生显著的价格波动。\n"
        interpretation_en += "- **Avoid in Sideways/Noisy Markets**: Fibonacci levels may produce false signals in choppy or non-trending markets.\n"
        interpretation_cn += "- **避免在横盘/嘈杂市场**：在波动剧烈或无趋势的市场中，斐波那契水平可能产生虚假信号。\n"

        return interpretation_en, interpretation_cn

    interpret_en, interpret_cn = detailed_interpretation(
        fib_levels, confluences, current_price, trend,
        swing_high, swing_low
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

