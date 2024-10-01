from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import re
from data.stock import get_stock_prices
import pytz
import numpy as np

# ---------------------------
# Hurst Analysis Function
# ---------------------------

def hurst_analysis(ticker):
    st.markdown(f"# 📈 Hurst指数 for {ticker.upper()}")

    # Sidebar for user inputs specific to Hurst Analysis
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

    # User input function with additional Hurst parameters
    def user_input_features():
        period = st.sidebar.selectbox(
            "📅 时间跨度 (Time Period)",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            help="选择分析的时间跨度。"
        )
        hurst_window = st.sidebar.number_input(
            "🔢 Hurst 窗口 (Hurst Window)",
            min_value=10,
            max_value=500,
            value=100,
            help="用于计算 Hurst 指数的滚动窗口期，通常设为100。"
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
            start_date, end_date, hurst_window,
            ema50_period, ema200_period, peaks_prominence
        )

    # Getting user input
    (
        start_date, end_date, hurst_window,
        ema50_period, ema200_period, peaks_prominence
    ) = user_input_features()

    # Step 1: Fetch Historical Data using custom get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("❌ 未获取到数据。请检查股票代码并重试。")
        st.stop()

    # Ensure the data is sorted by date
    df = df.sort_values('date').reset_index(drop=True)

    # Step 2: Calculate Hurst Exponent
    def calculate_hurst(ts, window=100):
        """
        Calculate the Hurst exponent for a time series using a rolling window approach.
        """
        hurst_exponents = []
        for i in range(len(ts)):
            if i + 1 < window:
                hurst_exponents.append(np.nan)
                continue
            window_ts = ts[i + 1 - window:i + 1]
            # Calculate the range of the series
            mean_ts = np.mean(window_ts)
            rescaled_range = np.max(window_ts - mean_ts) - np.min(window_ts - mean_ts)
            # Calculate the standard deviation
            std_ts = np.std(window_ts)
            if std_ts == 0:
                hurst = np.nan
            else:
                hurst = np.log(rescaled_range / std_ts) / np.log(window)
            hurst_exponents.append(hurst)
        return hurst_exponents

    df['Hurst'] = calculate_hurst(df['close'], window=hurst_window)

    # Step 3: Identify Trends Based on Hurst Exponent and Label as Momentum or Reversal
    def identify_trends(df, hurst_col='Hurst'):
        """
        Identify periods of trending (momentum), mean-reverting (reversal), or random walk based on Hurst exponent.
        """
        trend = []
        momentum_reversal = []
        for h in df[hurst_col]:
            if np.isnan(h):
                trend.append("N/A")
                momentum_reversal.append("N/A")
            elif h < 0.5:
                trend.append("Mean-Reverting")
                momentum_reversal.append("Reversal")
            elif h == 0.5:
                trend.append("Random Walk")
                momentum_reversal.append("Neutral")
            else:
                trend.append("Trending")
                momentum_reversal.append("Momentum")
        df['Trend'] = trend
        df['Momentum_Reversal'] = momentum_reversal
        return df

    df = identify_trends(df)

    # Step 4: Identify Price Peaks and Troughs for Confluence (Optional)
    def identify_peaks_troughs(df, window=5, prominence=1.0, price_col='close'):
        """
        Identify peaks and troughs in the price data.
        """
        peaks, _ = find_peaks(df[price_col], distance=window, prominence=prominence)
        troughs, _ = find_peaks(-df[price_col], distance=window, prominence=prominence)
        return peaks, troughs

    price_peaks, price_troughs = identify_peaks_troughs(df, window=5, prominence=peaks_prominence)

    # Step 5: Identify Confluence with Exponential Moving Averages (EMA)
    def find_confluence(df, ema50_period=50, ema200_period=200, hurst_threshold=0.5):
        """
        Identify if Hurst exponent aligns with other moving averages.
        """
        # Calculate EMAs
        df['EMA50'] = df['close'].ewm(span=ema50_period, adjust=False).mean()
        df['EMA200'] = df['close'].ewm(span=ema200_period, adjust=False).mean()

        latest_hurst = df['Hurst'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]
        latest_price = df['close'].iloc[-1]

        confluence_levels = {}

        # Define confluence based on Hurst thresholds and EMA alignment
        if latest_hurst > hurst_threshold and latest_price > latest_ema50 and latest_price > latest_ema200:
            confluence_levels['Trending Confluence'] = {
                'Hurst': latest_hurst,
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }
        elif latest_hurst < hurst_threshold and latest_price < latest_ema50 and latest_price < latest_ema200:
            confluence_levels['Mean-Reverting Confluence'] = {
                'Hurst': latest_hurst,
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }

        return confluence_levels, df

    confluences, df = find_confluence(df, ema50_period=ema50_period, ema200_period=ema200_period, hurst_threshold=0.5)

    # Step 6: Determine Market Trend Based on Hurst and EMAs
    def determine_trend(df, confluences, hurst_threshold=0.5):
        """
        Determine the current market trend based on Hurst exponent and EMAs.
        """
        latest_hurst = df['Hurst'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]
        latest_price = df['close'].iloc[-1]
        latest_momentum_reversal = df['Momentum_Reversal'].iloc[-1]

        if latest_hurst > hurst_threshold and latest_price > latest_ema50 and latest_price > latest_ema200:
            trend = "Trending Upwards (上升趋势)"
            momentum_reversal = "Momentum (动量)"
        elif latest_hurst < hurst_threshold and latest_price < latest_ema50 and latest_price < latest_ema200:
            trend = "Mean-Reverting Downwards (均值回归下降趋势)"
            momentum_reversal = "Reversal (反转)"
        else:
            trend = "Sideways or Mixed (震荡或混合趋势)"
            momentum_reversal = "Neutral (中性)"

        return trend, momentum_reversal, latest_price

    trend, momentum_reversal, current_price = determine_trend(df, confluences)

    # Step 7: Plot Using Plotly
    def plot_hurst(df, confluences, ticker, hurst_window=100, ema50_period=50, ema200_period=200):
        """
        Plot the stock price alongside the Hurst exponent using Plotly.
        """
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{ticker.upper()} 的股价和价格均线 (Price and EMAs)', '赫斯特指数 (Hurst Exponent)'),
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

        # Hurst Exponent
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['Hurst'],
                line=dict(color='orange', width=1),
                name='Hurst Exponent'
            ),
            row=2, col=1
        )

        # Hurst Threshold Line
        fig.add_hline(
            y=0.5, line=dict(color='gray', dash='dash'),
            row=2, col=1
        )

        # Highlight Confluence Zones
        for key, value in confluences.items():
            if key == 'Trending Confluence':
                color = 'green'
                y_position = value['Hurst']
                annotation_text = "Trending Confluence"
            elif key == 'Mean-Reverting Confluence':
                color = 'red'
                y_position = value['Hurst']
                annotation_text = "Mean-Reverting Confluence"
            else:
                color = 'yellow'
                y_position = 0.5
                annotation_text = "Confluence"

            fig.add_hline(
                y=y_position, line=dict(color=color, dash='dot'),
                row=2, col=1
            )
            # Optionally, add annotations
            fig.add_annotation(
                x=df['date'].iloc[-1],
                y=y_position,
                text=annotation_text,
                showarrow=False,
                yanchor="bottom" if color == 'green' else "top",
                font=dict(color=color),
                row=2, col=1
            )

        fig.update_layout(
            title=f'Hurst Exponent Analysis for {ticker.upper()}',
            yaxis_title='Price',
            xaxis_title='',
            template='plotly_dark',
            showlegend=True,
            height=800
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    fig = plot_hurst(
        df, confluences, ticker,
        hurst_window=hurst_window,
        ema50_period=ema50_period,
        ema200_period=ema200_period
    )
    st.plotly_chart(fig, use_container_width=True)

    # Step 8: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation(
        confluences, current_price, trend, momentum_reversal,
        hurst_threshold=0.5, ema50_period=50, ema200_period=200
    ):
        """
        Provide a detailed, actionable interpretation based on Hurst exponent and confluences in both English and Chinese.
        """
        interpretation_en = ""
        interpretation_cn = ""

        # 1. Trend Analysis
        interpretation_en += f"###### Current Market Trend: {trend}\n\n"
        interpretation_en += f"**Current Price**: {current_price:.2f}\n\n"
        interpretation_en += f"**Market Condition**: {momentum_reversal}\n\n"

        interpretation_cn += f"###### 当前市场趋势：{trend}\n\n"
        interpretation_cn += f"**当前价格**：{current_price:.2f}\n\n"
        interpretation_cn += f"**市场状况**：{momentum_reversal}\n\n"

        # 2. Confluence Analysis
        if confluences:
            interpretation_en += "###### Confluence Zones Detected:\n"
            interpretation_cn += "###### 检测到的共振区：\n"
            for key, indicators in confluences.items():
                if key == 'Trending Confluence':
                    interpretation_en += (
                        f"- **Trending Confluence**: Hurst Exponent is above {hurst_threshold} ({indicators['Hurst']:.2f}), "
                        f"and the price is above both EMA{ema50_period} ({indicators['EMA50']:.2f}) and EMA{ema200_period} ({indicators['EMA200']:.2f}).\n"
                    )
                    interpretation_cn += (
                        f"- **趋势共振区**：赫斯特指数高于 {hurst_threshold} ({indicators['Hurst']:.2f})，"
                        f"且价格高于 EMA{ema50_period} ({indicators['EMA50']:.2f}) 和 EMA{ema200_period} ({indicators['EMA200']:.2f})。\n"
                    )
                elif key == 'Mean-Reverting Confluence':
                    interpretation_en += (
                        f"- **Mean-Reverting Confluence**: Hurst Exponent is below {hurst_threshold} ({indicators['Hurst']:.2f}), "
                        f"and the price is below both EMA{ema50_period} ({indicators['EMA50']:.2f}) and EMA{ema200_period} ({indicators['EMA200']:.2f}).\n"
                    )
                    interpretation_cn += (
                        f"- **均值回归共振区**：赫斯特指数低于 {hurst_threshold} ({indicators['Hurst']:.2f})，"
                        f"且价格低于 EMA{ema50_period} ({indicators['EMA50']:.2f}) 和 EMA{ema200_period} ({indicators['EMA200']:.2f})。\n"
                    )
            interpretation_en += "\n"
            interpretation_cn += "\n"
        else:
            interpretation_en += "###### No Confluence Zones Detected.\n\n"
            interpretation_cn += "###### 未检测到共振区。\n\n"

        # 3. Price Position Analysis
        interpretation_en += "###### Price Position Relative to EMAs and Hurst Exponent:\n"
        interpretation_cn += "###### 当前价格相对于 EMA 和 Hurst 指数的位置：\n"
        if momentum_reversal == "Momentum (动量)":
            interpretation_en += f"- The current price is **above** EMA{ema50_period} and EMA{ema200_period}, with a Hurst Exponent above {hurst_threshold}, indicating strong **momentum**.\n"
            interpretation_cn += f"- 当前价格 **高于** EMA{ema50_period} 和 EMA{ema200_period}，且赫斯特指数高于 {hurst_threshold}，表明强劲的 **动量**。\n"
        elif momentum_reversal == "Reversal (反转)":
            interpretation_en += f"- The current price is **below** EMA{ema50_period} and EMA{ema200_period}, with a Hurst Exponent below {hurst_threshold}, indicating strong **reversal** tendencies.\n"
            interpretation_cn += f"- 当前价格 **低于** EMA{ema50_period} 和 EMA{ema200_period}，且赫斯特指数低于 {hurst_threshold}，表明强劲的 **反转** 趋势。\n"
        else:
            interpretation_en += f"- The current price is **between** EMA{ema50_period} and EMA{ema200_period}, with a Hurst Exponent around {hurst_threshold}, indicating a **sideways or neutral** market.\n"
            interpretation_cn += f"- 当前价格 **位于** EMA{ema50_period} 和 EMA{ema200_period} 之间，且赫斯特指数约为 {hurst_threshold}，表明 **横盘或中性** 市场。\n"
        interpretation_en += "\n"
        interpretation_cn += "\n"

        # 4. Actionable Recommendations
        interpretation_en += "###### Actionable Recommendations:\n"
        interpretation_cn += "###### 可操作的建议：\n"

        # Trending Confluence (Momentum)
        if 'Trending Confluence' in confluences:
            interpretation_en += (
                f"- **Buying Opportunity**: Consider buying when the Hurst Exponent remains above {hurst_threshold} "
                f"and the price is above EMA{ema50_period} and EMA{ema200_period}, confirming strong **momentum**.\n"
            )
            interpretation_cn += (
                f"- **买入机会**：当赫斯特指数保持在 {hurst_threshold} 以上，且价格高于 EMA{ema50_period} 和 EMA{ema200_period}，确认强劲的 **动量** 时，考虑买入。\n"
            )

        # Mean-Reverting Confluence (Reversal)
        if 'Mean-Reverting Confluence' in confluences:
            interpretation_en += (
                f"- **Selling Opportunity**: Consider selling when the Hurst Exponent remains below {hurst_threshold} "
                f"and the price is below EMA{ema50_period} and EMA{ema200_period}, confirming strong **reversal** tendencies.\n"
            )
            interpretation_cn += (
                f"- **卖出机会**：当赫斯特指数保持在 {hurst_threshold} 以下，且价格低于 EMA{ema50_period} 和 EMA{ema200_period}，确认强劲的 **反转** 趋势时，考虑卖出。\n"
            )

        # Momentum Scenario
        if momentum_reversal == "Momentum (动量)":
            interpretation_en += "\n- **Momentum Scenario**: In a momentum phase, consider holding long positions and setting trailing stop-loss orders to lock in profits.\n"
            interpretation_cn += "\n- **动量情景**：在动量阶段，考虑持有多头仓位并设置移动止损订单以锁定利润。\n"

        # Reversal Scenario
        if momentum_reversal == "Reversal (反转)":
            interpretation_en += "\n- **Reversal Scenario**: In a reversal phase, consider shorting the asset and setting stop-loss orders above recent highs.\n"
            interpretation_cn += "\n- **反转情景**：在反转阶段，考虑做空该资产并设置止损订单在近期高点之上。\n"

        # Confluence Zones
        if confluences:
            interpretation_en += "\n- **Confluence Zones**: Trades near these areas have a higher probability of success due to the alignment of the Hurst Exponent with EMAs.\n"
            interpretation_cn += "\n- **共振区**：由于赫斯特指数与 EMA 对齐，接近这些区域的交易成功概率更高。\n"

        # Risk Management
        interpretation_en += "\n###### Risk Management:\n"
        interpretation_cn += "\n###### 风险管理：\n"
        interpretation_en += "- **Stop-Loss**: Place stop-loss orders just beyond EMA50 or EMA200 to manage risk.\n"
        interpretation_cn += f"- **止损**：在 EMA{ema50_period} 或 EMA{ema200_period} 之外稍微放置止损订单以管理风险。\n"
        interpretation_en += "- **Take-Profit**: Set target levels based on recent support/resistance levels or use a trailing stop to lock in profits.\n"
        interpretation_cn += "- **止盈**：根据近期的支撑/阻力位设置目标水平或使用移动止盈以锁定利润。\n"

        # Market Conditions
        interpretation_en += "\n###### Optimal Market Conditions for Applying This Strategy:\n"
        interpretation_cn += "\n###### 应用此策略的最佳市场条件：\n"
        interpretation_en += "- **Momentum Markets**: Most effective in clear uptrends or downtrends where Hurst and EMAs confirm the direction.\n"
        interpretation_cn += "- **动量市场**：在 Hurst 和 EMA 确认方向的明显上升或下降趋势中最为有效。\n"
        interpretation_en += "- **Mean-Reverting Markets**: Effective in markets where prices oscillate around a mean.\n"
        interpretation_cn += "- **均值回归市场**：在价格围绕均值波动的市场中有效。\n"
        interpretation_en += "- **Avoid in Highly Volatile or Noisy Markets**: Hurst exponent may produce unreliable signals in choppy markets.\n"
        interpretation_cn += "- **避免在高波动或嘈杂市场**：在波动剧烈的市场中，赫斯特指数可能产生不可靠的信号。\n"

        return interpretation_en, interpretation_cn

    interpret_en, interpret_cn = detailed_interpretation(
        confluences, current_price, trend, momentum_reversal,
        hurst_threshold=0.5, ema50_period=ema50_period, ema200_period=ema200_period
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
