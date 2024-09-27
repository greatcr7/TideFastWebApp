from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf  # Using yfinance for data retrieval
from ta.trend import IchimokuIndicator, EMAIndicator
import pytz

from data.stock import get_stock_prices

# ---------------------------
# Ichimoku Cloud Analysis Function
# ---------------------------

def ichimoku_analysis(ticker):
    st.markdown(f"# 📈 一目均衡图 for {ticker.upper()}")

    # Sidebar for user inputs specific to Ichimoku Analysis
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
        period = st.sidebar.selectbox(
            "时间跨度 (Time Period)", 
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], 
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

    # Step 2: User Inputs for Ichimoku Parameters
    st.sidebar.header("一目均衡图参数")

    # Default Ichimoku periods
    default_conversion_line_period = 9
    default_base_line_period = 26
    default_span_b_period = 52
    default_displacement = 26

    conversion_line_period = st.sidebar.number_input(
        "转换线周期 (Conversion Line Period)", 
        min_value=1, 
        max_value=100, 
        value=default_conversion_line_period, 
        step=1,
        help="用于计算转换线的周期。推荐值：9。 (Period for calculating the Conversion Line. Recommended value: 9.)"
    )

    base_line_period = st.sidebar.number_input(
        "基准线周期 (Base Line Period)", 
        min_value=1, 
        max_value=200, 
        value=default_base_line_period, 
        step=1,
        help="用于计算基准线的周期。推荐值：26。 (Period for calculating the Base Line. Recommended value: 26.)"
    )

    span_b_period = st.sidebar.number_input(
        "领先跨度B周期 (Leading Span B Period)", 
        min_value=1, 
        max_value=300, 
        value=default_span_b_period, 
        step=1,
        help="用于计算领先跨度B的周期。推荐值：52。 (Period for calculating Leading Span B. Recommended value: 52.)"
    )

    displacement = st.sidebar.number_input(
        "位移 (Displacement)", 
        min_value=1, 
        max_value=100, 
        value=default_displacement, 
        step=1,
        help="领先跨度的位移。推荐值：26。 (Displacement for leading spans. Recommended value: 26.)"
    )

    # Additional Parameters
    st.sidebar.header("其他参数")

    ema_short_window = st.sidebar.number_input(
        "EMA 短期窗口 (EMA Short Window)", 
        min_value=10, 
        max_value=100, 
        value=50,  # Common default
        step=5,
        help="短期指数移动平均线（EMA）的窗口大小。较短的窗口使 EMA 更敏感。推荐值：50。 (Short-term EMA window size. A shorter window makes the EMA more sensitive. Recommended value: 50.)"
    )

    ema_long_window = st.sidebar.number_input(
        "EMA 长期窗口 (EMA Long Window)", 
        min_value=100, 
        max_value=300, 
        value=200,  # Common default
        step=10,
        help="长期指数移动平均线（EMA）的窗口大小。较长的窗口使 EMA 更平滑。推荐值：200。 (Long-term EMA window size. A longer window makes the EMA smoother. Recommended value: 200.)"
    )

    crossover_window = st.sidebar.number_input(
        "交叉检测窗口 (Crossover Window)", 
        min_value=1, 
        max_value=10, 
        value=1,  # Common default
        step=1,
        help="定义检测交叉的最小天数，以避免虚假信号。推荐值：1。 (Defines the minimum number of days to detect crossovers to avoid false signals. Recommended value: 1.)"
    )

    # Plotting Options
    st.sidebar.header("绘图选项")
    show_ema = st.sidebar.checkbox("显示 EMA (Show EMAs)", value=True)
    show_ichimoku = st.sidebar.checkbox("显示 Ichimoku Cloud (Show Ichimoku Cloud)", value=True)

    # Calculate Ichimoku using ta
    ichimoku_indicator = IchimokuIndicator(
        high=df['high'], 
        low=df['low'], 
        window1=conversion_line_period, 
        window2=base_line_period, 
        window3=span_b_period
    )
    df['Ichimoku_Conversion'] = ichimoku_indicator.ichimoku_conversion_line()
    df['Ichimoku_Base'] = ichimoku_indicator.ichimoku_base_line()
    df['Ichimoku_A'] = ichimoku_indicator.ichimoku_a()
    df['Ichimoku_B'] = ichimoku_indicator.ichimoku_b()
    df['Ichimoku_Cloud_Upper'] = df[['Ichimoku_A', 'Ichimoku_B']].max(axis=1)
    df['Ichimoku_Cloud_Lower'] = df[['Ichimoku_A', 'Ichimoku_B']].min(axis=1)

    # Calculate EMAs using ta
    ema_short_indicator = EMAIndicator(close=df['close'], window=ema_short_window)
    ema_long_indicator = EMAIndicator(close=df['close'], window=ema_long_window)
    df['EMA_Short'] = ema_short_indicator.ema_indicator()
    df['EMA_Long'] = ema_long_indicator.ema_indicator()

    # Identify Crossovers between EMAs
    def identify_crossovers(df, short_col='EMA_Short', long_col='EMA_Long', window=1):
        bullish_crossovers = []
        bearish_crossovers = []

        for i in range(window, len(df)):
            if (df[short_col].iloc[i] > df[long_col].iloc[i]) and (df[short_col].iloc[i - window] <= df[long_col].iloc[i - window]):
                bullish_crossovers.append({
                    'Date': df['date'].iloc[i],
                    'Price': df['close'].iloc[i],
                    short_col: df[short_col].iloc[i],
                    long_col: df[long_col].iloc[i]
                })
            elif (df[short_col].iloc[i] < df[long_col].iloc[i]) and (df[short_col].iloc[i - window] >= df[long_col].iloc[i - window]):
                bearish_crossovers.append({
                    'Date': df['date'].iloc[i],
                    'Price': df['close'].iloc[i],
                    short_col: df[short_col].iloc[i],
                    long_col: df[long_col].iloc[i]
                })

        return bullish_crossovers, bearish_crossovers

    bullish_crossovers, bearish_crossovers = identify_crossovers(df, window=crossover_window)

    # Identify Confluences
    def find_confluence(df, ema_short, ema_long, ichimoku_a, ichimoku_b, price_col='close'):
        latest_ema_short = df['EMA_Short'].iloc[-1]
        latest_ema_long = df['EMA_Long'].iloc[-1]
        latest_ichimoku_a = df['Ichimoku_A'].iloc[-1]
        latest_ichimoku_b = df['Ichimoku_B'].iloc[-1]
        latest_price = df[price_col].iloc[-1]

        confluence_levels = {}

        # Bullish Confluence
        if (latest_ema_short > latest_ema_long) and (latest_price > latest_ema_short) and (latest_ichimoku_a > latest_ichimoku_b):
            confluence_levels['Bullish Confluence'] = {
                'EMA_Short': latest_ema_short,
                'EMA_Long': latest_ema_long,
                'Ichimoku_A': latest_ichimoku_a,
                'Ichimoku_B': latest_ichimoku_b,
                'Price': latest_price
            }
        # Bearish Confluence
        elif (latest_ema_short < latest_ema_long) and (latest_price < latest_ema_short) and (latest_ichimoku_a < latest_ichimoku_b):
            confluence_levels['Bearish Confluence'] = {
                'EMA_Short': latest_ema_short,
                'EMA_Long': latest_ema_long,
                'Ichimoku_A': latest_ichimoku_a,
                'Ichimoku_B': latest_ichimoku_b,
                'Price': latest_price
            }

        return confluence_levels, df

    confluences, df = find_confluence(df, ema_short_window, ema_long_window, 'Ichimoku_A', 'Ichimoku_B')


    # Determine Trend
    def determine_trend(df, confluences):
        latest_price = df['close'].iloc[-1]
        latest_ema_short = df['EMA_Short'].iloc[-1]
        latest_ema_long = df['EMA_Long'].iloc[-1]
        latest_ichimoku_a = df['Ichimoku_A'].iloc[-1]
        latest_ichimoku_b = df['Ichimoku_B'].iloc[-1]

        if (latest_price > latest_ema_short) and (latest_price > latest_ema_long) and (latest_ichimoku_a > latest_ichimoku_b):
            trend = "上升趋势 (Uptrend)"
        elif (latest_price < latest_ema_short) and (latest_price < latest_ema_long) and (latest_ichimoku_a < latest_ichimoku_b):
            trend = "下降趋势 (Downtrend)"
        else:
            trend = "震荡区间 (Sideways)"

        return trend, latest_price

    trend, current_price = determine_trend(df, confluences)

    # Step 3: Plot Using Plotly
    def plot_ichimoku(df, bullish_crossovers, bearish_crossovers, confluences, ticker, 
                     show_ema=True, show_ichimoku=True):
        """
        Plot the Ichimoku Cloud along with price data and EMAs using Plotly.
        """
        fig = make_subplots(
            rows=1, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, 
            subplot_titles=(f'{ticker} 的股价和 Ichimoku Cloud (Price and Ichimoku Cloud)'),
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

        # Ichimoku Cloud
        if show_ichimoku:
            # Senkou Span A and B to form the cloud
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['Ichimoku_A'], 
                    line=dict(color='green', width=1, dash='dot'), 
                    name='Ichimoku A'
                ), 
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['Ichimoku_B'], 
                    line=dict(color='red', width=1, dash='dot'), 
                    name='Ichimoku B'
                ), 
                row=1, col=1
            )
            # Fill the cloud
            fig.add_traces([
                go.Scatter(
                    x=pd.concat([df['date'], df['date'][::-1]]),
                    y=pd.concat([df['Ichimoku_A'], df['Ichimoku_B'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(0, 255, 0, 0.1)' if df['Ichimoku_A'].iloc[-1] > df['Ichimoku_B'].iloc[-1] else 'rgba(255, 0, 0, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                )
            ])

        # Crossovers Markers on Price Chart
        crossover_dates_bull = [div['Date'] for div in bullish_crossovers]
        crossover_prices_bull = [div['Price'] for div in bullish_crossovers]
        crossover_dates_sell = [div['Date'] for div in bearish_crossovers]
        crossover_prices_sell = [div['Price'] for div in bearish_crossovers]

        if crossover_dates_bull and crossover_prices_bull:
            fig.add_trace(
                go.Scatter(
                    mode='markers', 
                    x=crossover_dates_bull, 
                    y=crossover_prices_bull,
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    name='看涨交叉 (Bullish Crossover)'
                ), 
                row=1, col=1
            )
        if crossover_dates_sell and crossover_prices_sell:
            fig.add_trace(
                go.Scatter(
                    mode='markers', 
                    x=crossover_dates_sell, 
                    y=crossover_prices_sell,
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    name='看跌交叉 (Bearish Crossover)'
                ), 
                row=1, col=1
            )

        # Highlight Confluence Zones
        for key, value in confluences.items():
            if key == 'Bullish Confluence':
                color = 'green'
                annotation_text = '看涨共振区 (Bullish Confluence)'
            elif key == 'Bearish Confluence':
                color = 'red'
                annotation_text = '看跌共振区 (Bearish Confluence)'
            else:
                color = 'yellow'
                annotation_text = '共振区 (Confluence)'

            fig.add_hline(
                y=value['Price'], 
                line=dict(color=color, dash='dash'), 
                row=1, col=1,
                annotation_text=annotation_text,
                annotation_position="top left"
            )

        fig.update_layout(
            title=f"{ticker} 的 Ichimoku Cloud 分析",
            yaxis_title='价格 (Price)',
            xaxis_title='日期 (Date)',
            template='plotly_dark',
            showlegend=True,
            height=800
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    fig = plot_ichimoku(df, bullish_crossovers, bearish_crossovers, confluences, ticker, 
                       show_ema=show_ema, show_ichimoku=show_ichimoku)
    st.plotly_chart(fig, use_container_width=True)

     # Step 4: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation(bullish_crossovers, bearish_crossovers, confluences, current_price, trend, 
                                conversion_line_period, base_line_period, span_b_period, displacement,
                                ema_short_window, ema_long_window):
        """
        Provide a detailed, actionable interpretation based on Ichimoku and crossovers in both English and Chinese.
        """
        interpretation_en = ""
        interpretation_cn = ""

        # 1. Trend Analysis
        interpretation_en += f"###### Current Market Trend: {trend}\n\n"
        interpretation_en += f"**Current Price**: {current_price:.2f}\n\n"

        interpretation_cn += f"###### 当前市场趋势：{trend}\n\n"
        interpretation_cn += f"**当前价格**：{current_price:.2f}\n\n"

        # 2. Ichimoku Cloud Components Explanation
        interpretation_en += "##### 📝 Ichimoku Cloud Components Explanation\n"
        interpretation_en += f"""
- **Conversion Line (Tenkan-sen)**: Calculated over **{conversion_line_period} periods**, it represents the average of the highest high and the lowest low over the specified periods. It is a faster-moving average that responds quickly to price changes.
- **Base Line (Kijun-sen)**: Calculated over **{base_line_period} periods**, it serves as an indicator of future price movement. A higher Kijun-sen suggests upward momentum, while a lower Kijun-sen indicates downward momentum.
- **Leading Span B (Senkou Span B)**: Calculated over **{span_b_period} periods**, it forms one boundary of the Ichimoku Cloud. It provides support and resistance levels and helps determine the overall trend.
- **Displacement**: The **{displacement} period(s)** displacement projects the Ichimoku Cloud into the future, creating the Senkou Span A and Senkou Span B lines which form the cloud.
"""

        interpretation_cn += "##### 📝 Ichimoku 云组件解释\n"
        interpretation_cn += f"""
- **转换线 (Tenkan-sen)**：计算{conversion_line_period}个周期的最高价和最低价的平均值。它是一个快速移动的平均线，能够迅速响应价格变化。
- **基准线 (Kijun-sen)**：计算{base_line_period}个周期的最高价和最低价的平均值。作为未来价格走势的指示器，基准线较高表明上升动能，较低则表明下降动能。
- **领先跨度B (Senkou Span B)**：计算{span_b_period}个周期的最高价和最低价的平均值，形成 Ichimoku 云的一个边界。它提供支撑和阻力位，帮助确定整体趋势。
- **位移**：将 Ichimoku 云向未来位移{displacement}个周期，创建领先跨度A和领先跨度B线，形成云层。
"""

        # 3. Confluence Analysis
        if confluences:
            interpretation_en += "##### 📍 Confluence Zones Detected:\n"
            interpretation_cn += "##### 📍 检测到的共振区：\n"
            for key, indicators in confluences.items():
                if key == 'Bullish Confluence':
                    interpretation_en += (f"- **Bullish Confluence**: EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) is above EMA{ema_long_window} "
                                           f"({indicators['EMA_Long']:.2f}), Ichimoku A ({indicators['Ichimoku_A']:.2f}) is above Ichimoku B "
                                           f"({indicators['Ichimoku_B']:.2f}), and the price is above the cloud. This alignment confirms strong bullish momentum.\n")
                    interpretation_cn += (f"- **看涨共振区**：EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) 高于 EMA{ema_long_window} "
                                           f"({indicators['EMA_Long']:.2f})，Ichimoku A ({indicators['Ichimoku_A']:.2f}) 高于 Ichimoku B "
                                           f"({indicators['Ichimoku_B']:.2f})，且价格高于云。这种对齐确认了强劲的看涨动能。\n")
                elif key == 'Bearish Confluence':
                    interpretation_en += (f"- **Bearish Confluence**: EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) is below EMA{ema_long_window} "
                                           f"({indicators['EMA_Long']:.2f}), Ichimoku A ({indicators['Ichimoku_A']:.2f}) is below Ichimoku B "
                                           f"({indicators['Ichimoku_B']:.2f}), and the price is below the cloud. This alignment confirms strong bearish momentum.\n")
                    interpretation_cn += (f"- **看跌共振区**：EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) 低于 EMA{ema_long_window} "
                                           f"({indicators['EMA_Long']:.2f})，Ichimoku A ({indicators['Ichimoku_A']:.2f}) 低于 Ichimoku B "
                                           f"({indicators['Ichimoku_B']:.2f})，且价格低于云。这种对齐确认了强劲的卖出动能。\n")
            interpretation_en += "\n"
            interpretation_cn += "\n"
        else:
            interpretation_en += "##### 📍 No Confluence Zones Detected.\n\n"
            interpretation_cn += "##### 📍 未检测到共振区。\n\n"

        # 4. Price Position Analysis
        interpretation_en += "##### 🔍 Price Position Relative to Ichimoku Cloud and EMAs:\n"
        interpretation_cn += "##### 🔍 当前价格相对于 Ichimoku 云和 EMA 的位置：\n"
        if trend == "上升趋势 (Uptrend)":
            interpretation_en += f"- The current price is **above** EMA{ema_short_window} and EMA{ema_long_window}, Ichimoku A is above Ichimoku B, and the price is above the cloud. This indicates a strong **uptrend** with robust buying pressure.\n"
            interpretation_cn += f"- 当前价格 **高于** EMA{ema_short_window} 和 EMA{ema_long_window}，Ichimoku A 高于 Ichimoku B，且价格高于云。这表明一个强劲的 **上升趋势**，具有强大的买入压力。\n"
        elif trend == "下降趋势 (Downtrend)":
            interpretation_en += f"- The current price is **below** EMA{ema_short_window} and EMA{ema_long_window}, Ichimoku A is below Ichimoku B, and the price is below the cloud. This indicates a strong **downtrend** with robust selling pressure.\n"
            interpretation_cn += f"- 当前价格 **低于** EMA{ema_short_window} 和 EMA{ema_long_window}，Ichimoku A 低于 Ichimoku B，且价格低于云。这表明一个强劲的 **下降趋势**，具有强大的卖出压力。\n"
        else:
            interpretation_en += f"- The current price is **within** EMA{ema_short_window} and EMA{ema_long_window}, Ichimoku A and B are interchanging, and the price is within the cloud. This indicates a **consolidating** or **sideways market** with no clear trend.\n"
            interpretation_cn += f"- 当前价格 **位于** EMA{ema_short_window} 和 EMA{ema_long_window} 之间，Ichimoku A 和 B 正在交替，且价格位于云内。这表明一个 **整合** 或 **横盘市场**，没有明显的趋势。\n"
        interpretation_en += "\n"
        interpretation_cn += "\n"

        # 5. Actionable Recommendations
        interpretation_en += "##### 💡 Actionable Recommendations:\n"
        interpretation_cn += "##### 💡 可操作的建议：\n"

        # Bullish Confluence
        if 'Bullish Confluence' in confluences:
            interpretation_en += f"- **Buying Opportunity**: When EMA{ema_short_window} is above EMA{ema_long_window}, Ichimoku A is above Ichimoku B, and the price is above the cloud, consider **entering a long position**. This alignment confirms strong bullish momentum.\n"
            interpretation_cn += f"- **买入机会**：当 EMA{ema_short_window} 高于 EMA{ema_long_window}，Ichimoku A 高于 Ichimoku B，且价格高于云时，考虑 **进入多头仓位**。这种对齐确认了强劲的看涨动能。\n"

        # Bearish Confluence
        if 'Bearish Confluence' in confluences:
            interpretation_en += f"- **Selling Opportunity**: When EMA{ema_short_window} is below EMA{ema_long_window}, Ichimoku A is below Ichimoku B, and the price is below the cloud, consider **entering a short position**. This alignment confirms strong bearish momentum.\n"
            interpretation_cn += f"- **卖出机会**：当 EMA{ema_short_window} 低于 EMA{ema_long_window}，Ichimoku A 低于 Ichimoku B，且价格低于云时，考虑 **进入空头仓位**。这种对齐确认了强劲的卖出动能。\n"

        # Bullish Crossovers
        if bullish_crossovers:
            interpretation_en += "\n- **Bullish Crossover Detected**: EMA{ema_short_window} has crossed above EMA{ema_long_window}, indicating a potential **upward trend**. Consider **entering a long position** when confirmed by bullish candlestick patterns.\n"
            interpretation_cn += "\n- **检测到看涨交叉**：EMA{ema_short_window} 已经上穿 EMA{ema_long_window}，表明可能出现 **上升趋势**。当通过看涨的烛台形态确认时，考虑 **进入多头仓位**。\n"

        # Bearish Crossovers
        if bearish_crossovers:
            interpretation_en += "\n- **Bearish Crossover Detected**: EMA{ema_short_window} has crossed below EMA{ema_long_window}, indicating a potential **downward trend**. Consider **entering a short position** when confirmed by bearish candlestick patterns.\n"
            interpretation_cn += "\n- **检测到卖出交叉**：EMA{ema_short_window} 已经下穿 EMA{ema_long_window}，表明可能出现 **下降趋势**。当通过看跌的烛台形态确认时，考虑 **进入空头仓位**。\n"

        # Confluence Zones
        if confluences:
            interpretation_en += "\n- **Confluence Zones**: Trades near these areas have a higher probability of success due to the alignment of Ichimoku components with EMAs. Monitor these zones closely for potential trade opportunities.\n"
            interpretation_cn += "\n- **共振区**：由于 Ichimoku 组件与 EMA 对齐，接近这些区域的交易成功概率更高。密切关注这些区域以寻找潜在的交易机会。\n"

        # Risk Management
        interpretation_en += "\n##### ⚠️ Risk Management:\n"
        interpretation_cn += "\n##### ⚠️ 风险管理：\n"
        interpretation_en += f"- **Stop-Loss**: Place stop-loss orders just below EMA{ema_short_window} or above EMA{ema_long_window} to manage risk in long and short positions respectively.\n"
        interpretation_cn += f"- **止损**：在多头仓位中，将止损订单放置在 EMA{ema_short_window} 或 EMA{ema_long_window} 下方，以管理风险；在空头仓位中，将止损订单放置在 EMA{ema_long_window} 上方，以管理风险。\n"
        interpretation_en += f"- **Take-Profit**: Set target levels based on recent support/resistance levels or use a trailing stop to lock in profits as the trend continues.\n"
        interpretation_cn += f"- **止盈**：根据近期的支撑/阻力位设置目标水平，或使用移动止盈以在趋势持续时锁定利润。\n"

        # Market Conditions
        interpretation_en += "\n##### 🌐 Optimal Market Conditions for Applying This Strategy:\n"
        interpretation_cn += "\n##### 🌐 应用此策略的最佳市场条件：\n"
        interpretation_en += "- **Trending Markets**: Most effective in clear uptrends or downtrends where Ichimoku and EMAs confirm the direction.\n"
        interpretation_cn += "- **趋势市场**：在 Ichimoku 和 EMA 确认方向的明显上升或下降趋势中最为有效。\n"
        interpretation_en += "- **High Volume**: Ensure significant price movements are supported by high volume to validate Ichimoku signals.\n"
        interpretation_cn += "- **高成交量**：确保重要的价格波动由高成交量支持，以验证 Ichimoku 信号。\n"
        interpretation_en += "- **Avoid in Sideways/Noisy Markets**: Ichimoku may produce false signals in choppy or non-trending markets.\n"
        interpretation_cn += "- **避免在横盘/嘈杂市场**：在波动剧烈或无趋势的市场中，Ichimoku 可能产生虚假信号。\n"

        return interpretation_en, interpretation_cn

    interpret_en, interpret_cn = detailed_interpretation(
        bullish_crossovers, bearish_crossovers, confluences, current_price, trend,
        conversion_line_period, base_line_period, span_b_period, displacement,
        ema_short_window, ema_long_window
    )

    # Display Interpretations
    st.markdown("##### 📄 指标解读")

    # Tabs for English and Chinese
    tab1, tab2 = st.tabs(["中文", "English"])

    with tab1:
        st.markdown(interpret_cn)

    with tab2:
        st.markdown(interpret_en)

    # Optional: Display Data Table
    with st.expander("📊 查看原始数据 (View Raw Data)"):
        st.dataframe(df)

        