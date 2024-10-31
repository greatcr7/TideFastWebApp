from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
import pytz

from data.stock import get_stock_prices

# ---------------------------
# SuperTrend Calculation Function
# ---------------------------

def calculate_supertrend(df, atr_period=10, multiplier=3.0):
    """
    Calculate the SuperTrend indicator manually.
    """
    # Calculate ATR
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=atr_period, min_periods=1).mean()

    # Calculate Basic Upper and Lower Bands
    df['Basic_Upper'] = ((df['high'] + df['low']) / 2) + (multiplier * df['ATR'])
    df['Basic_Lower'] = ((df['high'] + df['low']) / 2) - (multiplier * df['ATR'])

    # Initialize Final Upper and Lower Bands
    df['Final_Upper'] = 0.0
    df['Final_Lower'] = 0.0

    for i in range(len(df)):
        if i == 0:
            df.at[i, 'Final_Upper'] = df.at[i, 'Basic_Upper']
            df.at[i, 'Final_Lower'] = df.at[i, 'Basic_Lower']
        else:
            # Final Upper Band
            if (df.at[i, 'Basic_Upper'] < df.at[i-1, 'Final_Upper']) or (df.at[i-1, 'close'] > df.at[i-1, 'Final_Upper']):
                df.at[i, 'Final_Upper'] = df.at[i, 'Basic_Upper']
            else:
                df.at[i, 'Final_Upper'] = df.at[i-1, 'Final_Upper']
            
            # Final Lower Band
            if (df.at[i, 'Basic_Lower'] > df.at[i-1, 'Final_Lower']) or (df.at[i-1, 'close'] < df.at[i-1, 'Final_Lower']):
                df.at[i, 'Final_Lower'] = df.at[i, 'Basic_Lower']
            else:
                df.at[i, 'Final_Lower'] = df.at[i-1, 'Final_Lower']

    # Determine SuperTrend
    df['SuperTrend'] = 0.0
    df['SuperTrend_Direction'] = True  # True for uptrend, False for downtrend

    for i in range(len(df)):
        if i == 0:
            df.at[i, 'SuperTrend'] = df.at[i, 'Final_Upper']
            df.at[i, 'SuperTrend_Direction'] = True
        else:
            if df.at[i, 'close'] > df.at[i-1, 'SuperTrend']:
                df.at[i, 'SuperTrend'] = df.at[i, 'Final_Lower']
                df.at[i, 'SuperTrend_Direction'] = True
            elif df.at[i, 'close'] < df.at[i-1, 'SuperTrend']:
                df.at[i, 'SuperTrend'] = df.at[i, 'Final_Upper']
                df.at[i, 'SuperTrend_Direction'] = False
            else:
                df.at[i, 'SuperTrend'] = df.at[i-1, 'SuperTrend']
                df.at[i, 'SuperTrend_Direction'] = df.at[i-1, 'SuperTrend_Direction']

    # Clean up intermediate columns
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR', 'Basic_Upper', 'Basic_Lower',
             'Final_Upper', 'Final_Lower'], axis=1, inplace=True)

    return df

# ---------------------------
# SuperTrend Analysis Function
# ---------------------------

def supertrend_analysis(ticker):
    st.markdown(f"# 📈 超级趋势 for {ticker.upper()}")

    # Sidebar for user inputs specific to SuperTrend Analysis
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

    # User input function with additional SuperTrend parameters
    def user_input_features():
        period = st.sidebar.selectbox(
            "📅 时间跨度 (Time Period)",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            help="选择分析的时间跨度。"
        )
        supertrend_atr_period = st.sidebar.number_input(
            "🔢 ATR 周期 (ATR Period)",
            min_value=1,
            max_value=100,
            value=10,
            help="用于计算ATR的周期，通常设为10。"
        )
        supertrend_multiplier = st.sidebar.number_input(
            "🔢 SuperTrend 乘数 (SuperTrend Multiplier)",
            min_value=0.1,
            max_value=10.0,
            value=3.0,
            step=0.1,
            help="SuperTrend计算的乘数，通常设为3.0。"
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
            start_date, end_date, supertrend_atr_period,
            supertrend_multiplier, ema50_period, ema200_period, peaks_prominence
        )

    # Getting user input
    (
        start_date, end_date, supertrend_atr_period,
        supertrend_multiplier, ema50_period, ema200_period, peaks_prominence
    ) = user_input_features()

    # Step 1: Fetch Historical Data using custom get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("❌ 未获取到数据。请检查股票代码并重试。")
        st.stop()

    # Ensure the data is sorted by date
    df = df.sort_values('date').reset_index(drop=True)

    # Step 2: Calculate SuperTrend
    df = calculate_supertrend(df, atr_period=supertrend_atr_period, multiplier=supertrend_multiplier)

    # Step 3: Identify Buy and Sell Signals
    def identify_signals(df):
        """
        Identify buy and sell signals based on SuperTrend crossings.
        """
        buy_signals = []
        sell_signals = []
        for i in range(1, len(df)):
            # Buy Signal: Price crosses above SuperTrend
            if (df['close'].iloc[i-1] < df['SuperTrend'].iloc[i-1]) and (df['close'].iloc[i] > df['SuperTrend'].iloc[i]):
                buy_signals.append(df['date'].iloc[i])
            # Sell Signal: Price crosses below SuperTrend
            elif (df['close'].iloc[i-1] > df['SuperTrend'].iloc[i-1]) and (df['close'].iloc[i] < df['SuperTrend'].iloc[i]):
                sell_signals.append(df['date'].iloc[i])
        return buy_signals, sell_signals

    buy_signals, sell_signals = identify_signals(df)

    # Step 4: Calculate EMAs
    df['EMA50'] = df['close'].ewm(span=ema50_period, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=ema200_period, adjust=False).mean()

    # Step 5: Identify Confluence with SuperTrend and EMAs
    def find_confluence(df, ema50_period=50, ema200_period=200):
        """
        Identify if SuperTrend aligns with EMAs.
        """
        latest_supertrend = df['SuperTrend'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]
        latest_price = df['close'].iloc[-1]
        supertrend_direction = df['SuperTrend_Direction'].iloc[-1]

        confluence_levels = {}

        # Define confluence based on SuperTrend and EMA alignment
        if supertrend_direction and latest_price > latest_ema50 and latest_price > latest_ema200:
            confluence_levels['Bullish Confluence'] = {
                'SuperTrend': latest_supertrend,
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }
        elif not supertrend_direction and latest_price < latest_ema50 and latest_price < latest_ema200:
            confluence_levels['Bearish Confluence'] = {
                'SuperTrend': latest_supertrend,
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }

        return confluence_levels, df

    confluences, df = find_confluence(df, ema50_period=ema50_period, ema200_period=ema200_period)

    # Step 6: Determine Market Trend Based on SuperTrend and EMAs
    def determine_trend(df, confluences):
        """
        Determine the current market trend based on SuperTrend and EMAs.
        """
        latest_price = df['close'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]
        supertrend_direction = df['SuperTrend_Direction'].iloc[-1]

        if supertrend_direction and latest_price > latest_ema50 and latest_price > latest_ema200:
            trend = "上升趋势 (Uptrend)"
            market_condition = "Momentum (动量)"
        elif not supertrend_direction and latest_price < latest_ema50 and latest_price < latest_ema200:
            trend = "下降趋势 (Downtrend)"
            market_condition = "Reversal (反转)"
        else:
            trend = "震荡区间 (Sideways)"
            market_condition = "Neutral (中性)"

        return trend, market_condition, latest_price

    trend, market_condition, current_price = determine_trend(df, confluences)

    # Step 7: Plot Using Plotly
    def plot_supertrend(df, buy_signals, sell_signals, confluences, ticker,
                       atr_period=10, multiplier=3.0, ema50_period=50, ema200_period=200):
        """
        Plot the SuperTrend along with price data and EMAs using Plotly.
        """
        fig = make_subplots(
            rows=1, cols=1,
            shared_xaxes=True,
            subplot_titles=(f'{ticker.upper()} 的股价、EMA 和 SuperTrend (Price, EMA, and SuperTrend)',),
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

        # SuperTrend
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['SuperTrend'],
                line=dict(color='orange', width=2),
                name='SuperTrend'
            ),
            row=1, col=1
        )

        # Buy Signals
        for signal in buy_signals:
            price = df.loc[df['date'] == signal, 'close'].values[0]
            fig.add_annotation(
                x=signal, y=price,
                text="🔼 Buy",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowcolor='green',
                ax=0, ay=-40
            )

        # Sell Signals
        for signal in sell_signals:
            price = df.loc[df['date'] == signal, 'close'].values[0]
            fig.add_annotation(
                x=signal, y=price,
                text="🔽 Sell",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowcolor='red',
                ax=0, ay=40
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

            fig.add_hline(
                y=value['SuperTrend'], line=dict(color=color, dash='dot'),
                row=1, col=1
            )
            # Optionally, add annotations
            fig.add_annotation(
                x=df['date'].iloc[-1],
                y=value['SuperTrend'],
                text=annotation_text,
                showarrow=False,
                yanchor="bottom" if color == 'green' else "top",
                font=dict(color=color),
                row=1, col=1
            )

        fig.update_layout(
            title=f'SuperTrend Analysis for {ticker.upper()}',
            yaxis_title='Price',
            xaxis_title='',
            template='plotly_dark',
            showlegend=True,
            height=800
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    fig = plot_supertrend(
        df, buy_signals, sell_signals, confluences, ticker,
        atr_period=supertrend_atr_period,
        multiplier=supertrend_multiplier,
        ema50_period=ema50_period,
        ema200_period=ema200_period
    )
    st.plotly_chart(fig, use_container_width=True)

    # Step 8: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation(
        buy_signals, sell_signals, confluences,
        current_price, trend, market_condition,
        supertrend_atr_period=10, supertrend_multiplier=3.0,
        ema50_period=50, ema200_period=200
    ):
        """
        Provide a detailed, actionable interpretation based on SuperTrend and confluences in both English and Chinese.
        """
        interpretation_en = ""
        interpretation_cn = ""

        # 1. Trend Analysis
        interpretation_en += f"###### Current Market Trend: {trend}\n\n"
        interpretation_en += f"**Current Price**: {current_price:.2f}\n\n"
        interpretation_en += f"**Market Condition**: {market_condition}\n\n"

        interpretation_cn += f"###### 当前市场趋势：{trend}\n\n"
        interpretation_cn += f"**当前价格**：{current_price:.2f}\n\n"
        interpretation_cn += f"**市场状况**：{market_condition}\n\n"

        # 2. Confluence Analysis
        if confluences:
            interpretation_en += "###### Confluence Zones Detected:\n"
            interpretation_cn += "###### 检测到的共振区：\n"
            for key, indicators in confluences.items():
                if key == 'Bullish Confluence':
                    interpretation_en += (
                        f"- **Bullish Confluence**: Price is above EMA{ema50_period} ({indicators['EMA50']:.2f}) and EMA{ema200_period} ({indicators['EMA200']:.2f}), "
                        f"and SuperTrend is indicating an uptrend.\n"
                    )
                    interpretation_cn += (
                        f"- **看涨共振区**：价格高于 EMA{ema50_period} ({indicators['EMA50']:.2f}) 和 EMA{ema200_period} ({indicators['EMA200']:.2f})，"
                        f"且 SuperTrend 指示上升趋势。\n"
                    )
                elif key == 'Bearish Confluence':
                    interpretation_en += (
                        f"- **Bearish Confluence**: Price is below EMA{ema50_period} ({indicators['EMA50']:.2f}) and EMA{ema200_period} ({indicators['EMA200']:.2f}), "
                        f"and SuperTrend is indicating a downtrend.\n"
                    )
                    interpretation_cn += (
                        f"- **看跌共振区**：价格低于 EMA{ema50_period} ({indicators['EMA50']:.2f}) 和 EMA{ema200_period} ({indicators['EMA200']:.2f})，"
                        f"且 SuperTrend 指示下降趋势。\n"
                    )
            interpretation_en += "\n"
            interpretation_cn += "\n"
        else:
            interpretation_en += "###### No Confluence Zones Detected.\n\n"
            interpretation_cn += "###### 未检测到共振区。\n\n"

        # 3. Buy and Sell Signals Analysis
        interpretation_en += "###### Buy and Sell Signals:\n"
        interpretation_cn += "###### 买入和卖出信号：\n"

        if buy_signals:
            interpretation_en += f"- **Total Buy Signals**: {len(buy_signals)}\n"
            interpretation_cn += f"- **总买入信号**：{len(buy_signals)}\n"
        else:
            interpretation_en += "- **No Buy Signals Detected.**\n"
            interpretation_cn += "- **未检测到买入信号。**\n"

        if sell_signals:
            interpretation_en += f"- **Total Sell Signals**: {len(sell_signals)}\n"
            interpretation_cn += f"- **总卖出信号**：{len(sell_signals)}\n"
        else:
            interpretation_en += "- **No Sell Signals Detected.**\n"
            interpretation_cn += "- **未检测到卖出信号。**\n"

        interpretation_en += "\n"
        interpretation_cn += "\n"

        # 4. Price Position Analysis
        interpretation_en += "###### Price Position Relative to EMAs and SuperTrend:\n"
        interpretation_cn += "###### 当前价格相对于 EMA 和 SuperTrend 的位置：\n"
        if market_condition == "Momentum (动量)":
            interpretation_en += f"- The current price is **above** EMA{ema50_period} and EMA{ema200_period}, with SuperTrend indicating an uptrend, signaling strong **momentum**.\n"
            interpretation_cn += f"- 当前价格 **高于** EMA{ema50_period} 和 EMA{ema200_period}，且 SuperTrend 指示上升趋势，表明强劲的 **动量**。\n"
        elif market_condition == "Reversal (反转)":
            interpretation_en += f"- The current price is **below** EMA{ema50_period} and EMA{ema200_period}, with SuperTrend indicating a downtrend, signaling strong **reversal** tendencies.\n"
            interpretation_cn += f"- 当前价格 **低于** EMA{ema50_period} 和 EMA{ema200_period}，且 SuperTrend 指示下降趋势，表明强劲的 **反转** 趋势。\n"
        else:
            interpretation_en += f"- The current price is **between** EMA{ema50_period} and EMA{ema200_period}, with SuperTrend indicating a neutral market condition.\n"
            interpretation_cn += f"- 当前价格 **位于** EMA{ema50_period} 和 EMA{ema200_period} 之间，且 SuperTrend 指示中性市场状况。\n"
        interpretation_en += "\n"
        interpretation_cn += "\n"

        # 5. Actionable Recommendations
        interpretation_en += "###### Actionable Recommendations:\n"
        interpretation_cn += "###### 可操作的建议：\n"

        # Bullish Confluence
        if 'Bullish Confluence' in confluences:
            interpretation_en += (
                f"- **Buying Opportunity**: Consider buying when the price remains above EMA{ema50_period} and EMA{ema200_period}, and SuperTrend indicates an uptrend, confirming strong **momentum**.\n"
            )
            interpretation_cn += (
                f"- **买入机会**：当价格保持在 EMA{ema50_period} 和 EMA{ema200_period} 以上，且 SuperTrend 指示上升趋势，确认强劲的 **动量** 时，考虑买入。\n"
            )

        # Bearish Confluence
        if 'Bearish Confluence' in confluences:
            interpretation_en += (
                f"- **Selling Opportunity**: Consider selling when the price remains below EMA{ema50_period} and EMA{ema200_period}, and SuperTrend indicates a downtrend, confirming strong **reversal** tendencies.\n"
            )
            interpretation_cn += (
                f"- **卖出机会**：当价格保持在 EMA{ema50_period} 和 EMA{ema200_period} 以下，且 SuperTrend 指示下降趋势，确认强劲的 **反转** 趋势时，考虑卖出。\n"
            )

        # Buy Signals
        if buy_signals:
            interpretation_en += (
                f"- **Buy Signals**: {len(buy_signals)} opportunities detected where the price crossed above SuperTrend. Consider entering long positions with appropriate stop-loss orders.\n"
            )
            interpretation_cn += (
                f"- **买入信号**：检测到 {len(buy_signals)} 次价格上穿 SuperTrend 的机会。考虑建立多头仓位并设置适当的止损订单。\n"
            )

        # Sell Signals
        if sell_signals:
            interpretation_en += (
                f"- **Sell Signals**: {len(sell_signals)} opportunities detected where the price crossed below SuperTrend. Consider entering short positions with appropriate stop-loss orders.\n"
            )
            interpretation_cn += (
                f"- **卖出信号**：检测到 {len(sell_signals)} 次价格下穿 SuperTrend 的机会。考虑建立空头仓位并设置适当的止损订单。\n"
            )

        # Confluence Zones
        if confluences:
            interpretation_en += (
                f"- **Confluence Zones**: Trades near these areas have a higher probability of success due to the alignment of SuperTrend with EMAs.\n"
            )
            interpretation_cn += (
                f"- **共振区**：由于 SuperTrend 与 EMA 对齐，接近这些区域的交易成功概率更高。\n"
            )

        # Breakout Scenarios
        interpretation_en += "\n###### Breakout Scenarios:\n"
        interpretation_cn += "\n###### 突破情景：\n"
        interpretation_en += (
            f"- **Bullish Breakout**: If the price breaks above EMA{ema200_period} with increasing volume, consider **entering a long position**.\n"
        )
        interpretation_cn += (
            f"- **看涨突破**：如果价格在成交量增加的情况下突破 EMA{ema200_period}，考虑 **建立多头仓位**。\n"
        )
        interpretation_en += (
            f"- **Bearish Breakout**: If the price breaks below EMA{ema200_period} with decreasing volume, consider **entering a short position**.\n"
        )
        interpretation_cn += (
            f"- **看跌突破**：如果价格在成交量减少的情况下突破 EMA{ema200_period}，考虑 **建立空头仓位**。\n"
        )

        # Risk Management
        interpretation_en += "\n###### Risk Management:\n"
        interpretation_cn += "\n###### 风险管理：\n"
        interpretation_en += f"- **Stop-Loss**: Place stop-loss orders just beyond EMA{ema50_period} or EMA{ema200_period} to manage risk.\n"
        interpretation_cn += f"- **止损**：在 EMA{ema50_period} 或 EMA{ema200_period} 之外稍微放置止损订单以管理风险。\n"
        interpretation_en += f"- **Take-Profit**: Set target levels based on recent support/resistance levels or use a trailing stop to lock in profits.\n"
        interpretation_cn += f"- **止盈**：根据近期的支撑/阻力位设置目标水平或使用移动止盈以锁定利润。\n"

        # Market Conditions
        interpretation_en += "\n###### Optimal Market Conditions for Applying This Strategy:\n"
        interpretation_cn += "\n###### 应用此策略的最佳市场条件：\n"
        interpretation_en += "- **Trending Markets**: Most effective in clear uptrends or downtrends where SuperTrend and EMAs confirm the direction.\n"
        interpretation_cn += "- **趋势市场**：在 SuperTrend 和 EMA 确认方向的明显上升或下降趋势中最为有效。\n"
        interpretation_en += "- **High Volume**: Ensure significant price movements are supported by high volume to validate SuperTrend signals.\n"
        interpretation_cn += "- **高成交量**：确保重要的价格波动由高成交量支持，以验证 SuperTrend 信号。\n"
        interpretation_en += "- **Avoid in Sideways/Noisy Markets**: SuperTrend may produce false signals in choppy or non-trending markets.\n"
        interpretation_cn += "- **避免在横盘/嘈杂市场**：在波动剧烈或无趋势的市场中，SuperTrend 可能产生虚假信号。\n"

        return interpretation_en, interpretation_cn

    interpret_en, interpret_cn = detailed_interpretation(
        buy_signals, sell_signals, confluences,
        current_price, trend, market_condition,
        supertrend_atr_period=supertrend_atr_period,
        supertrend_multiplier=supertrend_multiplier,
        ema50_period=ema50_period,
        ema200_period=ema200_period
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

