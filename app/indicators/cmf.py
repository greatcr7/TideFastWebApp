from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from data.stock import get_stock_prices  # Ensure this module is available
from ta.volume import ChaikinMoneyFlowIndicator
from ta.trend import EMAIndicator
import pytz

# ---------------------------
# CMF Analysis Function
# ---------------------------

def cmf_analysis(ticker):
    st.markdown(f"# 📈 蔡金资金流量 (CMF) for {ticker.upper()}")

    # Sidebar for user inputs specific to CMF Analysis
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

    # Step 1: Fetch Historical Data using custom get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("未获取到数据。请检查股票代码并重试。 (No data fetched. Please check the ticker symbol and try again.)")
        st.stop()

    # Step 2: User Inputs for CMF Parameters
    st.sidebar.header("CMF 参数")

    cmf_period = st.sidebar.number_input(
        "周期 (Period)", 
        min_value=1, 
        max_value=100, 
        value=20,  # Common default
        step=1,
        help="用于计算 CMF 的周期。推荐值：20。 (The period over which CMF is calculated. Recommended value: 20.)"
    )

    # Additional Parameters
    st.sidebar.header("其他参数 (Other Parameters)")

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
    st.sidebar.header("绘图选项 (Plotting Options)")
    show_ema = st.sidebar.checkbox("显示 EMA (Show EMAs)", value=True)
    show_cmf = st.sidebar.checkbox("显示 CMF (Show CMF)", value=True)

    # Calculate CMF using ta
    cmf_indicator = ChaikinMoneyFlowIndicator(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        volume=df['volume'], 
        window=cmf_period
    )
    df['CMF'] = cmf_indicator.chaikin_money_flow()

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
    def find_confluence(df, ema_short, ema_long, cmf_col='CMF', price_col='close'):
        latest_cmf = df[cmf_col].iloc[-1]
        latest_ema_short = df['EMA_Short'].iloc[-1]
        latest_ema_long = df['EMA_Long'].iloc[-1]
        latest_price = df[price_col].iloc[-1]

        confluence_levels = {}

        if (latest_cmf > 0) and (df['EMA_Short'].iloc[-1] > df['EMA_Long'].iloc[-1]):
            confluence_levels['Bullish Confluence'] = {
                'CMF': latest_cmf,
                'EMA_Short': latest_ema_short,
                'EMA_Long': latest_ema_long
            }
        elif (latest_cmf < 0) and (df['EMA_Short'].iloc[-1] < df['EMA_Long'].iloc[-1]):
            confluence_levels['Bearish Confluence'] = {
                'CMF': latest_cmf,
                'EMA_Short': latest_ema_short,
                'EMA_Long': latest_ema_long
            }

        return confluence_levels, df

    confluences, df = find_confluence(df, ema_short_window, ema_long_window)

    # Determine Trend
    def determine_trend(df, confluences):
        latest_cmf = df['CMF'].iloc[-1]
        latest_ema_short = df['EMA_Short'].iloc[-1]
        latest_ema_long = df['EMA_Long'].iloc[-1]
        latest_price = df['close'].iloc[-1]

        if (latest_cmf > 0) and (latest_ema_short > latest_ema_long) and (latest_price > latest_ema_short):
            trend = "上升趋势 (Uptrend)"
        elif (latest_cmf < 0) and (latest_ema_short < latest_ema_long) and (latest_price < latest_ema_short):
            trend = "下降趋势 (Downtrend)"
        else:
            trend = "震荡区间 (Sideways)"

        return trend, latest_price

    trend, current_price = determine_trend(df, confluences)

    # Step 3: Plot Using Plotly
    def plot_cmf(df, bullish_crossovers, bearish_crossovers, confluences, ticker, show_ema=True, show_cmf=True):
        """
        Plot the CMF along with price data and EMAs using Plotly.
        """
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, 
            subplot_titles=(f'{ticker}的股价和移动平均线 (Price and EMAs)', 'Chaikin Money Flow (CMF)'),
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

        # CMF Line
        if show_cmf:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['CMF'], 
                    line=dict(color='orange', width=2), 
                    name='CMF'
                ), 
                row=2, col=1
            )
            # Add zero line
            fig.add_hline(
                y=0, 
                line=dict(color='gray', dash='dash'), 
                row=2, col=1
            )

        # Crossovers Markers on Price Chart
        crossover_dates_bull = [div['Date'] for div in bullish_crossovers]
        crossover_prices_bull = [div['Price'] for div in bullish_crossovers]
        crossover_dates_bear = [div['Date'] for div in bearish_crossovers]
        crossover_prices_bear = [div['Price'] for div in bearish_crossovers]

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
        fig.add_trace(
            go.Scatter(
                mode='markers', 
                x=crossover_dates_bear, 
                y=crossover_prices_bear,
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='看跌交叉 (Bearish Crossover)'
            ), 
            row=1, col=1
        )

        # Highlight Confluence Zones on Price Chart
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
                y=value['EMA_Short'], 
                line=dict(color=color, dash='dot'), 
                row=1, col=1,
                annotation_text=annotation_text,
                annotation_position="top left"
            )

        # Update Layout
        fig.update_layout(
            title=f"{ticker} 的 Chaikin Money Flow (CMF) 分析",
            yaxis_title='价格 (Price)',
            xaxis_title='',
            template='plotly_dark',
            showlegend=True,
            height=900
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    fig = plot_cmf(df, bullish_crossovers, bearish_crossovers, confluences, ticker, show_ema=show_ema, show_cmf=show_cmf)
    st.plotly_chart(fig, use_container_width=True)

    # Step 4: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation(bullish_crossovers, bearish_crossovers, confluences, current_price, trend):
        """
        Provide a detailed, actionable interpretation based on CMF and crossovers in both English and Chinese.
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
                    interpretation_en += f"- **Bullish Confluence**: CMF is positive, EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) is above EMA{ema_long_window} ({indicators['EMA_Long']:.2f}), and the price is above EMA{ema_short_window}.\n"
                    interpretation_cn += f"- **看涨共振区**：CMF 为正，EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) 高于 EMA{ema_long_window} ({indicators['EMA_Long']:.2f})，且价格高于 EMA{ema_short_window}。\n"
                elif key == 'Bearish Confluence':
                    interpretation_en += f"- **Bearish Confluence**: CMF is negative, EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) is below EMA{ema_long_window} ({indicators['EMA_Long']:.2f}), and the price is below EMA{ema_short_window}.\n"
                    interpretation_cn += f"- **看跌共振区**：CMF 为负，EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) 低于 EMA{ema_long_window} ({indicators['EMA_Long']:.2f})，且价格低于 EMA{ema_short_window}。\n"
            interpretation_en += "\n"
            interpretation_cn += "\n"
        else:
            interpretation_en += "###### No Confluence Zones Detected.\n\n"
            interpretation_cn += "###### 未检测到共振区。\n\n"

        # 3. CMF Analysis
        latest_cmf = df['CMF'].iloc[-1]
        interpretation_en += f"###### Chaikin Money Flow (CMF): {latest_cmf:.4f}\n\n"
        interpretation_cn += f"###### Chaikin Money Flow (CMF)：{latest_cmf:.4f}\n\n"

        if latest_cmf > 0:
            interpretation_en += "- **Positive CMF**: Indicates buying pressure is dominant.\n"
            interpretation_cn += "- **正的 CMF**：表明买盘压力占优。\n"
        elif latest_cmf < 0:
            interpretation_en += "- **Negative CMF**: Indicates selling pressure is dominant.\n"
            interpretation_cn += "- **负的 CMF**：表明卖盘压力占优。\n"
        else:
            interpretation_en += "- **Neutral CMF**: No clear buying or selling pressure.\n"
            interpretation_cn += "- **中性的 CMF**：无明显的买盘或卖盘压力。\n"

        interpretation_en += "\n"
        interpretation_cn += "\n"

        # 4. Price Position Analysis
        interpretation_en += "###### Price Position Relative to EMAs:\n"
        interpretation_cn += "###### 当前价格相对于 EMA 的位置：\n"
        if trend == "上升趋势 (Uptrend)":
            interpretation_en += f"- The current price is **above** EMA{ema_short_window} and EMA{ema_long_window}, indicating strong bullish momentum.\n"
            interpretation_cn += f"- 当前价格 **高于** EMA{ema_short_window} 和 EMA{ema_long_window}，表明强劲的看涨动能。\n"
        elif trend == "下降趋势 (Downtrend)":
            interpretation_en += f"- The current price is **below** EMA{ema_short_window} and EMA{ema_long_window}, indicating strong bearish momentum.\n"
            interpretation_cn += f"- 当前价格 **低于** EMA{ema_short_window} 和 EMA{ema_long_window}，表明强劲的卖出动能。\n"
        else:
            interpretation_en += f"- The current price is **between** EMA{ema_short_window} and EMA{ema_long_window}, indicating a consolidating market.\n"
            interpretation_cn += f"- 当前价格 **位于** EMA{ema_short_window} 和 EMA{ema_long_window} 之间，表明市场在整合中。\n"
        interpretation_en += "\n"
        interpretation_cn += "\n"

        # 5. Actionable Recommendations
        interpretation_en += "###### Actionable Recommendations:\n"
        interpretation_cn += "###### 可操作的建议：\n"

        # Bullish Confluence
        if 'Bullish Confluence' in confluences:
            interpretation_en += "- **Buying Opportunity**: Consider buying when CMF is positive, EMA{ema_short_window} is above EMA{ema_long_window}, and the price is above EMA{ema_short_window}, confirming strong bullish momentum.\n"
            interpretation_cn += "- **买入机会**：当 CMF 为正，EMA{ema_short_window} 高于 EMA{ema_long_window}，且价格高于 EMA{ema_short_window}，确认强劲的看涨动能时，考虑买入。\n"

        # Bearish Confluence
        if 'Bearish Confluence' in confluences:
            interpretation_en += "- **Selling Opportunity**: Consider selling when CMF is negative, EMA{ema_short_window} is below EMA{ema_long_window}, and the price is below EMA{ema_short_window}, confirming strong bearish momentum.\n"
            interpretation_cn += "- **卖出机会**：当 CMF 为负，EMA{ema_short_window} 低于 EMA{ema_long_window}，且价格低于 EMA{ema_short_window}，确认强劲的卖出动能时，考虑卖出。\n"

        # Bullish Crossovers
        if bullish_crossovers:
            interpretation_en += "\n- **Bullish Crossover Detected**: EMA{ema_short_window} has crossed above EMA{ema_long_window}, indicating potential upward trend. Consider entering a long position when confirmed by bullish candlestick patterns.\n"
            interpretation_cn += "\n- **检测到看涨交叉**：EMA{ema_short_window} 已经上穿 EMA{ema_long_window}，表明可能出现上升趋势。当通过看涨的烛台形态确认时，考虑买入。\n"

        # Bearish Crossovers
        if bearish_crossovers:
            interpretation_en += "\n- **Bearish Crossover Detected**: EMA{ema_short_window} has crossed below EMA{ema_long_window}, indicating potential downward trend. Consider entering a short position when confirmed by bearish candlestick patterns.\n"
            interpretation_cn += "\n- **检测到看跌交叉**：EMA{ema_short_window} 已经下穿 EMA{ema_long_window}，表明可能出现下降趋势。当通过看跌的烛台形态确认时，考虑卖出。\n"

        # Confluence Zones
        if confluences:
            interpretation_en += "\n- **Confluence Zones**: Trades near these areas have a higher probability of success due to the alignment of CMF with EMAs.\n"
            interpretation_cn += "\n- **共振区**：由于 CMF 与 EMA 对齐，接近这些区域的交易成功概率更高。\n"

        # Risk Management
        interpretation_en += "\n###### Risk Management:\n"
        interpretation_cn += "\n###### 风险管理：\n"
        interpretation_en += "- **Stop-Loss**: Place stop-loss orders just beyond EMA{ema_short_window} or EMA{ema_long_window} to manage risk.\n"
        interpretation_cn += "- **止损**：在 EMA{ema_short_window} 或 EMA{ema_long_window} 之外稍微放置止损订单以管理风险。\n"
        interpretation_en += "- **Take-Profit**: Set target levels based on recent support/resistance levels or use a trailing stop to lock in profits.\n"
        interpretation_cn += "- **止盈**：根据近期的支撑/阻力位设置目标水平或使用移动止盈以锁定利润。\n"

        # Market Conditions
        interpretation_en += "\n###### Optimal Market Conditions for Applying This Strategy:\n"
        interpretation_cn += "\n###### 应用此策略的最佳市场条件：\n"
        interpretation_en += "- **Trending Markets**: Most effective in clear uptrends or downtrends where CMF and EMAs confirm the direction.\n"
        interpretation_cn += "- **趋势市场**：在 CMF 和 EMA 确认方向的明显上升或下降趋势中最为有效。\n"
        interpretation_en += "- **High Volume**: Ensure significant price movements are supported by high volume to validate CMF signals.\n"
        interpretation_cn += "- **高成交量**：确保重要的价格波动由高成交量支持，以验证 CMF 信号。\n"
        interpretation_en += "- **Avoid in Sideways/Noisy Markets**: CMF may produce false signals in choppy or non-trending markets.\n"
        interpretation_cn += "- **避免在横盘/嘈杂市场**：在波动剧烈或无趋势的市场中，CMF 可能产生虚假信号。\n"

        return interpretation_en, interpretation_cn

    interpret_en, interpret_cn = detailed_interpretation(
        bullish_crossovers, bearish_crossovers, confluences, current_price, trend
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

