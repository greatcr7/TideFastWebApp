from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from data.stock import get_stock_prices  # Ensure this module is available
from ta.trend import PSARIndicator, EMAIndicator
import pytz

# ---------------------------
# Parabolic SAR Analysis Function
# ---------------------------

def parabolic_sar_analysis(ticker):
    st.markdown(f"# 📈 抛物线转向指标 for {ticker.upper()}")
    
    # Sidebar for user inputs specific to Parabolic SAR Analysis
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
    
    # Step 2: User Inputs for Parabolic SAR Parameters
    st.sidebar.header("Parabolic SAR 参数")
    
    step = st.sidebar.number_input(
        "步长 (Step)", 
        min_value=0.001, 
        max_value=0.5, 
        value=0.02,  # Common default
        step=0.001,
        format="%.3f",
        help="Parabolic SAR 的步长。推荐值：0.02。 (The step size for Parabolic SAR. Recommended value: 0.02.)"
    )
    
    max_step = st.sidebar.number_input(
        "最大步长 (Maximum Step)", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.2,  # Common default
        step=0.01,
        format="%.2f",
        help="Parabolic SAR 的最大步长。推荐值：0.2。 (The maximum step size for Parabolic SAR. Recommended value: 0.2.)"
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
    show_psar = st.sidebar.checkbox("显示 Parabolic SAR (Show Parabolic SAR)", value=True)
    
    # Calculate Parabolic SAR using ta
    psar_indicator = PSARIndicator(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        step=step, 
        max_step=max_step
    )
    df['PSAR'] = psar_indicator.psar()
    
    # Calculate EMAs using ta
    ema_short_indicator = EMAIndicator(close=df['close'], window=ema_short_window)
    ema_long_indicator = EMAIndicator(close=df['close'], window=ema_long_window)
    df['EMA_Short'] = ema_short_indicator.ema_indicator()
    df['EMA_Long'] = ema_long_indicator.ema_indicator()
    
    # Identify Buy and Sell Signals based on PSAR flips
    def identify_psar_signals(df):
        buy_signals = []
        sell_signals = []
        previous_psar = df['PSAR'].iloc[0]
        previous_close = df['close'].iloc[0]
        
        for i in range(1, len(df)):
            current_psar = df['PSAR'].iloc[i]
            current_close = df['close'].iloc[i]
            
            # Buy Signal: PSAR flips below the price
            if (previous_psar > previous_close) and (current_psar < current_close):
                buy_signals.append({
                    'Date': df['date'].iloc[i],
                    'Price': current_close,
                    'PSAR': current_psar
                })
            # Sell Signal: PSAR flips above the price
            elif (previous_psar < previous_close) and (current_psar > current_close):
                sell_signals.append({
                    'Date': df['date'].iloc[i],
                    'Price': current_close,
                    'PSAR': current_psar
                })
            
            previous_psar = current_psar
            previous_close = current_close
        
        return buy_signals, sell_signals
    
    buy_signals, sell_signals = identify_psar_signals(df)
    
    # Identify Confluences
    def find_confluence(df, ema_short, ema_long, price_col='close'):
        latest_ema_short = df['EMA_Short'].iloc[-1]
        latest_ema_long = df['EMA_Long'].iloc[-1]
        latest_psar = df['PSAR'].iloc[-1]
        latest_price = df[price_col].iloc[-1]
        
        confluence_levels = {}
        
        # Bullish Confluence
        if (latest_price > latest_ema_short) and (latest_price > latest_ema_long) and (latest_psar < latest_price):
            confluence_levels['Bullish Confluence'] = {
                'EMA_Short': latest_ema_short,
                'EMA_Long': latest_ema_long,
                'PSAR': latest_psar
            }
        # Bearish Confluence
        elif (latest_price < latest_ema_short) and (latest_price < latest_ema_long) and (latest_psar > latest_price):
            confluence_levels['Bearish Confluence'] = {
                'EMA_Short': latest_ema_short,
                'EMA_Long': latest_ema_long,
                'PSAR': latest_psar
            }
        
        return confluence_levels, df
    
    confluences, df = find_confluence(df, ema_short_window, ema_long_window)
    
    # Determine Trend
    def determine_trend(df, confluences):
        latest_psar = df['PSAR'].iloc[-1]
        latest_ema_short = df['EMA_Short'].iloc[-1]
        latest_ema_long = df['EMA_Long'].iloc[-1]
        latest_price = df['close'].iloc[-1]
        
        if (latest_price > latest_ema_short) and (latest_price > latest_ema_long) and (latest_psar < latest_price):
            trend = "上升趋势 (Uptrend)"
        elif (latest_price < latest_ema_short) and (latest_price < latest_ema_long) and (latest_psar > latest_price):
            trend = "下降趋势 (Downtrend)"
        else:
            trend = "震荡区间 (Sideways)"
        
        return trend, latest_price
    
    trend, current_price = determine_trend(df, confluences)
    
    # Step 3: Plot Using Plotly
    def plot_psar(df, buy_signals, sell_signals, confluences, ticker, show_ema=True, show_psar=True):
        """
        Plot the Parabolic SAR along with price data and EMAs using Plotly.
        """
        fig = make_subplots(
            rows=1, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, 
            subplot_titles=(f'{ticker} 的股价和 Parabolic SAR (Price and Parabolic SAR)'),
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
        
        # Parabolic SAR
        if show_psar:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['PSAR'], 
                    mode='markers', 
                    marker=dict(color='orange', size=7, symbol='circle'),
                    name='Parabolic SAR'
                ), 
                row=1, col=1
            )
        
        # Buy Signals
        if buy_signals:
            crossover_dates_buy = [signal['Date'] for signal in buy_signals]
            crossover_prices_buy = [signal['Price'] for signal in buy_signals]
            fig.add_trace(
                go.Scatter(
                    mode='markers', 
                    x=crossover_dates_buy, 
                    y=crossover_prices_buy,
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    name='买入信号 (Buy Signal)'
                ), 
                row=1, col=1
            )
        
        # Sell Signals
        if sell_signals:
            crossover_dates_sell = [signal['Date'] for signal in sell_signals]
            crossover_prices_sell = [signal['Price'] for signal in sell_signals]
            fig.add_trace(
                go.Scatter(
                    mode='markers', 
                    x=crossover_dates_sell, 
                    y=crossover_prices_sell,
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    name='卖出信号 (Sell Signal)'
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
                y=value['PSAR'], 
                line=dict(color=color, dash='dash'), 
                row=1, col=1,
                annotation_text=annotation_text,
                annotation_position="top left"
            )
        
        fig.update_layout(
            title=f"{ticker} 的 Parabolic SAR 分析",
            yaxis_title='价格 (Price)',
            xaxis_title='日期 (Date)',
            template='plotly_dark',
            showlegend=True,
            height=800
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    fig = plot_psar(df, buy_signals, sell_signals, confluences, ticker, show_ema=show_ema, show_psar=show_psar)
    st.plotly_chart(fig, use_container_width=True)
    
    # Step 4: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation(buy_signals, sell_signals, confluences, current_price, trend, step, max_step):
        """
        Provide a detailed, actionable interpretation based on Parabolic SAR and crossovers in both English and Chinese.
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
                    interpretation_en += (f"- **Bullish Confluence**: EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) is above EMA{ema_long_window} "
                                           f"({indicators['EMA_Long']:.2f}), Parabolic SAR ({indicators['PSAR']:.2f}) is below the price.\n")
                    interpretation_cn += (f"- **看涨共振区**：EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) 高于 EMA{ema_long_window} "
                                           f"({indicators['EMA_Long']:.2f})，Parabolic SAR ({indicators['PSAR']:.2f}) 低于价格。\n")
                elif key == 'Bearish Confluence':
                    interpretation_en += (f"- **Bearish Confluence**: EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) is below EMA{ema_long_window} "
                                           f"({indicators['EMA_Long']:.2f}), Parabolic SAR ({indicators['PSAR']:.2f}) is above the price.\n")
                    interpretation_cn += (f"- **看跌共振区**：EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) 低于 EMA{ema_long_window} "
                                           f"({indicators['EMA_Long']:.2f})，Parabolic SAR ({indicators['PSAR']:.2f}) 高于价格。\n")
            interpretation_en += "\n"
            interpretation_cn += "\n"
        else:
            interpretation_en += "###### No Confluence Zones Detected.\n\n"
            interpretation_cn += "###### 未检测到共振区。\n\n"
        
        # 3. Parabolic SAR Analysis
        interpretation_en += f"###### Parabolic SAR Parameters:\n"
        interpretation_en += f"- **Step**: {step}\n"
        interpretation_en += f"- **Maximum Step**: {max_step}\n\n"
        
        interpretation_cn += f"###### Parabolic SAR 参数：\n"
        interpretation_cn += f"- **步长**：{step}\n"
        interpretation_cn += f"- **最大步长**：{max_step}\n\n"
        
        # 4. Price Position Analysis
        interpretation_en += "###### Price Position Relative to Parabolic SAR and EMAs:\n"
        interpretation_cn += "###### 当前价格相对于 Parabolic SAR 和 EMA 的位置：\n"
        if trend == "上升趋势 (Uptrend)":
            interpretation_en += f"- The current price is **above** EMA{ema_short_window} and EMA{ema_long_window}, and Parabolic SAR is **below** the price, indicating strong bullish momentum.\n"
            interpretation_cn += f"- 当前价格 **高于** EMA{ema_short_window} 和 EMA{ema_long_window}，且 Parabolic SAR **低于**价格，表明强劲的看涨动能。\n"
        elif trend == "下降趋势 (Downtrend)":
            interpretation_en += f"- The current price is **below** EMA{ema_short_window} and EMA{ema_long_window}, and Parabolic SAR is **above** the price, indicating strong bearish momentum.\n"
            interpretation_cn += f"- 当前价格 **低于** EMA{ema_short_window} 和 EMA{ema_long_window}，且 Parabolic SAR **高于**价格，表明强劲的卖出动能。\n"
        else:
            interpretation_en += f"- The current price is **between** EMA{ema_short_window} and EMA{ema_long_window}, and Parabolic SAR is **around** the price, indicating a consolidating or sideways market.\n"
            interpretation_cn += f"- 当前价格 **位于** EMA{ema_short_window} 和 EMA{ema_long_window} 之间，且 Parabolic SAR **接近** 价格，表明横盘或整合市场。\n"
        interpretation_en += "\n"
        interpretation_cn += "\n"
        
        # 5. Actionable Recommendations
        interpretation_en += "###### Actionable Recommendations:\n"
        interpretation_cn += "###### 可操作的建议：\n"
        
        # Bullish Confluence
        if 'Bullish Confluence' in confluences:
            interpretation_en += f"- **Buying Opportunity**: Consider buying when EMA{ema_short_window} is above EMA{ema_long_window}, and Parabolic SAR is below the price, confirming strong bullish momentum.\n"
            interpretation_cn += f"- **买入机会**：当 EMA{ema_short_window} 高于 EMA{ema_long_window}，且 Parabolic SAR 低于价格，确认强劲的看涨动能时，考虑买入。\n"
        
        # Bearish Confluence
        if 'Bearish Confluence' in confluences:
            interpretation_en += f"- **Selling Opportunity**: Consider selling when EMA{ema_short_window} is below EMA{ema_long_window}, and Parabolic SAR is above the price, confirming strong bearish momentum.\n"
            interpretation_cn += f"- **卖出机会**：当 EMA{ema_short_window} 低于 EMA{ema_long_window}，且 Parabolic SAR 高于价格，确认强劲的卖出动能时，考虑卖出。\n"
        
        # Bullish Signals
        if buy_signals:
            interpretation_en += "\n- **Bullish Signal Detected**: Parabolic SAR has flipped below the price, indicating a potential upward trend. Consider entering a long position when confirmed by bullish candlestick patterns.\n"
            interpretation_cn += "\n- **检测到买入信号**：Parabolic SAR 已经下穿价格，表明可能出现上升趋势。当通过看涨的烛台形态确认时，考虑买入。\n"
        
        # Bearish Signals
        if sell_signals:
            interpretation_en += "\n- **Bearish Signal Detected**: Parabolic SAR has flipped above the price, indicating a potential downward trend. Consider entering a short position when confirmed by bearish candlestick patterns.\n"
            interpretation_cn += "\n- **检测到卖出信号**：Parabolic SAR 已经上穿价格，表明可能出现下降趋势。当通过看跌的烛台形态确认时，考虑卖出。\n"
        
        # Confluence Zones
        if confluences:
            interpretation_en += "\n- **Confluence Zones**: Trades near these areas have a higher probability of success due to the alignment of Parabolic SAR with EMAs.\n"
            interpretation_cn += "\n- **共振区**：由于 Parabolic SAR 与 EMA 对齐，接近这些区域的交易成功概率更高。\n"
        
        # Risk Management
        interpretation_en += "\n###### Risk Management:\n"
        interpretation_cn += "\n###### 风险管理：\n"
        interpretation_en += f"- **Stop-Loss**: Place stop-loss orders just beyond EMA{ema_short_window} or EMA{ema_long_window} to manage risk.\n"
        interpretation_cn += f"- **止损**：在 EMA{ema_short_window} 或 EMA{ema_long_window} 之外稍微放置止损订单以管理风险。\n"
        interpretation_en += f"- **Take-Profit**: Set target levels based on recent support/resistance levels or use a trailing stop to lock in profits.\n"
        interpretation_cn += f"- **止盈**：根据近期的支撑/阻力位设置目标水平或使用移动止盈以锁定利润。\n"
        
        # Market Conditions
        interpretation_en += "\n###### Optimal Market Conditions for Applying This Strategy:\n"
        interpretation_cn += "\n###### 应用此策略的最佳市场条件：\n"
        interpretation_en += "- **Trending Markets**: Most effective in clear uptrends or downtrends where Parabolic SAR and EMAs confirm the direction.\n"
        interpretation_cn += "- **趋势市场**：在 Parabolic SAR 和 EMA 确认方向的明显上升或下降趋势中最为有效。\n"
        interpretation_en += "- **High Volume**: Ensure significant price movements are supported by high volume to validate Parabolic SAR signals.\n"
        interpretation_cn += "- **高成交量**：确保重要的价格波动由高成交量支持，以验证 Parabolic SAR 信号。\n"
        interpretation_en += "- **Avoid in Sideways/Noisy Markets**: Parabolic SAR may produce false signals in choppy or non-trending markets.\n"
        interpretation_cn += "- **避免在横盘/嘈杂市场**：在波动剧烈或无趋势的市场中，Parabolic SAR 可能产生虚假信号。\n"
        
        return interpretation_en, interpretation_cn
    
    interpret_en, interpret_cn = detailed_interpretation(
        buy_signals, sell_signals, confluences, current_price, trend, step, max_step
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
        parabolic_sar_analysis(ticker.upper())

if __name__ == "__main__":
    main()