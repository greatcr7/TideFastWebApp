from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import pandas as pd
from data.stock import get_stock_prices
import pandas as pd
from ta.momentum import KAMAIndicator
from ta.trend import EMAIndicator
import pytz

# ---------------------------
# KAMA Analysis Function
# ---------------------------

def kama_analysis(ticker):
    st.markdown(f"# 📈 KAMA均线")
    
    # Sidebar for user inputs specific to KAMA Analysis
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
        st.error("No data fetched. Please check the ticker symbol and try again.")
        st.stop()
    
    # Step 2: User Inputs for KAMA Parameters
    st.sidebar.header("KAMA 参数")
    
    kama_period = st.sidebar.number_input(
        "效率周期 (Efficiency Ratio Period)", 
        min_value=1, 
        max_value=50, 
        value=10,  # Best practice default
        step=1,
        help="效率比率的计算周期。较短的周期使KAMA对价格变动更敏感。推荐值：10。"
    )
    
    kama_fast = st.sidebar.number_input(
        "快速常数 (Fast SC)", 
        min_value=1, 
        max_value=10, 
        value=2,  # Best practice default
        step=1,
        help="快速平滑常数，用于调整KAMA的敏感度。较低的值使KAMA更敏感。推荐值：2。"
    )
    
    kama_slow = st.sidebar.number_input(
        "慢速常数 (Slow SC)", 
        min_value=10, 
        max_value=100, 
        value=30,  # Best practice default
        step=1,
        help="慢速平滑常数，用于调整KAMA的平滑程度。较高的值使KAMA更平滑。推荐值：30。"
    )
    
    # Additional Parameters
    st.sidebar.header("其他参数 (Additional Parameters)")
    
    ema_short_window = st.sidebar.number_input(
        "EMA 短期窗口 (EMA Short Window)", 
        min_value=10, 
        max_value=100, 
        value=50,  # Best practice default
        step=5,
        help="短期指数移动平均线（EMA）的窗口大小。较短的窗口使EMA更敏感。推荐值：50。"
    )
    
    ema_long_window = st.sidebar.number_input(
        "EMA 长期窗口 (EMA Long Window)", 
        min_value=100, 
        max_value=300, 
        value=200,  # Best practice default
        step=10,
        help="长期指数移动平均线（EMA）的窗口大小。较长的窗口使EMA更平滑。推荐值：200。"
    )
    
    crossover_window = st.sidebar.number_input(
        "交叉检测窗口 (Crossover Window)", 
        min_value=1, 
        max_value=10, 
        value=1,  # Best practice default
        step=1,
        help="定义检测交叉的最小天数，以避免虚假信号。推荐值：1。"
    )
        
    # Plotting Options
    st.sidebar.header("绘图选项 (Plotting Options)")
    show_ema = st.sidebar.checkbox("显示 EMA (Show EMAs)", value=True)
    show_kama = st.sidebar.checkbox("显示 KAMA (Show KAMA)", value=True)
    

    # Parameters
    kama_period = 10
    kama_fast = 2
    kama_slow = 30
    ema_short_window = 12
    ema_long_window = 26
    crossover_window = 1

    # Calculate KAMA using ta
    kama_indicator = KAMAIndicator(close=df['close'], window=kama_period, pow1=kama_fast, pow2=kama_slow)
    df['KAMA'] = kama_indicator.kama()

    # Calculate EMAs using ta
    ema_short_indicator = EMAIndicator(close=df['close'], window=ema_short_window)
    ema_long_indicator = EMAIndicator(close=df['close'], window=ema_long_window)
    df['EMA_Short'] = ema_short_indicator.ema_indicator()
    df['EMA_Long'] = ema_long_indicator.ema_indicator()

    # Identify Crossovers
    def identify_crossovers(df, price_col='close', kama_col='KAMA', window=1):
        bullish_crossovers = []
        bearish_crossovers = []
        
        for i in range(window, len(df)):
            if (df[kama_col].iloc[i] > df[price_col].iloc[i]) and (df[kama_col].iloc[i - window] <= df[price_col].iloc[i - window]):
                bullish_crossovers.append({
                    'Date': df['date'].iloc[i],
                    'Price': df[price_col].iloc[i],
                    'KAMA': df[kama_col].iloc[i]
                })
            elif (df[kama_col].iloc[i] < df[price_col].iloc[i]) and (df[kama_col].iloc[i - window] >= df[price_col].iloc[i - window]):
                bearish_crossovers.append({
                    'Date': df['date'].iloc[i],
                    'Price': df[price_col].iloc[i],
                    'KAMA': df[kama_col].iloc[i]
                })
        
        return bullish_crossovers, bearish_crossovers

    bullish_crossovers, bearish_crossovers = identify_crossovers(df, window=crossover_window)

    # Identify Confluences
    def find_confluence(df, ema_short, ema_long, kama_col='KAMA', price_col='close'):
        latest_kama = df[kama_col].iloc[-1]
        latest_ema_short = df['EMA_Short'].iloc[-1]
        latest_ema_long = df['EMA_Long'].iloc[-1]
        latest_price = df[price_col].iloc[-1]
        
        confluence_levels = {}
        
        if (latest_price > latest_ema_short) and (latest_price > latest_ema_long) and (latest_kama > latest_ema_short):
            confluence_levels['Bullish Confluence'] = {
                'KAMA': latest_kama,
                'EMA_Short': latest_ema_short,
                'EMA_Long': latest_ema_long
            }
        elif (latest_price < latest_ema_short) and (latest_price < latest_ema_long) and (latest_kama < latest_ema_short):
            confluence_levels['Bearish Confluence'] = {
                'KAMA': latest_kama,
                'EMA_Short': latest_ema_short,
                'EMA_Long': latest_ema_long
            }
        
        return confluence_levels, df

    confluences, df = find_confluence(df, ema_short_window, ema_long_window)

    # Determine Trend
    def determine_trend(df, confluences):
        latest_kama = df['KAMA'].iloc[-1]
        latest_ema_short = df['EMA_Short'].iloc[-1]
        latest_ema_long = df['EMA_Long'].iloc[-1]
        latest_price = df['close'].iloc[-1]
        
        if (latest_price > latest_ema_short) and (latest_price > latest_ema_long) and (latest_kama > latest_ema_short):
            trend = "上升趋势 (Uptrend)"
        elif (latest_price < latest_ema_short) and (latest_price < latest_ema_long) and (latest_kama < latest_ema_short):
            trend = "下降趋势 (Downtrend)"
        else:
            trend = "震荡区间 (Sideways)"
        
        return trend, latest_price

    trend, current_price = determine_trend(df, confluences)
 
    # Step 8: Plot Using Plotly
    def plot_kama(df, bullish_crossovers, bearish_crossovers, confluences, ticker, show_ema=True, show_kama=True):
        """
        Plot the KAMA along with price data, EMAs, and crossovers using Plotly.
        """
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, 
            subplot_titles=(f'{ticker}的股价和移动平均线', 'KAMA'),
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
        
        # KAMA with increased line width for better visibility
        if show_kama:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['KAMA'], 
                    line=dict(color='black', width=3),  # Increased width from 1 to 3
                    name='KAMA'
                ), 
                row=1, col=1
            )
        
        # Crossovers Markers
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
                name='Bullish Crossover'
            ), 
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                mode='markers', 
                x=crossover_dates_bear, 
                y=crossover_prices_bear,
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Bearish Crossover'
            ), 
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
            fig.add_hline(
                y=value['KAMA'], 
                line=dict(color=color, dash='dot'), 
                row=1, col=1
            )
        
        fig.update_layout(
            title="Kaufman's Adaptive Moving Average (KAMA)",
            yaxis_title='Price',
            xaxis_title='',
            template='plotly_dark',
            showlegend=True,
            height=800
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig

    
    fig = plot_kama(df, bullish_crossovers, bearish_crossovers, confluences, ticker, show_ema=show_ema, show_kama=show_kama)
    st.plotly_chart(fig, use_container_width=True)
    
    # Step 9: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation(bullish_crossovers, bearish_crossovers, confluences, current_price, trend):
        """
        Provide a detailed, actionable interpretation based on KAMA and crossovers in both English and Chinese.
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
                    interpretation_en += f"- **Bullish Confluence**: KAMA is above EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) and EMA{ema_long_window} ({indicators['EMA_Long']:.2f}), and the price is above both EMAs.\n"
                    interpretation_cn += f"- **看涨共振区**：KAMA 高于 EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) 和 EMA{ema_long_window} ({indicators['EMA_Long']:.2f})，且价格高于这两条均线。\n"
                elif key == 'Bearish Confluence':
                    interpretation_en += f"- **Bearish Confluence**: KAMA is below EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) and EMA{ema_long_window} ({indicators['EMA_Long']:.2f}), and the price is below both EMAs.\n"
                    interpretation_cn += f"- **看跌共振区**：KAMA 低于 EMA{ema_short_window} ({indicators['EMA_Short']:.2f}) 和 EMA{ema_long_window} ({indicators['EMA_Long']:.2f})，且价格低于这两条均线。\n"
            interpretation_en += "\n"
            interpretation_cn += "\n"
        else:
            interpretation_en += "###### No Confluence Zones Detected.\n\n"
            interpretation_cn += "###### 未检测到共振区。\n\n"
        
        # 3. Price Position Analysis
        interpretation_en += "###### Price Position Relative to KAMA and EMAs:\n"
        interpretation_cn += "###### 当前价格相对于 KAMA 和 EMA 的位置：\n"
        if trend == "上升趋势 (Uptrend)":
            interpretation_en += "- The current price is **above** EMA50 and EMA200, with KAMA above EMA50, indicating strong buying pressure.\n"
            interpretation_cn += "- 当前价格 **高于** EMA50 和 EMA200，且 KAMA 高于 EMA50，表明强劲的买入压力。\n"
        elif trend == "下降趋势 (Downtrend)":
            interpretation_en += "- The current price is **below** EMA50 and EMA200, with KAMA below EMA50, indicating strong selling pressure.\n"
            interpretation_cn += "- 当前价格 **低于** EMA50 和 EMA200，且 KAMA 低于 EMA50，表明强劲的卖出压力。\n"
        else:
            interpretation_en += "- The current price is **between** EMA50 and EMA200, with KAMA around EMA50, indicating a sideways or consolidating market.\n"
            interpretation_cn += "- 当前价格 **位于** EMA50 和 EMA200 之间，且 KAMA 约为 EMA50，表明横盘或整合市场。\n"
        interpretation_en += "\n"
        interpretation_cn += "\n"
        
        # 4. Actionable Recommendations
        interpretation_en += "###### Actionable Recommendations:\n"
        interpretation_cn += "###### 可操作的建议：\n"
        
        # Bullish Confluence
        if 'Bullish Confluence' in confluences:
            interpretation_en += "- **Buying Opportunity**: Consider buying when KAMA is above EMA50 and EMA200, and the price is above both EMAs, confirming strong bullish momentum.\n"
            interpretation_cn += "- **买入机会**：当 KAMA 高于 EMA50 和 EMA200，且价格高于这两条均线，确认强劲的看涨动能时，考虑买入。\n"
        
        # Bearish Confluence
        if 'Bearish Confluence' in confluences:
            interpretation_en += "- **Selling Opportunity**: Consider selling when KAMA is below EMA50 and EMA200, and the price is below both EMAs, confirming strong bearish momentum.\n"
            interpretation_cn += "- **卖出机会**：当 KAMA 低于 EMA50 和 EMA200，且价格低于这两条均线，确认强劲的卖出动能时，考虑卖出。\n"
        
        # Bullish Crossovers
        if bullish_crossovers:
            interpretation_en += "\n- **Bullish Crossover Detected**: Indicates a potential upward trend. Consider entering a long position when confirmed by bullish candlestick patterns.\n"
            interpretation_cn += "\n- **检测到看涨交叉**：表明可能出现上升趋势。当通过看涨的烛台形态确认时，考虑买入。\n"
        
        # Bearish Crossovers
        if bearish_crossovers:
            interpretation_en += "\n- **Bearish Crossover Detected**: Indicates a potential downward trend. Consider entering a short position when confirmed by bearish candlestick patterns.\n"
            interpretation_cn += "\n- **检测到看跌交叉**：表明可能出现下降趋势。当通过看跌的烛台形态确认时，考虑卖出。\n"
        
        # Confluence Zones
        if confluences:
            interpretation_en += "\n- **Confluence Zones**: Trades near these areas have a higher probability of success due to the alignment of KAMA with EMAs.\n"
            interpretation_cn += "\n- **共振区**：由于 KAMA 与 EMA 对齐，接近这些区域的交易成功概率更高。\n"
        
        # Risk Management
        interpretation_en += "\n###### Risk Management:\n"
        interpretation_cn += "\n###### 风险管理：\n"
        interpretation_en += "- **Stop-Loss**: Place stop-loss orders just beyond EMA50 or EMA200 to manage risk.\n"
        interpretation_cn += "- **止损**：在 EMA50 或 EMA200 之外稍微放置止损订单以管理风险。\n"
        interpretation_en += "- **Take-Profit**: Set target levels based on recent support/resistance levels or use a trailing stop to lock in profits.\n"
        interpretation_cn += "- **止盈**：根据近期的支撑/阻力位设置目标水平或使用移动止盈以锁定利润。\n"
        
        # Market Conditions
        interpretation_en += "\n###### Optimal Market Conditions for Applying This Strategy:\n"
        interpretation_cn += "\n###### 应用此策略的最佳市场条件：\n"
        interpretation_en += "- **Trending Markets**: Most effective in clear uptrends or downtrends where KAMA and EMAs confirm the direction.\n"
        interpretation_cn += "- **趋势市场**：在 KAMA 和 EMA 确认方向的明显上升或下降趋势中最为有效。\n"
        interpretation_en += "- **High Volume**: Ensure significant price movements are supported by high volume to validate KAMA signals.\n"
        interpretation_cn += "- **高成交量**：确保重要的价格波动由高成交量支持，以验证 KAMA 信号。\n"
        interpretation_en += "- **Avoid in Sideways/Noisy Markets**: KAMA may produce false signals in choppy or non-trending markets.\n"
        interpretation_cn += "- **避免在横盘/嘈杂市场**：在波动剧烈或无趋势的市场中，KAMA 可能产生虚假信号。\n"
        
        return interpretation_en, interpretation_cn
    
    interpret_en, interpret_cn = detailed_interpretation(
        bullish_crossovers, bearish_crossovers, confluences, current_price, trend
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
        kama_analysis(ticker.upper())

if __name__ == "__main__":
    main()