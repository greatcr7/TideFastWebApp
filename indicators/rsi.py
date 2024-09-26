from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from scipy.signal import find_peaks

from data.stock import get_stock_prices

# ---------------------------
# RSI Analysis Demo Function
# ---------------------------

def rsi_analysis():
    st.markdown(f"# 📈 RSI")

    # Sidebar for user inputs specific to RSI Analysis
    st.sidebar.header("指标参数")

    # Function to convert period to start and end dates
    def convert_period_to_dates(period):
        end_date = datetime.now()
        
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
        
        # Convert to 'yyyymmdd' format
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    # User input function
    def user_input_features():
        ticker = st.sidebar.text_input("股票代码/名称 (e.g. 特斯拉, 600519)", value="600519.SH")
        period = st.sidebar.selectbox("时间跨度", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=3)
        # interval = st.sidebar.selectbox("Interval", options=["1d", "1wk", "1mo"], index=1)
        

        # Convert period to start and end dates
        start_date, end_date = convert_period_to_dates(period)
        
        return ticker.upper(), start_date, end_date

    # Getting user input
    ticker, start_date, end_date = user_input_features()


    # Step 1: Fetch Historical Data using custom get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("No data fetched. Please check the ticker symbol and try again.")
        st.stop()

    # Step 2: Calculate Relative Strength Index (RSI)
    def calculate_rsi(df, window=14):
        """
        Calculate Relative Strength Index (RSI) using the ta library.
        """
        rsi_indicator = RSIIndicator(close=df['close'], window=window)
        df['RSI'] = rsi_indicator.rsi()
        return df

    df = calculate_rsi(df, window=14)

    # Step 3: Identify Price Divergence
    def identify_divergence(df, window=5, rsi_col='RSI', price_col='close'):
        """
        Identify bullish and bearish divergences between price and RSI.
        """
        bullish_divergences = []
        bearish_divergences = []

        # Find peaks and troughs in price
        price_peaks, _ = find_peaks(df[price_col], distance=window, prominence=1)
        price_troughs, _ = find_peaks(-df[price_col], distance=window, prominence=1)

        # Find peaks and troughs in RSI
        rsi_peaks, _ = find_peaks(df[rsi_col], distance=window, prominence=1)
        rsi_troughs, _ = find_peaks(-df[rsi_col], distance=window, prominence=1)

        # Bullish Divergence: Price makes lower low, RSI makes higher low
        for i in range(1, len(price_troughs)):
            price_idx_prev = price_troughs[i-1]
            price_idx_curr = price_troughs[i]

            # Price makes a lower low
            if df[price_col].iloc[price_idx_curr] < df[price_col].iloc[price_idx_prev]:
                # Find RSI troughs between these price troughs
                rsi_troughs_in_range = [idx for idx in rsi_troughs if price_idx_prev <= idx <= price_idx_curr]
                if len(rsi_troughs_in_range) >= 2:
                    rsi_idx_prev = rsi_troughs_in_range[0]
                    rsi_idx_curr = rsi_troughs_in_range[-1]
                    # RSI makes a higher low
                    if df[rsi_col].iloc[rsi_idx_curr] > df[rsi_col].iloc[rsi_idx_prev]:
                        bullish_divergences.append({
                            'Date': df['date'].iloc[price_idx_curr],
                            'Price': df[price_col].iloc[price_idx_curr],
                            'RSI': df[rsi_col].iloc[rsi_idx_curr]
                        })

        # Bearish Divergence: Price makes higher high, RSI makes lower high
        for i in range(1, len(price_peaks)):
            price_idx_prev = price_peaks[i-1]
            price_idx_curr = price_peaks[i]

            # Price makes a higher high
            if df[price_col].iloc[price_idx_curr] > df[price_col].iloc[price_idx_prev]:
                # Find RSI peaks between these price peaks
                rsi_peaks_in_range = [idx for idx in rsi_peaks if price_idx_prev <= idx <= price_idx_curr]
                if len(rsi_peaks_in_range) >= 2:
                    rsi_idx_prev = rsi_peaks_in_range[0]
                    rsi_idx_curr = rsi_peaks_in_range[-1]
                    # RSI makes a lower high
                    if df[rsi_col].iloc[rsi_idx_curr] < df[rsi_col].iloc[rsi_idx_prev]:
                        bearish_divergences.append({
                            'Date': df['date'].iloc[price_idx_curr],
                            'Price': df[price_col].iloc[price_idx_curr],
                            'RSI': df[rsi_col].iloc[rsi_idx_curr]
                        })

        return bullish_divergences, bearish_divergences

    bullish_divergences, bearish_divergences = identify_divergence(df)

    # Step 4: Identify Confluence with Exponential Moving Averages (EMA)
    def find_confluence(df, rsi_threshold=50):
        """
        Identify if RSI aligns with other moving averages.
        """
        # Calculate EMAs
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()

        latest_rsi = df['RSI'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]
        latest_price = df['close'].iloc[-1]

        confluence_levels = {}

        # Define confluence based on RSI thresholds and EMA alignment
        if latest_rsi > rsi_threshold and latest_price > latest_ema50 and latest_price > latest_ema200:
            confluence_levels['Bullish Confluence'] = {
                'RSI': latest_rsi,
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }
        elif latest_rsi < rsi_threshold and latest_price < latest_ema50 and latest_price < latest_ema200:
            confluence_levels['Bearish Confluence'] = {
                'RSI': latest_rsi,
                'EMA50': latest_ema50,
                'EMA200': latest_ema200
            }

        return confluence_levels, df

    confluences, df = find_confluence(df, rsi_threshold=50)

    # Step 5: Determine Market Trend Based on RSI and EMAs
    def determine_trend(df, confluences):
        """
        Determine the current market trend based on RSI and EMAs.
        """
        latest_rsi = df['RSI'].iloc[-1]
        latest_ema50 = df['EMA50'].iloc[-1]
        latest_ema200 = df['EMA200'].iloc[-1]
        latest_price = df['close'].iloc[-1]

        if latest_rsi > 50 and latest_price > latest_ema50 and latest_price > latest_ema200:
            trend = "上升趋势 (Uptrend)"
        elif latest_rsi < 50 and latest_price < latest_ema50 and latest_price < latest_ema200:
            trend = "下降趋势 (Downtrend)"
        else:
            trend = "震荡区间 (Sideways)"

        return trend, latest_price

    trend, current_price = determine_trend(df, confluences)

    # Step 6: Plot Using Plotly
    def plot_rsi(df, bullish_divergences, bearish_divergences, confluences, ticker):
        """
        Plot the RSI along with price data, EMAs, and divergences using Plotly.
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, subplot_titles=(f'{ticker}的股价和价格均线', '相对强弱指数 (RSI)'),
                            row_width=[0.2, 0.7])

        # Candlestick for Price
        fig.add_trace(go.Candlestick(x=df['date'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='Price'), row=1, col=1)

        # EMAs
        fig.add_trace(go.Scatter(x=df['date'], y=df['EMA50'], line=dict(color='blue', width=1), name='EMA50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['EMA200'], line=dict(color='purple', width=1), name='EMA200'), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], line=dict(color='orange', width=1), name='RSI'), row=2, col=1)

        # Overbought and Oversold lines
        fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=2, col=1)
        fig.add_hline(y=50, line=dict(color='gray', dash='dash'), row=2, col=1)

        # Highlight Bullish Divergences
        for div in bullish_divergences:
            fig.add_annotation(x=div['Date'], y=div['Price'],
                               text="Bullish Div.",
                               showarrow=True,
                               arrowhead=1,
                               ax=0, ay=-40,
                               arrowcolor='green',
                               row=1, col=1)
            fig.add_annotation(x=div['Date'], y=div['RSI'],
                               text="Bullish Div.",
                               showarrow=True,
                               arrowhead=1,
                               ax=0, ay=-40,
                               arrowcolor='green',
                               row=2, col=1)

        # Highlight Bearish Divergences
        for div in bearish_divergences:
            fig.add_annotation(x=div['Date'], y=div['Price'],
                               text="Bearish Div.",
                               showarrow=True,
                               arrowhead=1,
                               ax=0, ay=40,
                               arrowcolor='red',
                               row=1, col=1)
            fig.add_annotation(x=div['Date'], y=div['RSI'],
                               text="Bearish Div.",
                               showarrow=True,
                               arrowhead=1,
                               ax=0, ay=40,
                               arrowcolor='red',
                               row=2, col=1)

        # Highlight Confluence Zones
        for key, value in confluences.items():
            if key == 'Bullish Confluence':
                color = 'green'
            elif key == 'Bearish Confluence':
                color = 'red'
            else:
                color = 'yellow'
            fig.add_hline(y=50, line=dict(color=color, dash='dot'), row=2, col=1)

        fig.update_layout(title=f'相对强弱指数 (RSI)',
                          yaxis_title='Price',
                          xaxis_title='',
                          template='plotly_dark',
                          showlegend=True,
                          height=800)

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    fig = plot_rsi(df, bullish_divergences, bearish_divergences, confluences, ticker)
    st.plotly_chart(fig, use_container_width=True)

    # Step 7: Detailed Actionable Interpretation in Both English and Chinese
    def detailed_interpretation(bullish_divergences, bearish_divergences, confluences, current_price, trend):
        """
        Provide a detailed, actionable interpretation based on RSI and divergences in both English and Chinese.
        """
        interpretation_en = ""
        interpretation_cn = ""

        # 1. Trend Analysis
        interpretation_en += f"## Current Market Trend: {trend}\n\n"
        interpretation_en += f"**Current Price**: {current_price:.2f}\n\n"

        interpretation_cn += f"## 当前市场趋势：{trend}\n\n"
        interpretation_cn += f"**当前价格**：{current_price:.2f}\n\n"

        # 2. Confluence Analysis
        if confluences:
            interpretation_en += "### Confluence Zones Detected:\n"
            interpretation_cn += "### 检测到的共振区：\n"
            for key, indicators in confluences.items():
                if key == 'Bullish Confluence':
                    interpretation_en += f"- **Bullish Confluence**: RSI is above 50 ({indicators['RSI']:.2f}), and the price is above both EMA50 ({indicators['EMA50']:.2f}) and EMA200 ({indicators['EMA200']:.2f}).\n"
                    interpretation_cn += f"- **看涨共振区**：RSI 高于50 ({indicators['RSI']:.2f})，价格高于 EMA50 ({indicators['EMA50']:.2f}) 和 EMA200 ({indicators['EMA200']:.2f})。\n"
                elif key == 'Bearish Confluence':
                    interpretation_en += f"- **Bearish Confluence**: RSI is below 50 ({indicators['RSI']:.2f}), and the price is below both EMA50 ({indicators['EMA50']:.2f}) and EMA200 ({indicators['EMA200']:.2f}).\n"
                    interpretation_cn += f"- **看跌共振区**：RSI 低于50 ({indicators['RSI']:.2f})，价格低于 EMA50 ({indicators['EMA50']:.2f}) 和 EMA200 ({indicators['EMA200']:.2f})。\n"
            interpretation_en += "\n"
            interpretation_cn += "\n"
        else:
            interpretation_en += "### No Confluence Zones Detected.\n\n"
            interpretation_cn += "### 未检测到共振区。\n\n"

        # 3. Price Position Analysis
        interpretation_en += "### Price Position Relative to RSI and EMAs:\n"
        interpretation_cn += "### 当前价格相对于 RSI 和 EMA 的位置：\n"
        if trend == "Uptrend":
            interpretation_en += "- The current price is **above** EMA50 and EMA200, with RSI above 50, indicating strong buying pressure.\n"
            interpretation_cn += "- 当前价格 **高于** EMA50 和 EMA200，且 RSI 高于50，表明强劲的买入压力。\n"
        elif trend == "Downtrend":
            interpretation_en += "- The current price is **below** EMA50 and EMA200, with RSI below 50, indicating strong selling pressure.\n"
            interpretation_cn += "- 当前价格 **低于** EMA50 和 EMA200，且 RSI 低于50，表明强劲的卖出压力。\n"
        else:
            interpretation_en += "- The current price is **between** EMA50 and EMA200, with RSI around 50, indicating a sideways or consolidating market.\n"
            interpretation_cn += "- 当前价格 **位于** EMA50 和 EMA200 之间，且 RSI 约为50，表明横盘或整合市场。\n"
        interpretation_en += "\n"
        interpretation_cn += "\n"

        # 4. Actionable Recommendations
        interpretation_en += "### Actionable Recommendations:\n"
        interpretation_cn += "### 可操作的建议：\n"

        # Bullish Confluence
        if 'Bullish Confluence' in confluences:
            interpretation_en += "- **Buying Opportunity**: Consider buying when RSI remains above 50 and the price is above EMA50 and EMA200, confirming strong bullish momentum.\n"
            interpretation_cn += "- **买入机会**：当 RSI 保持在50以上，且价格高于 EMA50 和 EMA200，确认强劲的看涨动能时，考虑买入。\n"

        # Bearish Confluence
        if 'Bearish Confluence' in confluences:
            interpretation_en += "- **Selling Opportunity**: Consider selling when RSI remains below 50 and the price is below EMA50 and EMA200, confirming strong bearish momentum.\n"
            interpretation_cn += "- **卖出机会**：当 RSI 保持在50以下，且价格低于 EMA50 和 EMA200，确认强劲的卖出动能时，考虑卖出。\n"

        # Bullish Divergence
        if bullish_divergences:
            interpretation_en += "\n- **Bullish Divergence Detected**: Indicates potential reversal to the upside. Consider entering a long position when price confirms the reversal with bullish candlestick patterns.\n"
            interpretation_cn += "\n- **检测到看涨背离**：表明可能出现向上的反转。当价格通过看涨的烛台形态确认反转时，考虑买入。\n"

        # Bearish Divergence
        if bearish_divergences:
            interpretation_en += "\n- **Bearish Divergence Detected**: Indicates potential reversal to the downside. Consider entering a short position when price confirms the reversal with bearish candlestick patterns.\n"
            interpretation_cn += "\n- **检测到看跌背离**：表明可能出现向下的反转。当价格通过看跌的烛台形态确认反转时，考虑卖出。\n"

        # Confluence Zones
        if confluences:
            interpretation_en += "\n- **Confluence Zones**: Trades near these areas have a higher probability of success due to the alignment of RSI with EMAs.\n"
            interpretation_cn += "\n- **共振区**：由于 RSI 与 EMA 对齐，接近这些区域的交易成功概率更高。\n"

        # Breakout Scenarios
        interpretation_en += "\n### Breakout Scenarios:\n"
        interpretation_cn += "\n### 突破情景：\n"
        interpretation_en += "- **Bullish Breakout**: If the price breaks above EMA200 with increasing RSI and volume, consider **entering a long position**.\n"
        interpretation_cn += "- **看涨突破**：如果价格在 RSI 和成交量增加的情况下突破 EMA200，考虑 **建立多头仓位**。\n"
        interpretation_en += "- **Bearish Breakout**: If the price breaks below EMA200 with decreasing RSI and volume, consider **entering a short position**.\n"
        interpretation_cn += "- **看跌突破**：如果价格在 RSI 和成交量减少的情况下突破 EMA200，考虑 **建立空头仓位**。\n"

        # Risk Management
        interpretation_en += "\n### Risk Management:\n"
        interpretation_cn += "\n### 风险管理：\n"
        interpretation_en += "- **Stop-Loss**: Place stop-loss orders just beyond EMA50 or EMA200 to manage risk.\n"
        interpretation_cn += "- **止损**：在 EMA50 或 EMA200 之外稍微放置止损订单以管理风险。\n"
        interpretation_en += "- **Take-Profit**: Set target levels based on recent support/resistance levels or use a trailing stop to lock in profits.\n"
        interpretation_cn += "- **止盈**：根据近期的支撑/阻力位设置目标水平或使用移动止盈以锁定利润。\n"

        # Market Conditions
        interpretation_en += "\n### Optimal Market Conditions for Applying This Strategy:\n"
        interpretation_cn += "\n### 应用此策略的最佳市场条件：\n"
        interpretation_en += "- **Trending Markets**: Most effective in clear uptrends or downtrends where RSI and EMAs confirm the direction.\n"
        interpretation_cn += "- **趋势市场**：在 RSI 和 EMA 确认方向的明显上升或下降趋势中最为有效。\n"
        interpretation_en += "- **High Volume**: Ensure significant price movements are supported by high volume to validate RSI signals.\n"
        interpretation_cn += "- **高成交量**：确保重要的价格波动由高成交量支持，以验证 RSI 信号。\n"
        interpretation_en += "- **Avoid in Sideways/Noisy Markets**: RSI may produce false signals in choppy or non-trending markets.\n"
        interpretation_cn += "- **避免在横盘/嘈杂市场**：在波动剧烈或无趋势的市场中，RSI 可能产生虚假信号。\n"

        return interpretation_en, interpretation_cn

    interpret_en, interpret_cn = detailed_interpretation(
        bullish_divergences, bearish_divergences, confluences, current_price, trend
    )

    # Display Interpretations
    st.markdown("### 📄 指标解读")

    # Tabs for English and Chinese
    tab1, tab2 = st.tabs(["中文", "English"])

    with tab1:
        st.markdown(interpret_cn)

    with tab2:
        st.markdown(interpret_en)

    # Optional: Display Data Table
    with st.expander("📊 查看原始数据"):
        st.dataframe(df)

    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ by TideFast")
