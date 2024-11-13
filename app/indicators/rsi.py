from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from itertools import product
from data.stock import get_stock_prices  # Ensure this custom module is available
from ta.momentum import RSIIndicator
import pytz

# ---------------------------
# RSI Analysis Function with Enhanced Features
# ---------------------------

def rsi_analysis(ticker):
    st.markdown(f"# 📈 Relative Strength Index (RSI) Analysis - {ticker.upper()}")
    
    # Sidebar for user inputs specific to RSI Analysis
    st.sidebar.header("📊 参数设置")
    
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

    # User input function with additional RSI parameters
    def user_input_features():
        period = st.sidebar.selectbox(
            "📅 时间跨度 (Time Period)",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            help="选择分析的时间跨度。"
        )
        rsi_period = st.sidebar.number_input(
            "🔢 RSI 周期 (RSI Period)",
            min_value=1,
            max_value=50,
            value=14,
            help="RSI的计算周期。较短的周期使RSI对价格变动更敏感。推荐值：14。"
        )
        overbought = st.sidebar.number_input(
            "🔢 超买阈值 (Overbought Threshold)",
            min_value=50,
            max_value=100,
            value=70,
            help="RSI超过此阈值时视为超买。推荐值：70。"
        )
        oversold = st.sidebar.number_input(
            "🔢 超卖阈值 (Oversold Threshold)",
            min_value=0,
            max_value=50,
            value=30,
            help="RSI低于此阈值时视为超卖。推荐值：30。"
        )
        smoothing = st.sidebar.number_input(
            "🔢 平滑参数 (Smoothing Parameter)",
            min_value=1,
            max_value=10,
            value=3,
            help="用于RSI平滑的参数。"
        )

        # Convert period to start and end dates
        start_date, end_date = convert_period_to_dates(period)

        return (
            start_date, end_date, rsi_period, overbought,
            oversold, smoothing
        )

    # Getting user input
    (
        start_date, end_date, rsi_period, overbought,
        oversold, smoothing
    ) = user_input_features()

    # ---------------------------
    # Fetch Historical Data using custom get_stock_prices function
    # ---------------------------
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("❌ 未获取到数据。请检查股票代码并重试。")
        st.stop()

    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # ---------------------------
    # Parameter Tuning Function
    # ---------------------------
    def tune_parameters(df, parameter_grid, initial_investment=100000):
        """
        Perform grid search to find the best parameter combination based on Sharpe Ratio.
        """
        best_sharpe = -np.inf
        best_params = {}
        results = []

        total_combinations = len(parameter_grid['rsi_period']) * len(parameter_grid['overbought']) * \
                            len(parameter_grid['oversold']) * len(parameter_grid['smoothing'])

        progress_bar = st.progress(0)
        status_text = st.empty()

        combination = 0

        for rsi_p, overb, underb, smooth_p in product(
            parameter_grid['rsi_period'],
            parameter_grid['overbought'],
            parameter_grid['oversold'],
            parameter_grid['smoothing']
        ):
            combination += 1
            status_text.text(f"参数调优中: 组合 {combination}/{total_combinations}")
            progress_bar.progress(combination / total_combinations)

            try:
                # Calculate RSI with current parameters
                df_temp = calculate_rsi(df.copy(), rsi_p, overb, underb, smooth_p)
                bullish_cross, bearish_cross = identify_signals(df_temp)
                # Unpack all returned values and extract sharpe_ratio
                _, _, _, _, sharpe_ratio, _, _, _ = evaluate_performance(df_temp, bullish_cross, bearish_cross, initial_investment)
            except Exception as e:
                # Handle any errors during calculation to prevent the tuning process from stopping
                st.warning(f"参数错误 (RSI_P: {rsi_p}, OB: {overb}, OS: {underb}, Smooth: {smooth_p}): {e}")
                sharpe_ratio = -np.inf  # Assign a poor sharpe ratio for failed combinations

            # Check if current sharpe is better
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_params = {
                    'rsi_period': rsi_p,
                    'overbought': overb,
                    'oversold': underb,
                    'smoothing': smooth_p
                }

            # Optional: Store results for further analysis
            results.append({
                'rsi_period': rsi_p,
                'overbought': overb,
                'oversold': underb,
                'smoothing': smooth_p,
                'sharpe_ratio': sharpe_ratio
            })

        progress_bar.empty()
        status_text.empty()
        return best_params, pd.DataFrame(results)

    # ---------------------------
    # Performance Evaluation Helper
    # ---------------------------
    def evaluate_performance(df, bullish_signals, bearish_signals, initial_investment=100000):
        """
        Compute performance metrics including Sharpe Ratio.
        """
        # Ensure data is sorted chronologically
        df = df.sort_values(by="date").reset_index(drop=True)

        trades = []
        returns = []
        portfolio_values = [initial_investment]
        position_open = False

        # Get the indices of bearish signals
        bearish_indices = bearish_signals.index.tolist()

        for bull_idx, bull_row in bullish_signals.iterrows():
            # If already holding a position, skip new buy signal
            if position_open:
                st.warning(f"警告: 已持有头寸于索引 {bull_idx}，跳过新的买入信号。")
                continue

            # Trade signal position index
            bull_position = df.index.get_loc(bull_row.name)

            # Buy date and price must be the next trading day after the signal
            if bull_position + 1 >= len(df):
                st.warning(f"警告: 索引 {bull_position} 后数据不足以执行买入，跳过此信号。")
                continue

            entry_date = df.loc[bull_position + 1, 'date']
            entry_price = df.loc[bull_position + 1, 'open']

            # Find the first bearish signal after the bullish signal
            future_bearish = [idx for idx in bearish_indices if idx > bull_position]

            # Ensure exit_position is within data range
            if not future_bearish:
                st.warning("警告: 未找到更多的卖出信号，结束交易循环。")
                break

            exit_position = future_bearish[0]

            # Exit date and price must be the next trading day after the bearish signal
            if exit_position + 1 >= len(df):
                st.warning(f"警告: 索引 {exit_position} 后数据不足以执行卖出，结束交易循环。")
                break

            exit_date = df.loc[exit_position + 1, 'date']
            exit_price = df.loc[exit_position + 1, 'open']

            # Check if exit date is after entry date
            if exit_date <= entry_date:
                st.warning(f"警告: 卖出日期 {exit_date} 不晚于买入日期 {entry_date}，跳过此交易。")
                continue

            return_pct = (exit_price - entry_price) / entry_price
            returns.append(return_pct)
            trades.append({
                "买入日期": entry_date.strftime("%Y-%m-%d"),
                "买入价格": entry_price,
                "卖出日期": exit_date.strftime("%Y-%m-%d"),
                "卖出价格": exit_price,
                "收益率": f"{return_pct:.2%}"
            })

            last_portfolio = portfolio_values[-1]
            portfolio_value = last_portfolio * (1 + return_pct)
            portfolio_values.append(portfolio_value)

            # Mark position as closed
            position_open = False

        # Create DataFrame for trades
        trades_df = pd.DataFrame(trades)

        avg_return = np.mean(returns) if returns else 0
        success_rate = sum([1 for ret in returns if ret > 0]) / len(returns) if returns else 0
        total_return = (portfolio_values[-1] - initial_investment) / initial_investment

        num_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
        annual_return = (portfolio_values[-1] / initial_investment) ** (1 / num_years) - 1 if num_years > 0 else 0

        risk_free_rate = 0.03
        excess_returns = [ret - risk_free_rate / 252 for ret in returns]
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) != 0 else 0

        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.cummax()
        drawdowns = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        return (
            avg_return,
            success_rate,
            total_return,
            annual_return,
            sharpe_ratio,
            max_drawdown,
            portfolio_values,
            trades_df
        )

    # ---------------------------
    # RSI Calculation Function
    # ---------------------------
    def calculate_rsi(df, rsi_period=14, overbought=70, oversold=30, smoothing=3):
        """
        Calculate RSI and apply smoothing if necessary.
        """
        # Calculate RSI using ta
        rsi_indicator = RSIIndicator(close=df['close'], window=rsi_period)
        df['RSI'] = rsi_indicator.rsi()

        # Apply smoothing (e.g., moving average) if smoothing > 1
        if smoothing > 1:
            df['RSI_Smooth'] = df['RSI'].rolling(window=smoothing, min_periods=1).mean()
        else:
            df['RSI_Smooth'] = df['RSI']

        return df

    # ---------------------------
    # Signal Identification Function
    # ---------------------------
    def identify_signals(df, overbought=70, oversold=30):
        """
        Identify bullish (oversold) and bearish (overbought) signals based on RSI.
        """
        # Ensure there are no NaN values in RSI
        df = df.dropna(subset=['RSI_Smooth'])

        # Identify crossovers for signals
        df['RSI_Overbought'] = np.where(df['RSI_Smooth'] > overbought, 1, 0)
        df['RSI_Oversold'] = np.where(df['RSI_Smooth'] < oversold, 1, 0)

        df['Crossover_Overbought'] = df['RSI_Overbought'].diff()
        df['Crossover_Oversold'] = df['RSI_Oversold'].diff()

        # Bullish signals: RSI crosses above oversold
        bullish_signals = df[df['Crossover_Oversold'] == 1]

        # Bearish signals: RSI crosses below overbought
        bearish_signals = df[df['Crossover_Overbought'] == -1]

        return bullish_signals, bearish_signals

    # ---------------------------
    # Find Confluence Zones Function
    # ---------------------------
    def find_confluence(df, rsi_col='RSI_Smooth', price_col='close'):
        """
        Identify confluence zones where RSI signals align with price movements.
        """
        latest_rsi = df[rsi_col].iloc[-1]
        latest_price = df[price_col].iloc[-1]

        confluence_levels = {}

        # Example: If RSI is above overbought and price is declining, it's bearish confluence
        # Adjust conditions based on strategy
        if latest_rsi > overbought and latest_price < df['close'].iloc[-2]:
            confluence_levels['Bearish Confluence'] = {
                'RSI': latest_rsi,
                'Price': latest_price
            }
        elif latest_rsi < oversold and latest_price > df['close'].iloc[-2]:
            confluence_levels['Bullish Confluence'] = {
                'RSI': latest_rsi,
                'Price': latest_price
            }

        return confluence_levels, df

    # ---------------------------
    # Plotting Function
    # ---------------------------
    def plot_rsi(df, bullish_signals, bearish_signals, confluences, ticker, rsi_period, overbought, oversold, show_rsi=True):
        """
        Plot the price data and RSI using Plotly.
        """
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{ticker.upper()} 的股价 (Price)', '相对强弱指数 (RSI)'),
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

        # RSI Line
        if show_rsi:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=df['RSI_Smooth'],
                    line=dict(color='orange', width=2),
                    name='RSI'
                ),
                row=2, col=1
            )

        # Overbought and Oversold Lines
        fig.add_hline(
            y=overbought, 
            line=dict(color='red', dash='dash'),
            row=2, col=1,
            annotation_text='Overbought',
            annotation_position='top left'
        )
        fig.add_hline(
            y=oversold, 
            line=dict(color='green', dash='dash'),
            row=2, col=1,
            annotation_text='Oversold',
            annotation_position='bottom left'
        )

        # Highlight Bullish Signals
        fig.add_trace(
            go.Scatter(
                x=bullish_signals['date'],
                y=bullish_signals['RSI_Smooth'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name='Bullish Signal'
            ),
            row=2, col=1
        )

        # Highlight Bearish Signals
        fig.add_trace(
            go.Scatter(
                x=bearish_signals['date'],
                y=bearish_signals['RSI_Smooth'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='Bearish Signal'
            ),
            row=2, col=1
        )

        # Highlight Confluence Zones
        for key, value in confluences.items():
            if key == 'Bullish Confluence':
                color = 'green'
                annotation = 'Bullish Confluence'
            elif key == 'Bearish Confluence':
                color = 'red'
                annotation = 'Bearish Confluence'
            else:
                color = 'yellow'
                annotation = 'Confluence'
            fig.add_hline(
                y=value['RSI'], 
                line=dict(color=color, dash='dot'), 
                row=2, col=1,
                annotation_text=annotation,
                annotation_position="top left"
            )

        fig.update_layout(
            title=f'Relative Strength Index (RSI) for {ticker.upper()}',
            yaxis_title='Price',
            template='plotly_dark',
            showlegend=True,
            height=800
        )

        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

        return fig

    # ---------------------------
    # Performance Analysis Function
    # ---------------------------
    def performance_analysis(df, bullish_signals, bearish_signals, initial_investment=100000):
        """
        计算并展示 RSI 指标的表现，包括最大回撤、总累计收益、年化收益率和夏普比率。
        还展示每笔交易的详细信息。信号在收盘时确认，交易在次日开盘价执行。
        """
        (
            avg_return,
            success_rate,
            total_return,
            annual_return,
            sharpe_ratio,
            max_drawdown,
            portfolio_values,
            trades_df
        ) = evaluate_performance(df, bullish_signals, bearish_signals, initial_investment)

        # 使用更小的字体展示指标表现
        st.markdown("""
            <style>
            .small-font {
                font-size: 14px !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # 指标表现展示
        st.markdown("## 📈 RSI 信号历史回测")

        # 投资组合增长图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df['date']),
            y=portfolio_values,
            mode='lines+markers',
            name='投资组合价值'
        ))
        fig.update_layout(
            title="假设初始投资为 10万 人民币的投资组合增长",
            xaxis_title="日期",
            yaxis_title="投资组合价值 (人民币)",
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Create a grid with columns
        col1, col2 = st.columns(2)

        # Layout the form inputs in a grid
        with col1:
            st.text_input("平均收益率", f"{avg_return:.2%}")
            st.text_input("总累计收益率", f"{total_return:.2%}")
            st.text_input("夏普比率", f"{sharpe_ratio:.2f}")

        with col2:
            st.text_input("成功率", f"{success_rate:.2%}")
            st.text_input("年化收益率", f"{annual_return:.2%}")
            st.text_input("最大回撤", f"{max_drawdown:.2%}")

        st.text("")  # Empty line for spacing
        st.text("")  # Empty line for spacing

        # 展示交易详情
        with st.expander("💼 查看交易详情", expanded=True):
            st.dataframe(trades_df, use_container_width=True)

        return sharpe_ratio  # Return Sharpe Ratio for tuning purposes

    # ---------------------------
    # Latest Signal and Recommendation
    # ---------------------------
    def get_latest_signal(bullish_signals, bearish_signals):
        if bullish_signals.empty and bearish_signals.empty:
            return "无最新信号", "无操作建议", "N/A"
        
        # Get the latest bullish and bearish signal dates
        latest_bullish_date = bullish_signals['date'].max() if not bullish_signals.empty else pd.Timestamp.min
        latest_bearish_date = bearish_signals['date'].max() if not bearish_signals.empty else pd.Timestamp.min

        # Determine which signal is more recent
        if latest_bullish_date > latest_bearish_date:
            latest_signal = "当前看涨"
            recommendation = "持股"
            latest_signal_date = latest_bullish_date.strftime("%Y-%m-%d")
        elif latest_bearish_date > latest_bullish_date:
            latest_signal = "当前看跌"
            recommendation = "空仓"
            latest_signal_date = latest_bearish_date.strftime("%Y-%m-%d")
        else:
            latest_signal = "无最新信号"
            recommendation = "无操作建议"
            latest_signal_date = "N/A"

        return latest_signal, recommendation, latest_signal_date

    # ---------------------------
    # RSI Calculation and Signal Identification
    # ---------------------------
    df = calculate_rsi(df, rsi_period, overbought, oversold, smoothing)
    bullish_signals, bearish_signals = identify_signals(df, overbought, oversold)


    # ---------------------------
    # Latest Signal and Recommendation
    # ---------------------------
    latest_signal, recommendation, latest_signal_date = get_latest_signal(bullish_signals, bearish_signals)

    # Display Latest Signal, Recommendation, and Timestamp with Custom HTML
    st.markdown("""
        <style>
        .info-box {
            background-color: #1e1e1e;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .info-title {
            font-size: 16px;
            color: #ffffff;
            margin-bottom: 5px;
        }
        .info-content-hold {
            font-size: 18px;
            color: #32CD32;  /* LimeGreen */
            font-weight: bold; /* This makes the text bold */
        }
        .info-content-dont-hold {
            font-size: 18px;
            color: #FF4500;  /* OrangeRed */
            font-weight: bold; /* This makes the text bold */
        }
        .info-content-no-action {
            font-size: 18px;
            color: #a9a9a9;  /* DarkGray */
            font-weight: bold; /* This makes the text bold */
        }
        .info-content-timestamp {
            font-size: 18px;
            color: #87CEFA;  /* LightSkyBlue */
            font-weight: bold; /* This makes the text bold */
        }
        </style>
    """, unsafe_allow_html=True)

    # Assign CSS class based on recommendation
    if recommendation == "持股":
        recommendation_class = "info-content-hold"
    elif recommendation == "空仓":
        recommendation_class = "info-content-dont-hold"
    else:
        recommendation_class = "info-content-no-action"

    # Display the information
    st.markdown(f"""
        <div class="info-box">
            <div class="info-title">🔔 最新信号</div>
            <div class="{recommendation_class}">&nbsp;&nbsp;&nbsp;{latest_signal}</div>
        </div>
        <div class="info-box">
            <div class="info-title">📅 最新信号生成时间</div>
            <div class="info-content-timestamp">&nbsp;&nbsp;&nbsp;{latest_signal_date}</div>
        </div>
        <div class="info-box">
            <div class="info-title">💡 投资建议</div>
            <div class="{recommendation_class}">&nbsp;&nbsp;&nbsp;{recommendation}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # ---------------------------
    # Find Confluence Zones
    # ---------------------------
    confluences, df = find_confluence(df)

    # ---------------------------
    # Plot Using Plotly
    # ---------------------------
    fig = plot_rsi(df, bullish_signals, bearish_signals, confluences, ticker, rsi_period, overbought, oversold)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

    # ---------------------------
    # Performance Analysis
    # ---------------------------
    performance_analysis(df, bullish_signals, bearish_signals, initial_investment=100000)

    # Optional: Display Data Table
    with st.expander("📊 查看原始数据 (View Raw Data)"):
        st.dataframe(df)

# ---------------------------
# Helper Function: Find Confluence Zones
# ---------------------------
def find_confluence(df, rsi_col='RSI_Smooth', price_col='close'):
    latest_rsi = df[rsi_col].iloc[-1]
    latest_price = df[price_col].iloc[-1]
    
    confluence_levels = {}
    
    # Example Conditions:
    # - Bullish Confluence: RSI < oversold and price is rising
    # - Bearish Confluence: RSI > overbought and price is falling
    # Adjust conditions based on your strategy

    # Since overbought and oversold are dynamic based on user input, they should be passed or stored appropriately
    # For simplicity, let's assume overbought=70 and oversold=30
    overbought = 70
    oversold = 30

    if (latest_rsi < oversold) and (latest_price > df['close'].iloc[-2]):
        confluence_levels['Bullish Confluence'] = {
            'RSI': latest_rsi,
            'Price': latest_price
        }
    elif (latest_rsi > overbought) and (latest_price < df['close'].iloc[-2]):
        confluence_levels['Bearish Confluence'] = {
            'RSI': latest_rsi,
            'Price': latest_price
        }

    return confluence_levels, df