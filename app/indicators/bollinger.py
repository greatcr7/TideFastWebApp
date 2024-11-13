from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from data.stock import get_stock_prices  # Ensure this custom module is available
import pytz
from itertools import product

# ---------------------------
# Bollinger Bands Analysis Function
# ---------------------------

def bollinger_band_analysis(ticker):
    st.markdown(f"# 📈 布林带指标 - {ticker.upper()}")
    
    # Sidebar for user inputs specific to Bollinger Bands Analysis
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

        # Convert to 'yyyy-mm-dd' format
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    # User input function with additional Bollinger Bands parameters
    def user_input_features(ma_window=None, num_std=None):
        period = st.sidebar.selectbox(
            "📅 时间跨度",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            help="选择分析的时间跨度。"
        )
        moving_average_window = st.sidebar.number_input(
            "🔢 移动平均窗口",
            min_value=1,
            max_value=200,
            value=ma_window if ma_window else 20,
            help="移动平均线的窗口期，通常设为20。"
        )
        number_of_std = st.sidebar.number_input(
            "🔢 标准差倍数",
            min_value=1,
            max_value=5,
            value=num_std if num_std else 2,
            help="布林带的标准差倍数，通常设为2。"
        )

        # Convert period to start and end dates
        start_date, end_date = convert_period_to_dates(period)

        return (
            start_date, end_date, moving_average_window, number_of_std
        )

    # Getting user input
    (
        start_date, end_date, ma_window, num_std
    ) = user_input_features()

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

        total_combinations = len(parameter_grid['ma_window']) * len(parameter_grid['num_std'])

        progress_bar = st.progress(0)
        status_text = st.empty()

        combination = 0

        for ma, std in product(
            parameter_grid['ma_window'],
            parameter_grid['num_std']
        ):
            combination += 1
            status_text.text(f"调优参数中: 组合 {combination}/{total_combinations}")
            progress_bar.progress(combination / total_combinations)

            try:
                # Calculate Bollinger Bands with current parameters
                df_temp = calculate_bollinger_bands(df.copy(), ma, std)
                buy_signals, sell_signals = identify_signals(df_temp)
                # Unpack all returned values and extract sharpe_ratio
                _, _, _, _, sharpe_ratio, _, _, _ = evaluate_performance(df_temp, buy_signals, sell_signals, initial_investment)
            except Exception as e:
                # Handle any errors during calculation to prevent the tuning process from stopping
                st.warning(f"参数错误 (MA: {ma}, Std: {std}): {e}")
                sharpe_ratio = -np.inf  # Assign a poor sharpe ratio for failed combinations

            # Check if current sharpe is better
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_params = {
                    'ma_window': ma,
                    'num_std': std
                }

            # Optional: Store results for further analysis
            results.append({
                'ma_window': ma,
                'num_std': std,
                'sharpe_ratio': sharpe_ratio
            })

        progress_bar.empty()
        status_text.empty()
        return best_params, pd.DataFrame(results)

    # ---------------------------
    # Performance Evaluation Helper
    # ---------------------------
    def evaluate_performance(df, buy_signals, sell_signals, initial_investment=100000):
        """
        Compute performance metrics including Sharpe Ratio.
        """
        # Ensure data is sorted chronologically
        df = df.sort_values(by="date").reset_index(drop=True)

        trades = []
        bullish_returns = []
        portfolio_values = [initial_investment]
        position_open = False

        # 获取卖出信号的索引列表
        sell_indices = sell_signals.index.tolist()

        for buy_idx, buy_row in buy_signals.iterrows():
            # 如果已经持有头寸，跳过新的买入信号
            if position_open:
                print(f"警告: 在索引 {buy_idx} 已有未平仓头寸，跳过此买入信号。")
                continue

            # 交易信号的实际位置
            buy_position = df.index.get_loc(buy_row.name)

            # 买入日期和价格必须为当前信号后的下一个交易日
            if buy_position + 1 >= len(df):
                print(f"警告: 在索引 {buy_position} 没有足够的数据来进行买入交易，跳过此信号。")
                continue

            entry_date = df.loc[buy_position + 1, 'date']
            entry_price = df.loc[buy_position + 1, 'open']

            # 找到第一个卖出信号出现的位置
            future_sell = [idx for idx in sell_indices if idx > buy_position]

            # 确认 exit_position 在未来数据范围内
            if not future_sell:
                print("警告: 没有找到更多的卖出信号，结束交易循环。")
                break

            exit_position = future_sell[0]

            # 退出日期和价格必须为卖出信号出现后的下一个交易日
            if exit_position + 1 >= len(df):
                print(f"警告: 在索引 {exit_position} 没有足够的数据来进行卖出交易，结束交易循环。")
                break

            exit_date = df.loc[exit_position + 1, 'date']
            exit_price = df.loc[exit_position + 1, 'open']

            # 检查退出日期是否在买入日期之后
            if exit_date <= entry_date:
                print(f"警告: 卖出日期 {exit_date} 早于或等于买入日期 {entry_date}，跳过不合理的交易。")
                continue

            bullish_return = (exit_price - entry_price) / entry_price
            bullish_returns.append(bullish_return)
            trades.append({
                "买入日期": entry_date,
                "买入价格": entry_price,
                "卖出日期": exit_date,
                "卖出价格": exit_price,
                "收益率": f"{bullish_return:.2%}"
            })

            last_portfolio_value = portfolio_values[-1]
            portfolio_value = last_portfolio_value * (1 + bullish_return)
            portfolio_values.append(portfolio_value)

            # 标记头寸已关闭
            position_open = False

        # 创建 DataFrame 记录交易
        trades_df = pd.DataFrame(trades)

        avg_bullish_return = np.mean(bullish_returns) if bullish_returns else 0
        bullish_success_rate = sum([1 for ret in bullish_returns if ret > 0]) / len(bullish_returns) if bullish_returns else 0
        total_cumulative_return = (portfolio_values[-1] - initial_investment) / initial_investment

        num_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
        annualized_return = (portfolio_values[-1] / initial_investment) ** (1 / num_years) - 1 if num_years > 0 else 0

        risk_free_rate = 0.03
        excess_returns = [ret - risk_free_rate / 252 for ret in bullish_returns]
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) != 0 else 0

        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.cummax()
        drawdowns = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        return (
            avg_bullish_return,
            bullish_success_rate,
            total_cumulative_return,
            annualized_return,
            sharpe_ratio,
            max_drawdown,
            portfolio_values,
            trades_df
        )

    # ---------------------------
    # Bollinger Bands Calculation Function
    # ---------------------------
    def calculate_bollinger_bands(df, ma_window=20, num_std=2):
        """
        Calculate Bollinger Bands: MA, Upper Band, Lower Band.
        """
        df['MA'] = df['close'].rolling(window=ma_window, min_periods=1).mean()
        df['STD'] = df['close'].rolling(window=ma_window, min_periods=1).std()
        df['Upper_Band'] = df['MA'] + (df['STD'] * num_std)
        df['Lower_Band'] = df['MA'] - (df['STD'] * num_std)

        return df

    # ---------------------------
    # Signal Identification Function
    # ---------------------------
    def identify_signals(df):
        """
        Identify buy and sell signals based on Bollinger Bands.
        Buy signal: Price crosses below Lower Band.
        Sell signal: Price crosses above Upper Band.
        """
        df = df.dropna(subset=['Upper_Band', 'Lower_Band'])

        # Identify buy signals
        df['Buy_Signal'] = np.where((df['close'] < df['Lower_Band']) & (df['close'].shift(1) >= df['Lower_Band'].shift(1)), 1, 0)
        buy_signals = df[df['Buy_Signal'] == 1]

        # Identify sell signals
        df['Sell_Signal'] = np.where((df['close'] > df['Upper_Band']) & (df['close'].shift(1) <= df['Upper_Band'].shift(1)), 1, 0)
        sell_signals = df[df['Sell_Signal'] == 1]

        return buy_signals, sell_signals

    # ---------------------------
    # Plotting Function
    # ---------------------------
    def plot_bollinger_bands(df, buy_signals, sell_signals, ticker,
                             ma_window=20, num_std=2):
        """
        Plot the price data and Bollinger Bands using Plotly.
        """
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{ticker.upper()} 的股价 (Price)', '布林带指标 (Bollinger Bands)'),
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

        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['MA'],
                line=dict(color='blue', width=2),
                name='移动平均线 (MA)'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['Upper_Band'],
                line=dict(color='orange', width=2),
                name='上轨 (Upper Band)'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['Lower_Band'],
                line=dict(color='orange', width=2),
                name='下轨 (Lower Band)'
            ),
            row=2, col=1
        )

        # Highlight Buy Signals
        fig.add_trace(
            go.Scatter(
                x=buy_signals['date'],
                y=buy_signals['close'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name='买入信号'
            ),
            row=2, col=1
        )

        # Highlight Sell Signals
        fig.add_trace(
            go.Scatter(
                x=sell_signals['date'],
                y=sell_signals['close'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='卖出信号'
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f'布林带指标 (Bollinger Bands) for {ticker.upper()}',
            yaxis_title='Price',
            template='plotly_dark',
            showlegend=True,
            height=1000
        )

        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

        return fig

    # ---------------------------
    # Performance Analysis Function
    # ---------------------------
    def performance_analysis(df, buy_signals, sell_signals, initial_investment=100000):
        """
        计算并展示布林带指标的表现，包括最大回撤、总累计收益、年化收益率和夏普比率。
        还展示每笔交易的详细信息。信号在收盘时确认，交易在次日开盘价执行。
        """
        (
            avg_bullish_return,
            bullish_success_rate,
            total_cumulative_return,
            annualized_return,
            sharpe_ratio,
            max_drawdown,
            portfolio_values,
            trades_df
        ) = evaluate_performance(df, buy_signals, sell_signals, initial_investment)

        # 使用更小的字体展示指标表现
        st.markdown("""
            <style>
            .small-font {
                font-size: 14px !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # 指标表现展示
        st.markdown("## 📈 布林带信号历史回测")

        # 投资组合增长图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime([trade['卖出日期'] for _, trade in trades_df.iterrows()] + [df['date'].iloc[-1]]), 
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
            st.text_input("平均收益率", f"{avg_bullish_return:.2%}")
            st.text_input("总累计收益率", f"{total_cumulative_return:.2%}")
            st.text_input("夏普比率", f"{sharpe_ratio:.2f}")

        with col2:
            st.text_input("成功率", f"{bullish_success_rate:.2%}")
            st.text_input("年化收益率", f"{annualized_return:.2%}")
            st.text_input("最大回撤", f"{max_drawdown:.2%}")

        st.text("")  # Empty line for spacing
        st.text("")  # Empty line for spacing

        # 展示交易详情
        with st.expander("💼 查看交易详情", expanded=True):
            st.dataframe(trades_df, use_container_width=True)

        return sharpe_ratio  # Return Sharpe Ratio for tuning purposes

    # ---------------------------
    # Main Logic
    # ---------------------------

    # Step 1: Fetch Historical Data using custom get_stock_prices function
    df = get_stock_prices(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error("❌ 未获取到数据。请检查股票代码并重试。")
        st.stop()

    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Initialize parameters (may be updated by tuning)
    params = {
        'ma_window': ma_window,
        'num_std': num_std
    }

    # Custom CSS for button styling
    st.markdown("""
        <style>
        .stButton > button {
            border: 2px solid #007BFF; /* Change the color and thickness as needed */
            border-radius: 8px; /* Adjust the border radius for a rounded effect */
            padding: 8px 16px; /* Increase padding to make the button more prominent */
            font-weight: bold; /* Make the text bold */
        }
        </style>
    """, unsafe_allow_html=True)

    # Add a button for parameter tuning
    if st.sidebar.button("🔍 自动参数调优"):
        st.sidebar.write("开始参数调优，请稍候...")
        # Define parameter grid
        parameter_grid = {
            'ma_window': [10, 15, 20, 25, 30],
            'num_std': [1, 2, 2.5, 3]
        }

        # Perform tuning
        best_params, tuning_results = tune_parameters(df, parameter_grid)

        if best_params:
            st.sidebar.success("参数调优完成！最佳参数已应用。")
            st.sidebar.write(f"**最佳移动平均窗口**: {best_params['ma_window']}")
            st.sidebar.write(f"**最佳标准差倍数**: {best_params['num_std']}")
        else:
            st.sidebar.error("参数调优失败。请检查数据或参数范围。")

        # Update parameters with best_params
        params = best_params if best_params else params  # Retain original params if tuning failed

        # Optionally, display tuning results
        with st.expander("🔍 查看调优结果"):
            st.dataframe(tuning_results.sort_values(by='sharpe_ratio', ascending=False).reset_index(drop=True), use_container_width=True)

    # Apply the selected or tuned parameters
    ma_window = params['ma_window']
    num_std = params['num_std']

    # Step 2: Calculate Bollinger Bands
    df = calculate_bollinger_bands(df, ma_window, num_std)

    # Step 3: Identify Buy and Sell Signals
    buy_signals, sell_signals = identify_signals(df)
     
     # ---------------------------
    # New Features: Latest Signal and Hold Recommendation
    # ---------------------------
    def get_latest_signal(buy_signals, sell_signals):
        if buy_signals.empty and sell_signals.empty:
            return "无最新信号", "无操作建议", "N/A"
        
        # Get the latest buy and sell crossover dates
        latest_buy_date = buy_signals['date'].max() if not buy_signals.empty else pd.Timestamp.min
        latest_sell_date = sell_signals['date'].max() if not sell_signals.empty else pd.Timestamp.min

        # Determine which signal is more recent
        if latest_buy_date > latest_sell_date:
            latest_signal = "当前买入信号"
            recommendation = "买入"
            latest_signal_date = latest_buy_date.strftime("%Y-%m-%d")
        elif latest_sell_date > latest_buy_date:
            latest_signal = "当前卖出信号"
            recommendation = "卖出"
            latest_signal_date = latest_sell_date.strftime("%Y-%m-%d")
        else:
            latest_signal = "无最新信号"
            recommendation = "无操作建议"
            latest_signal_date = "N/A"

        return latest_signal, recommendation, latest_signal_date

    latest_signal, recommendation, latest_signal_date = get_latest_signal(buy_signals, sell_signals)

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
    if recommendation == "买入":
        recommendation_class = "info-content-hold"
    elif recommendation == "卖出":
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
            <div class="info-title">💡 操作建议</div>
            <div class="{recommendation_class}">&nbsp;&nbsp;&nbsp;{recommendation}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Step 4: Plot Using Plotly
    fig = plot_bollinger_bands(
        df, buy_signals, sell_signals, ticker,
        ma_window, num_std
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

    # Step 5: Performance Analysis
    performance_analysis(df, buy_signals, sell_signals, initial_investment=100000)

    with st.expander("📊 查看原始信号数据"):
        st.dataframe(df)
