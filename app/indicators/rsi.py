from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from itertools import product
import pytz
from data.stock import get_stock_prices  # Ensure this custom module is available
import ta  # Technical Analysis library for RSI

# ---------------------------
# RSI Analysis Function
# ---------------------------

def rsi_analysis(ticker):
    st.markdown(f"# 📈 相对强弱指数 (RSI) 分析 - {ticker.upper()}")

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

        # Convert to 'yyyy-mm-dd' format
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    # User input function with additional RSI parameters
    def user_input_features(rsi_period=None, overbought=None, oversold=None):
        period = st.sidebar.selectbox(
            "📅 时间跨度",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            help="选择分析的时间跨度。"
        )
        rsi_period = st.sidebar.number_input(
            "🔢 RSI 周期",
            min_value=1,
            max_value=100,
            value=rsi_period if rsi_period else 14,
            help="RSI的计算周期，通常设为14。"
        )
        overbought = st.sidebar.number_input(
            "🔢 超买阈值",
            min_value=50,
            max_value=100,
            value=overbought if overbought else 70,
            help="RSI超过此值时视为超买。"
        )
        oversold = st.sidebar.number_input(
            "🔢 超卖阈值",
            min_value=0,
            max_value=50,
            value=oversold if oversold else 30,
            help="RSI低于此值时视为超卖。"
        )

        # Convert period to start and end dates
        start_date, end_date = convert_period_to_dates(period)

        return (
            start_date, end_date, rsi_period, overbought, oversold
        )

    # Getting user input
    (
        start_date, end_date, rsi_period, overbought, oversold
    ) = user_input_features()

    # ---------------------------
    # Parameter Tuning Function
    # ---------------------------
    def tune_parameters(df, parameter_grid, initial_investment=10000):
        """
        Perform grid search to find the best RSI parameter combination based on Sharpe Ratio.
        """
        best_sharpe = -np.inf
        best_params = {}
        results = []

        total_combinations = len(parameter_grid['rsi_period']) * len(parameter_grid['overbought']) * \
                            len(parameter_grid['oversold'])

        progress_bar = st.progress(0)
        status_text = st.empty()

        combination = 0

        for rsi_p, ob, os in product(
            parameter_grid['rsi_period'],
            parameter_grid['overbought'],
            parameter_grid['oversold']
        ):
            combination += 1
            status_text.text(f"Tuning parameters: Combination {combination}/{total_combinations}")
            progress_bar.progress(combination / total_combinations)

            try:
                # Calculate RSI with current parameters
                df_temp = calculate_rsi(df.copy(), rsi_p, ob, os)
                buy_signals, sell_signals = identify_signals(df_temp, ob, os)
                # Evaluate performance
                _, _, _, _, sharpe_ratio, _, _ = evaluate_performance(df_temp, buy_signals, sell_signals, initial_investment)
            except Exception as e:
                # Handle any errors during calculation to prevent the tuning process from stopping
                st.warning(f"Error with parameters (RSI Period: {rsi_p}, Overbought: {ob}, Oversold: {os}): {e}")
                sharpe_ratio = -np.inf  # Assign a poor sharpe ratio for failed combinations

            # Check if current sharpe is better
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_params = {
                    'rsi_period': rsi_p,
                    'overbought': ob,
                    'oversold': os
                }

            # Optional: Store results for further analysis
            results.append({
                'rsi_period': rsi_p,
                'overbought': ob,
                'oversold': os,
                'sharpe_ratio': sharpe_ratio
            })

        progress_bar.empty()
        status_text.empty()
        return best_params, pd.DataFrame(results)

    # ---------------------------
    # Performance Evaluation Helper
    # ---------------------------
    def evaluate_performance(df, buy_signals, sell_signals, initial_investment=10000):
        """
        Compute performance metrics including Sharpe Ratio.
        """
        # Ensure data is sorted chronologically
        df = df.sort_values(by="date").reset_index(drop=True)

        trades = []
        buy_returns = []
        portfolio_values = [initial_investment]
        position_open = False

        sell_indices = sell_signals.index.tolist()

        for buy_idx, buy_row in buy_signals.iterrows():
            # 如果已经持有头寸，跳过新的买入信号
            if position_open:
                print(f"警告: 在索引 {buy_idx} 已经有未平仓头寸，跳过此买入信号。")
                continue

            # 交易信号的实际位置
            buy_position = df.index.get_loc(buy_row.name)

            # 买入日期和价格必须为当前信号后的下一个交易日
            if buy_position + 1 >= len(df):
                print(f"警告: 在索引 {buy_position} 没有足够的数据来进行买入交易，跳过此信号。")
                continue

            entry_date = df.loc[buy_position + 1, 'date']
            entry_price = df.loc[buy_position + 1, 'open']

            # 找到第一个 sell 信号出现的位置
            future_sell = [idx for idx in sell_indices if idx > buy_position]

            # 确认 exit_position 在未来数据范围内
            if not future_sell:
                print("警告: 没有找到更多的卖出信号，结束交易循环。")
                break

            exit_position = future_sell[0]

            # 退出日期和价格必须为 sell 信号出现后的下一个交易日
            if exit_position + 1 >= len(df):
                print(f"警告: 在索引 {exit_position} 没有足够的数据来进行卖出交易，结束交易循环。")
                break

            exit_date = df.loc[exit_position + 1, 'date']
            exit_price = df.loc[exit_position + 1, 'open']

            # 检查退出日期是否在买入日期之后
            if exit_date <= entry_date:
                print(f"警告: 卖出日期 {exit_date} 早于或等于买入日期 {entry_date}，跳过不合理的交易。")
                continue

            buy_return = (exit_price - entry_price) / entry_price
            buy_returns.append(buy_return)
            trades.append({
                "买入日期": entry_date,
                "买入价格": entry_price,
                "卖出日期": exit_date,
                "卖出价格": exit_price,
                "收益率": f"{buy_return:.2%}"
            })

            last_portfolio_value = portfolio_values[-1]
            portfolio_value = last_portfolio_value * (1 + buy_return)
            portfolio_values.append(portfolio_value)

            # 标记头寸已关闭
            position_open = False

        # 创建 DataFrame 记录交易
        trades_df = pd.DataFrame(trades)

        avg_buy_return = np.mean(buy_returns) if buy_returns else 0
        buy_success_rate = sum([1 for ret in buy_returns if ret > 0]) / len(buy_returns) if buy_returns else 0
        total_cumulative_return = (portfolio_values[-1] - initial_investment) / initial_investment

        num_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
        annualized_return = (portfolio_values[-1] / initial_investment) ** (1 / num_years) - 1 if num_years > 0 else 0

        risk_free_rate = 0.03
        excess_returns = [ret - risk_free_rate / 252 for ret in buy_returns]
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) != 0 else 0

        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.cummax()
        drawdowns = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        return (
            avg_buy_return,
            buy_success_rate,
            total_cumulative_return,
            annualized_return,
            sharpe_ratio,
            max_drawdown,
            portfolio_values,
            trades_df
        )

    # ---------------------------
    # RSI Calculation Function
    # ---------------------------
    def calculate_rsi(df, rsi_period=14, overbought=70, oversold=30):
        """
        Calculate Relative Strength Index (RSI).
        """
        df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=rsi_period).rsi()
        return df

    # ---------------------------
    # Signal Identification Function
    # ---------------------------
    def identify_signals(df, overbought=70, oversold=30):
        """
        Identify buy and sell signals based on RSI crossovers.
        """
        df = df.dropna(subset=['RSI'])

        # Identify buy signals (RSI crossing above oversold)
        df['Buy_Signal'] = np.where((df['RSI'].shift(1) < oversold) & (df['RSI'] >= oversold), 1, 0)
        buy_signals = df[df['Buy_Signal'] == 1]

        # Identify sell signals (RSI crossing below overbought)
        df['Sell_Signal'] = np.where((df['RSI'].shift(1) > overbought) & (df['RSI'] <= overbought), -1, 0)
        sell_signals = df[df['Sell_Signal'] == -1]

        return buy_signals, sell_signals

    # ---------------------------
    # Plotting Function
    # ---------------------------
    def plot_rsi(df, buy_signals, sell_signals, ticker,
                rsi_period=14, overbought=70, oversold=30):
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
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['RSI'],
                line=dict(color='blue', width=2),
                name='RSI'
            ),
            row=2, col=1
        )

        # Overbought and Oversold Lines
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=[overbought]*len(df),
                line=dict(color='red', width=1, dash='dash'),
                name='超买阈值'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df['date'], y=[oversold]*len(df),
                line=dict(color='green', width=1, dash='dash'),
                name='超卖阈值'
            ),
            row=2, col=1
        )

        # Highlight Buy Signals
        fig.add_trace(
            go.Scatter(
                x=buy_signals['date'],
                y=buy_signals['RSI'],
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
                y=sell_signals['RSI'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='卖出信号'
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f'相对强弱指数 (RSI) 分析 - {ticker.upper()}',
            yaxis_title='价格',
            template='plotly_dark',
            showlegend=True,
            height=900
        )

        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

        return fig

    # ---------------------------
    # Performance Analysis Function
    # ---------------------------
    def performance_analysis(df, buy_signals, sell_signals, initial_investment=10000):
        """
        计算并展示 RSI 指标的表现，包括最大回撤、总累计收益、年化收益率和夏普比率。
        还展示每笔交易的详细信息。信号在收盘时确认，交易在次日开盘价执行。
        """
        (
            avg_buy_return,
            buy_success_rate,
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
        st.markdown("## 📈 RSI 信号历史回测")

        # 投资组合增长图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=portfolio_values,
            mode='lines',
            name='投资组合价值'
        ))
        fig.update_layout(
            title="假设初始投资为 10,000 人民币的投资组合增长",
            xaxis_title="日期",
            yaxis_title="投资组合价值 (人民币)",
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Create a grid with columns
        col1, col2 = st.columns(2)

        # Layout the form inputs in a grid
        with col1:
            st.text_input("平均买入收益率", f"{avg_buy_return:.2%}")
            st.text_input("总累计收益率", f"{total_cumulative_return:.2%}")
            st.text_input("夏普比率", f"{sharpe_ratio:.2f}")

        with col2:
            st.text_input("买入信号成功率", f"{buy_success_rate:.2%}")
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
        'rsi_period': rsi_period,
        'overbought': overbought,
        'oversold': oversold
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
            'rsi_period': [7, 14, 21],
            'overbought': [65, 70, 75],
            'oversold': [25, 30, 35]
        }

        # Perform tuning
        best_params, tuning_results = tune_parameters(df, parameter_grid)

        if best_params:
            st.sidebar.success("参数调优完成！最佳参数已应用。")
            st.sidebar.write(f"**最佳 RSI 周期**: {best_params['rsi_period']}")
            st.sidebar.write(f"**最佳超买阈值**: {best_params['overbought']}")
            st.sidebar.write(f"**最佳超卖阈值**: {best_params['oversold']}")
        else:
            st.sidebar.error("参数调优失败。请检查数据或参数范围。")

        # Update parameters with best_params
        params = best_params if best_params else params  # Retain original params if tuning failed

        # Optionally, display tuning results
        with st.expander("🔍 查看调优结果"):
            st.dataframe(tuning_results.sort_values(by='sharpe_ratio', ascending=False).reset_index(drop=True))

    # Apply the selected or tuned parameters
    rsi_period = params['rsi_period']
    overbought = params['overbought']
    oversold = params['oversold']

    # Step 2: Calculate RSI
    df = calculate_rsi(df, rsi_period, overbought, oversold)

    # Step 3: Identify Buy and Sell Signals
    buy_signals, sell_signals = identify_signals(df, overbought, oversold)

    # ---------------------------
    # New Features: Latest Signal and Recommendation
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
            recommendation = "持股"
            latest_signal_date = latest_buy_date.strftime("%Y-%m-%d")
        elif latest_sell_date > latest_buy_date:
            latest_signal = "当前卖出信号"
            recommendation = "空仓"
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
            <div class="info-title">💡 持股建议</div>
            <div class="{recommendation_class}">&nbsp;&nbsp;&nbsp;{recommendation}</div>
        </div>
    """, unsafe_allow_html=True)

    # Step 4: Plot Using Plotly
    fig = plot_rsi(
        df, buy_signals, sell_signals, ticker,
        rsi_period, overbought, oversold
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Step 5: Performance Analysis
    performance_analysis(df, buy_signals, sell_signals, initial_investment=10000)

    with st.expander("📊 查看原始信号数据"):
        st.dataframe(df)
