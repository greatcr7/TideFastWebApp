from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from data.stock import get_stock_prices  # Ensure this custom module is available
import pytz
from itertools import product

# ---------------------------
# EMD-MACD Analysis Function
# ---------------------------

def emd_macd_analysis(ticker):
    st.markdown(f"# 📈 Ehlers’ MACD均线 (EMD-MACD) - {ticker.upper()}")
    
    # # Add a selection box for allowing shorting
    # allow_shorting = st.sidebar.selectbox(
    #     "📉 是否允许策略做空",
    #     options=["不允许", "允许"],
    #     index=0,
    #     help="选择是否在策略中允许做空操作。\n\n美股和港股允许做空，A股不鼓励做空。"
    # )

    # Sidebar for user inputs specific to EMD-MACD Analysis
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

    # User input function with additional EMD-MACD parameters
    def user_input_features(short=None, long_=None, signal=None, smooth=None):
        period = st.sidebar.selectbox(
            "📅 时间跨度",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            help="选择分析的时间跨度。"
        )
        short_window = st.sidebar.number_input(
            "🔢 短期EMA窗口",
            min_value=1,
            max_value=100,
            value=short if short else 12,
            help="短期EMA的窗口期，通常设为12。"
        )
        long_window = st.sidebar.number_input(
            "🔢 长期EMA窗口",
            min_value=1,
            max_value=200,
            value=long_ if long_ else 26,
            help="长期EMA的窗口期，通常设为26。"
        )
        signal_window = st.sidebar.number_input(
            "🔢 信号线窗口",
            min_value=1,
            max_value=100,
            value=signal if signal else 9,
            help="信号线的窗口期，通常设为9。"
        )
        smoothing = st.sidebar.number_input(
            "🔢 平滑参数",
            min_value=1,
            max_value=100,
            value=smooth if smooth else 3,
            help="用于Savitzky-Golay滤波器的平滑参数，通常设为3。"
        )

        # Convert period to start and end dates
        start_date, end_date = convert_period_to_dates(period)

        return (
            start_date, end_date, short_window, long_window,
            signal_window, smoothing
        )

    # Getting user input
    (
        start_date, end_date, short_window, long_window,
        signal_window, smoothing
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

        total_combinations = len(parameter_grid['short_window']) * len(parameter_grid['long_window']) * \
                            len(parameter_grid['signal_window']) * len(parameter_grid['smoothing'])

        progress_bar = st.progress(0)
        status_text = st.empty()

        combination = 0

        for short, long_, signal, smooth in product(
            parameter_grid['short_window'],
            parameter_grid['long_window'],
            parameter_grid['signal_window'],
            parameter_grid['smoothing']
        ):
            combination += 1
            status_text.text(f"Tuning parameters: Combination {combination}/{total_combinations}")
            progress_bar.progress(combination / total_combinations)

            try:
                # Calculate EMD-MACD with current parameters
                df_temp = calculate_emd_macd(df.copy(), short, long_, signal, smooth)
                bullish_cross, bearish_cross = identify_crossovers(df_temp)
                # Unpack all returned values and extract sharpe_ratio
                _, _, _, _, sharpe_ratio, _, _, _ = evaluate_performance(df_temp, bullish_cross, bearish_cross, initial_investment)
            except Exception as e:
                # Handle any errors during calculation to prevent the tuning process from stopping
                st.warning(f"Error with parameters (Short: {short}, Long: {long_}, Signal: {signal}, Smooth: {smooth}): {e}")
                sharpe_ratio = -np.inf  # Assign a poor sharpe ratio for failed combinations

            # Check if current sharpe is better
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_params = {
                    'short_window': short,
                    'long_window': long_,
                    'signal_window': signal,
                    'smoothing': smooth
                }

            # Optional: Store results for further analysis
            results.append({
                'short_window': short,
                'long_window': long_,
                'signal_window': signal,
                'smoothing': smooth,
                'sharpe_ratio': sharpe_ratio
            })

        progress_bar.empty()
        status_text.empty()
        return best_params, pd.DataFrame(results)

    # ---------------------------
    # Performance Evaluation Helper
    # ---------------------------
    def evaluate_performance(df, bullish_crossovers, bearish_crossovers, initial_investment=100000):
        """
        Compute performance metrics including Sharpe Ratio.
        """
        # Ensure data is sorted chronologically
        df = df.sort_values(by="date").reset_index(drop=True)

        trades = []
        bullish_returns = []
        portfolio_values = [initial_investment]
        position_open = False

        # 获取 bearish_crossovers 的位置索引列表
        bearish_indices = bearish_crossovers.index.tolist()

        for bull_idx, bull_row in bullish_crossovers.iterrows():
            # 如果已经持有头寸，跳过新的买入信号
            if position_open:
                print(f"警告: 在索引 {bull_idx} 已经有未平仓头寸，跳过此买入信号。")
                continue

            # 交易信号的实际位置
            bull_position = df.index.get_loc(bull_row.name)

            # 买入日期和价格必须为当前信号后的下一个交易日
            if bull_position + 1 >= len(df):
                print(f"警告: 在索引 {bull_position} 没有足够的数据来进行买入交易，跳过此信号。")
                continue

            entry_date = df.loc[bull_position + 1, 'date']
            entry_price = df.loc[bull_position + 1, 'open']

            # 找到第一个 bearish crossover 出现的位置
            future_bearish = [idx for idx in bearish_indices if idx > bull_position]

            # 确认 exit_position 在未来数据范围内
            if not future_bearish:
                print("警告: 没有找到更多的 bearish crossover，结束交易循环。")
                break

            exit_position = future_bearish[0]

            # 退出日期和价格必须为 bearish crossover 出现后的下一个交易日
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
    # EMD-MACD Calculation Function
    # ---------------------------
    def calculate_emd_macd(df, short_window=12, long_window=26, signal_window=9, smoothing=3):
        """
        Calculate Ehlers’ Modified MACD using zero-lag EMA and Savitzky-Golay filter for smoothing.
        """
        # Zero-lag EMA implementation (simple approximation)
        def zero_lag_ema(series, span):
            ema = series.ewm(span=span, adjust=False).mean()
            lag = (series - ema).shift(1)
            return ema + lag

        # Calculate zero-lag EMAs
        df['EMA_short'] = zero_lag_ema(df['close'], span=short_window)
        df['EMA_long'] = zero_lag_ema(df['close'], span=long_window)

        # MACD line
        df['MACD'] = df['EMA_short'] - df['EMA_long']

        # Apply Savitzky-Golay filter for smoothing the MACD line
        # 使用右对齐的移动窗口滤波
        def right_aligned_smoothing(series, window):
            return series.rolling(window=window, min_periods=1).mean()
        
        window_length = smoothing*2+1
        if window_length > len(df):
            window_length = len(df) // 2 * 2 + 1 if len(df) >= 3 else 3
        
        # df['MACD_smooth'] = savgol_filter(df['MACD'].fillna(0), window_length=window_length, polyorder=2)
        df['MACD_smooth'] = right_aligned_smoothing(df['MACD'], window_length)

        # Signal line
        df['Signal'] = df['MACD_smooth'].ewm(span=signal_window, adjust=False).mean()

        # Histogram
        df['Histogram'] = df['MACD_smooth'] - df['Signal']

        return df

    # ---------------------------
    # Crossover Identification Function
    # ---------------------------
    def identify_crossovers(df):
        """
        Identify bullish and bearish crossovers in the EMD-MACD.
        """
        # Ensure there are no NaN values in the necessary columns
        df = df.dropna(subset=['MACD_smooth', 'Signal'])

        # Identify crossovers
        df['Crossover'] = np.where(df['MACD_smooth'] > df['Signal'], 1, 0)
        df['Crossover_Signal'] = df['Crossover'].diff()

        # Extract bullish and bearish crossovers
        bullish_crossovers = df[df['Crossover_Signal'] == 1]
        bearish_crossovers = df[df['Crossover_Signal'] == -1]

        return bullish_crossovers, bearish_crossovers
    
    # ---------------------------
    # Plotting Function
    # ---------------------------
    def plot_emd_macd(df, bullish_crossovers, bearish_crossovers, ticker,
                     short_window=12, long_window=26, signal_window=9):
        """
        Plot the price data and EMD-MACD using Plotly.
        """
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{ticker.upper()} 的股价 (Price)', 'Ehlers’ Modified MACD (EMD-MACD)'),
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

        # EMD-MACD Line
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['MACD_smooth'],
                line=dict(color='blue', width=2),
                name='EMD-MACD'
            ),
            row=2, col=1
        )

        # Signal Line
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['Signal'],
                line=dict(color='orange', width=2),
                name='Signal Line'
            ),
            row=2, col=1
        )

        # Histogram
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['Histogram'],
                marker_color=np.where(df['Histogram'] >= 0, 'green', 'red'),
                name='Histogram'
            ),
            row=2, col=1
        )

        # Highlight Bullish Crossovers
        fig.add_trace(
            go.Scatter(
                x=bullish_crossovers['date'],
                y=bullish_crossovers['MACD_smooth'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name='Bullish Crossover'
            ),
            row=2, col=1
        )

        # Highlight Bearish Crossovers
        fig.add_trace(
            go.Scatter(
                x=bearish_crossovers['date'],
                y=bearish_crossovers['MACD_smooth'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='Bearish Crossover'
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f'Ehlers’ Modified MACD (EMD-MACD) for {ticker.upper()}',
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
    def performance_analysis(df, bullish_crossovers, bearish_crossovers, initial_investment=100000):
        """
        计算并展示 EMD-MACD 指标的表现，包括最大回撤、总累计收益、年化收益率和夏普比率。
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
        ) = evaluate_performance(df, bullish_crossovers, bearish_crossovers, initial_investment)

        # 使用更小的字体展示指标表现
        st.markdown("""
            <style>
            .small-font {
                font-size: 14px !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # 指标表现展示
        st.markdown("## 📈 EMD-MACD 信号历史回测")

        # 投资组合增长图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(bullish_crossovers['date'].tolist() + [df['date'].iloc[-1]]), 
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
            st.text_input("平均看涨收益率", f"{avg_bullish_return:.2%}")
            st.text_input("总累计收益率", f"{total_cumulative_return:.2%}")
            st.text_input("夏普比率", f"{sharpe_ratio:.2f}")

        with col2:
            st.text_input("看涨信号成功率", f"{bullish_success_rate:.2%}")
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
        'short_window': short_window,
        'long_window': long_window,
        'signal_window': signal_window,
        'smoothing': smoothing
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
            'short_window': [5, 10, 12, 15, 20],
            'long_window': [20, 26, 30, 35],
            'signal_window': [5, 9, 12],
            'smoothing': [3, 5, 7]
        }

        # Perform tuning
        best_params, tuning_results = tune_parameters(df, parameter_grid)

        if best_params:
            st.sidebar.success("参数调优完成！最佳参数已应用。")
            st.sidebar.write(f"**最佳短期EMA窗口**: {best_params['short_window']}")
            st.sidebar.write(f"**最佳长期EMA窗口**: {best_params['long_window']}")
            st.sidebar.write(f"**最佳信号线窗口**: {best_params['signal_window']}")
            st.sidebar.write(f"**最佳平滑参数**: {best_params['smoothing']}")
        else:
            st.sidebar.error("参数调优失败。请检查数据或参数范围。")

        # Update parameters with best_params
        params = best_params if best_params else params  # Retain original params if tuning failed

        # Optionally, display tuning results
        with st.expander("🔍 查看调优结果"):
            st.dataframe(tuning_results.sort_values(by='sharpe_ratio', ascending=False).reset_index(drop=True), use_container_width=True)

    # Apply the selected or tuned parameters
    short_window = params['short_window']
    long_window = params['long_window']
    signal_window = params['signal_window']
    smoothing = params['smoothing']

    # Step 2: Calculate Ehlers’ Modified MACD (EMD-MACD)
    df = calculate_emd_macd(df, short_window, long_window, signal_window, smoothing)

    # Step 3: Identify MACD Crossovers
    bullish_crossovers, bearish_crossovers = identify_crossovers(df)
     
     # ---------------------------
    # New Features: Latest Signal and Hold Recommendation
    # ---------------------------
    def get_latest_signal(bullish_crossovers, bearish_crossovers):
        if bullish_crossovers.empty and bearish_crossovers.empty:
            return "无最新信号", "无操作建议", "N/A"
        
        # Get the latest bullish and bearish crossover dates
        latest_bullish_date = bullish_crossovers['date'].max() if not bullish_crossovers.empty else pd.Timestamp.min
        latest_bearish_date = bearish_crossovers['date'].max() if not bearish_crossovers.empty else pd.Timestamp.min

        # Determine which crossover is more recent
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

    latest_signal, recommendation, latest_signal_date = get_latest_signal(bullish_crossovers, bearish_crossovers)

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
            color: #FF4500;  /* LimeGreen */
            font-weight: bold; /* This makes the text bold */
        }
        .info-content-dont-hold {
            font-size: 18px;
            color: #32CD32;  /* OrangeRed */
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
    
    # Step 5: Plot Using Plotly
    fig = plot_emd_macd(
        df, bullish_crossovers, bearish_crossovers, ticker,
        short_window, long_window, signal_window
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

    # Step 6: Performance Analysis
    performance_analysis(df, bullish_crossovers, bearish_crossovers, initial_investment=100000)

    with st.expander("📊 查看原始信号数据"):
        st.dataframe(df)
