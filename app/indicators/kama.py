from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from itertools import product
from data.stock import get_stock_prices  # Ensure this custom module is available
from ta.momentum import KAMAIndicator
from ta.trend import EMAIndicator
import pytz

# ---------------------------
# KAMA Analysis Function with Enhanced Features
# ---------------------------

def kama_analysis(ticker):
    st.markdown(f"# 📈 KAMA 指标 - {ticker.upper()}")
    
    # Sidebar for user inputs specific to KAMA Analysis
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

    # User input function with additional KAMA parameters
    def user_input_features(short=None, long_=None, signal=None, smooth=None):
        period = st.sidebar.selectbox(
            "📅 时间跨度 (Time Period)",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            help="选择分析的时间跨度。"
        )
        kama_period = st.sidebar.number_input(
            "🔢 KAMA 周期 (KAMA Period)",
            min_value=1,
            max_value=50,
            value=short if short else 10,
            help="KAMA的计算周期。较短的周期使KAMA对价格变动更敏感。推荐值：10。"
        )
        kama_fast = st.sidebar.number_input(
            "🔢 快速常数 (Fast SC)",
            min_value=1,
            max_value=10,
            value=long_ if long_ else 2,
            help="快速平滑常数，用于调整KAMA的敏感度。较低的值使KAMA更敏感。推荐值：2。"
        )
        kama_slow = st.sidebar.number_input(
            "🔢 慢速常数 (Slow SC)",
            min_value=10,
            max_value=100,
            value=signal if signal else 30,
            help="慢速平滑常数，用于调整KAMA的平滑程度。较高的值使KAMA更平滑。推荐值：30。"
        )
        ema_short_window = st.sidebar.number_input(
            "🔢 EMA 短期窗口 (EMA Short Window)",
            min_value=10,
            max_value=100,
            value=smooth['ema_short'] if smooth and 'ema_short' in smooth else 50,
            help="短期指数移动平均线（EMA）的窗口大小。较短的窗口使EMA更敏感。推荐值：50。"
        )
        ema_long_window = st.sidebar.number_input(
            "🔢 EMA 长期窗口 (EMA Long Window)",
            min_value=100,
            max_value=300,
            value=smooth['ema_long'] if smooth and 'ema_long' in smooth else 200,
            help="长期指数移动平均线（EMA）的窗口大小。较长的窗口使EMA更平滑。推荐值：200。"
        )
        crossover_window = st.sidebar.number_input(
            "🔢 交叉检测窗口 (Crossover Window)",
            min_value=1,
            max_value=10,
            value=signal if signal else 1,
            help="定义检测交叉的最小天数，以避免虚假信号。推荐值：1。"
        )
        smoothing = st.sidebar.number_input(
            "🔢 平滑参数 (Smoothing Parameter)",
            min_value=1,
            max_value=10,
            value=3,
            help="用于KAMA的平滑参数。"
        )

        # Convert period to start and end dates
        start_date, end_date = convert_period_to_dates(period)

        return (
            start_date, end_date, kama_period, kama_fast,
            kama_slow, ema_short_window, ema_long_window,
            crossover_window, smoothing
        )

    # Getting user input
    (
        start_date, end_date, kama_period, kama_fast,
        kama_slow, ema_short_window, ema_long_window,
        crossover_window, smoothing
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

        total_combinations = len(parameter_grid['kama_period']) * len(parameter_grid['kama_fast']) * \
                            len(parameter_grid['kama_slow']) * len(parameter_grid['ema_short_window']) * \
                            len(parameter_grid['ema_long_window']) * len(parameter_grid['crossover_window'])

        progress_bar = st.progress(0)
        status_text = st.empty()

        combination = 0

        for kama_p, kama_f, kama_s, ema_s, ema_l, cross_w in product(
            parameter_grid['kama_period'],
            parameter_grid['kama_fast'],
            parameter_grid['kama_slow'],
            parameter_grid['ema_short_window'],
            parameter_grid['ema_long_window'],
            parameter_grid['crossover_window']
        ):
            combination += 1
            status_text.text(f"Tuning parameters: Combination {combination}/{total_combinations}")
            progress_bar.progress(combination / total_combinations)

            try:
                # Calculate KAMA with current parameters
                df_temp = calculate_kama(df.copy(), kama_p, kama_f, kama_s, ema_s, ema_l)
                bullish_cross, bearish_cross = identify_crossovers(df_temp, window=cross_w)
                # Unpack all returned values and extract sharpe_ratio
                _, _, _, _, sharpe_ratio, _, _, _ = evaluate_performance(df_temp, bullish_cross, bearish_cross, initial_investment)
            except Exception as e:
                # Handle any errors during calculation to prevent the tuning process from stopping
                #print(f"Error with parameters (KAMA_P: {kama_p}, KAMA_F: {kama_f}, KAMA_S: {kama_s}, EMA_S: {ema_s}, EMA_L: {ema_l}, Crossover_W: {cross_w}): {e}")
                sharpe_ratio = -np.inf  # Assign a poor sharpe ratio for failed combinations

            # Check if current sharpe is better
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_params = {
                    'kama_period': kama_p,
                    'kama_fast': kama_f,
                    'kama_slow': kama_s,
                    'ema_short_window': ema_s,
                    'ema_long_window': ema_l,
                    'crossover_window': cross_w
                }

            # Optional: Store results for further analysis
            results.append({
                'kama_period': kama_p,
                'kama_fast': kama_f,
                'kama_slow': kama_s,
                'ema_short_window': ema_s,
                'ema_long_window': ema_l,
                'crossover_window': cross_w,
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

        # Get the indices of bearish crossovers
        bearish_indices = bearish_crossovers.index.tolist()

        for bull_idx, bull_row in bullish_crossovers.iterrows():
            # If already holding a position, skip new buy signal
            if position_open:
                #print(f"Warning: Position already open at index {bull_idx}, skipping new buy signal.")
                continue

            # Trade signal position index
            bull_position = df.index.get_loc(bull_row.name)

            # Buy date and price must be the next trading day after the signal
            if bull_position + 1 >= len(df):
                #print(f"Warning: Not enough data after index {bull_position} to execute buy, skipping this signal.")
                continue

            entry_date = df.loc[bull_position + 1, 'date']
            entry_price = df.loc[bull_position + 1, 'open']

            # Find the first bearish crossover after the bullish crossover
            future_bearish = [idx for idx in bearish_indices if idx > bull_position]

            # Ensure exit_position is within data range
            if not future_bearish:
                #print("Warning: No further bearish crossovers found, ending trade loop.")
                break

            exit_position = future_bearish[0]

            # Exit date and price must be the next trading day after the bearish crossover
            if exit_position + 1 >= len(df):
                #print(f"Warning: Not enough data after index {exit_position} to execute sell, ending trade loop.")
                break

            exit_date = df.loc[exit_position + 1, 'date']
            exit_price = df.loc[exit_position + 1, 'open']

            # Check if exit date is after entry date
            if exit_date <= entry_date:
                #print(f"Warning: Exit date {exit_date} is not after entry date {entry_date}, skipping this trade.")
                continue

            bullish_return = (exit_price - entry_price) / entry_price
            bullish_returns.append(bullish_return)
            trades.append({
                "买入日期": entry_date.strftime("%Y-%m-%d"),
                "买入价格": entry_price,
                "卖出日期": exit_date.strftime("%Y-%m-%d"),
                "卖出价格": exit_price,
                "收益率": f"{bullish_return:.2%}"
            })

            last_portfolio_value = portfolio_values[-1]
            portfolio_value = last_portfolio_value * (1 + bullish_return)
            portfolio_values.append(portfolio_value)

            # Mark position as closed
            position_open = False

        # Create DataFrame for trades
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
    # KAMA Calculation Function
    # ---------------------------
    def calculate_kama(df, kama_period=10, kama_fast=2, kama_slow=30, ema_short_window=50, ema_long_window=200):
        """
        Calculate KAMA and EMAs.
        """
        # Calculate KAMA using ta
        kama_indicator = KAMAIndicator(close=df['close'], window=kama_period, pow1=kama_fast, pow2=kama_slow)
        df['KAMA'] = kama_indicator.kama()

        # Calculate EMAs using ta
        ema_short_indicator = EMAIndicator(close=df['close'], window=ema_short_window)
        ema_long_indicator = EMAIndicator(close=df['close'], window=ema_long_window)
        df['EMA_Short'] = ema_short_indicator.ema_indicator()
        df['EMA_Long'] = ema_long_indicator.ema_indicator()

        return df

    # ---------------------------
    # Crossover Identification Function
    # ---------------------------
    def identify_crossovers(df, price_col='close', kama_col='KAMA', window=1):
        """
        Identify bullish and bearish crossovers in KAMA.
        """
        # Ensure there are no NaN values in the necessary columns
        df = df.dropna(subset=[kama_col, price_col])

        bullish_crossovers = df[
            (df[kama_col] > df[price_col]) & 
            (df[kama_col].shift(window) <= df[price_col].shift(window))
        ]

        bearish_crossovers = df[
            (df[kama_col] < df[price_col]) & 
            (df[kama_col].shift(window) >= df[price_col].shift(window))
        ]

        return bullish_crossovers, bearish_crossovers

    # ---------------------------
    # Find Confluence Zones Function
    # ---------------------------
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

    # ---------------------------
    # Plotting Function
    # ---------------------------
    def plot_kama(df, bullish_crossovers, bearish_crossovers, confluences, ticker, ema_short_window, ema_long_window, show_ema=True, show_kama=True):
        """
        Plot the KAMA along with price data, EMAs, and crossovers using Plotly.
        """
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, 
            subplot_titles=(f'{ticker.upper()} 的股价和移动平均线', 'KAMA'),
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
                    line=dict(color='orange', width=3),  # Increased width from 1 to 3
                    name='KAMA'
                ), 
                row=1, col=1
            )
        
        # Crossovers Markers
        crossover_dates_bull = bullish_crossovers['date']
        crossover_prices_bull = bullish_crossovers['close']
        crossover_dates_bear = bearish_crossovers['date']
        crossover_prices_bear = bearish_crossovers['close']
        
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
                name = '看涨信号线'
            elif key == 'Bearish Confluence':
                color = 'red'
                name = '看跌信号线'
            else:
                color = 'yellow'
                name = 'Confluence'
            fig.add_hline(
                y=value['KAMA'], 
                line=dict(color=color, dash='dot'), 
                row=1, col=1,
                annotation_text=name,
                annotation_position="top left"
            )
        
        # KAMA Plot
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['KAMA'], 
                line=dict(color='orange', width=3),
                name='KAMA'
            ),
            row=2, col=1
        )
        
        # Add EMA lines in KAMA subplot if needed
        if show_ema:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['EMA_Short'], 
                    line=dict(color='blue', width=1, dash='dash'), 
                    name=f'EMA{ema_short_window}'
                ), 
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['EMA_Long'], 
                    line=dict(color='purple', width=1, dash='dash'), 
                    name=f'EMA{ema_long_window}'
                ), 
                row=2, col=1
            )
        
        fig.update_layout(
            title=f'Kaufman\'s Adaptive Moving Average (KAMA) for {ticker.upper()}',
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
        计算并展示 KAMA 指标的表现，包括最大回撤、总累计收益、年化收益率和夏普比率。
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
        st.markdown("## 📈 KAMA 信号历史回测")

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
    # Latest Signal and Recommendation
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

    # ---------------------------
    # Execute the KAMA Analysis
    # ---------------------------

    # Parameter Tuning UI
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
            'kama_period': [5, 10, 15, 20],
            'kama_fast': [2, 3, 4],
            'kama_slow': [20, 30, 40],
            'ema_short_window': [20, 50, 100],
            'ema_long_window': [100, 200, 300],
            'crossover_window': [1, 2]
        }

        # Perform tuning
        best_params, tuning_results = tune_parameters(df, parameter_grid)

        if best_params:
            st.sidebar.success("参数调优完成！最佳参数已应用。")
            st.sidebar.write(f"**最佳 KAMA 周期**: {best_params['kama_period']}")
            st.sidebar.write(f"**最佳 快速常数**: {best_params['kama_fast']}")
            st.sidebar.write(f"**最佳 慢速常数**: {best_params['kama_slow']}")
            st.sidebar.write(f"**最佳 EMA 短期窗口**: {best_params['ema_short_window']}")
            st.sidebar.write(f"**最佳 EMA 长期窗口**: {best_params['ema_long_window']}")
            st.sidebar.write(f"**最佳 交叉检测窗口**: {best_params['crossover_window']}")
        else:
            st.sidebar.error("参数调优失败。请检查数据或参数范围。")

        # Update parameters with best_params
        if best_params:
            kama_period = best_params['kama_period']
            kama_fast = best_params['kama_fast']
            kama_slow = best_params['kama_slow']
            ema_short_window = best_params['ema_short_window']
            ema_long_window = best_params['ema_long_window']
            crossover_window = best_params['crossover_window']

        # Optionally, display tuning results
        with st.expander("🔍 查看调优结果"):
            st.dataframe(tuning_results.sort_values(by='sharpe_ratio', ascending=False).reset_index(drop=True), use_container_width=True)

    # Step 2: Calculate KAMA and EMAs
    df = calculate_kama(df, kama_period, kama_fast, kama_slow, ema_short_window, ema_long_window)

    # Step 3: Identify Crossovers
    bullish_crossovers, bearish_crossovers = identify_crossovers(df, window=crossover_window)

    # ---------------------------
    # Latest Signal and Recommendation
    # ---------------------------
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
    confluences, df = find_confluence(df, ema_short_window, ema_long_window)
    fig = plot_kama(df, bullish_crossovers, bearish_crossovers, confluences, ticker, ema_short_window, ema_long_window)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

    # Step 5: Performance Analysis
    performance_analysis(df, bullish_crossovers, bearish_crossovers, initial_investment=100000)

    # Optional: Display Data Table
    with st.expander("📊 查看原始数据 (View Raw Data)"):
        st.dataframe(df)

# ---------------------------
# Helper Function: Find Confluence Zones
# ---------------------------
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