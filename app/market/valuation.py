import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


@st.cache_data
def fetch_valuation_data(symbol: str, indicator: str, period: str) -> pd.DataFrame:
    """
    Fetch A-Share Valuation Indicator data using AkShare.

    Args:
        symbol (str): A-share stock code (e.g., "002044").
        indicator (str): Valuation indicator. Choices:
                         {"总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"}.
        period (str): Time period. Choices:
                      {"近一年", "近三年", "近五年", "近十年", "全部"}.

    Returns:
        pd.DataFrame: DataFrame containing 'date' and 'value'.
    """
    df = ak.stock_zh_valuation_baidu(symbol=symbol, indicator=indicator, period=period)
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])
    # Sort by date
    df = df.sort_values('date')
    return df

def plot_valuation(df: pd.DataFrame, indicator: str, symbol: str) -> go.Figure:
    """
    Create Plotly plot for A-Share Valuation Indicators.

    Args:
        df (pd.DataFrame): DataFrame with 'date' and 'value'.
        indicator (str): The valuation indicator for the title.
        symbol (str): The stock symbol for the title.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = go.Figure()

    # Define color palette for different indicators
    color_palette = {
        "总市值": "rgba(54, 162, 235, 0.6)",       # Blue with transparency
        "市盈率(TTM)": "rgba(255, 99, 132, 0.6)",  # Red with transparency
        "市盈率(静)": "rgba(75, 192, 192, 0.6)",   # Green with transparency
        "市净率": "rgba(255, 206, 86, 0.6)",       # Yellow with transparency
        "市现率": "rgba(153, 102, 255, 0.6)",       # Purple with transparency
    }

    line_color_palette = {
        "总市值": "RoyalBlue",
        "市盈率(TTM)": "Crimson",
        "市盈率(静)": "MediumSeaGreen",
        "市净率": "Gold",
        "市现率": "RebeccaPurple",
    }

    # Add Bar trace
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['value'],
            name=indicator,
            marker_color=color_palette.get(indicator, 'rgba(100, 100, 100, 0.6)')  # Default grey
        )
    )

    # Add a line for average value
    average_value = df['value'].mean()
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=[average_value]*len(df),
            mode='lines',
            name=f"平均 {indicator}",
            line=dict(color=line_color_palette.get(indicator, 'Black'), dash='dash'),
        )
    )

    # Set x-axis title
    fig.update_xaxes(title_text="日期", showgrid=False)

    # Set y-axis title
    fig.update_yaxes(title_text=indicator, showgrid=False, zeroline=False)

    # Update layout
    fig.update_layout(
        title_text=f"{symbol} {indicator} 走势",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white",
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    return fig

def display_valuation_statistics():
    """
    Fetches, filters, and displays the A-Share Valuation Indicator data in Streamlit.
    Includes symbol selection, indicator selection, period selection, date range filters,
    raw data display, plot, and statistical summary.
    """
    st.header("A 股估值指标")

    # Sidebar - Symbol, Indicator, and Period Selection
    st.sidebar.header("参数选择")

    # Symbol Selection
    symbol = st.sidebar.text_input("指数", value="002044", max_chars=6)

    # Indicator Selection
    indicator_options = ["总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"]
    selected_indicator = st.sidebar.selectbox("选择估值指标 (Indicator)", options=indicator_options)

    # Period Selection
    period_options = ["近一年", "近三年", "近五年", "近十年", "全部"]
    selected_period = st.sidebar.selectbox("选择时间周期 (Period)", options=period_options)

    # Fetch the data
    with st.spinner("正在获取数据..."):
        try:
            data = fetch_valuation_data(symbol, selected_indicator, selected_period)
            if data.empty:
                st.error("未找到数据，请检查股票代码或参数选择。")
                return
        except Exception as e:
            st.error(f"数据获取失败: {e}")
            return

    # Sidebar - Date Range Filter based on fetched data
    st.sidebar.header("日期范围选择")
    min_date = data['date'].min().date()
    max_date = data['date'].max().date()

    # Default date range: Entire available data
    start_date = st.sidebar.date_input(
        "开始日期 (Start Date)",
        min_value=min_date,
        max_value=max_date,
        value=min_date
    )
    end_date = st.sidebar.date_input(
        "结束日期 (End Date)",
        min_value=min_date,
        max_value=max_date,
        value=max_date
    )

    if start_date > end_date:
        st.sidebar.error("错误: 开始日期必须早于结束日期。")
        return  # Exit the function early

    # Filter data based on date range
    mask = (data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))
    filtered_data = data.loc[mask]

    if filtered_data.empty:
        st.warning("在所选日期范围内未找到数据。")
        return

    # Generate and display the plot
    fig = plot_valuation(filtered_data, selected_indicator, symbol)
    st.plotly_chart(fig, use_container_width=True)

    # Display raw data (optional)
    with st.expander("查看原始数据"):
        st.dataframe(filtered_data)
