# equity_bond_spread.py

import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


@st.cache_data
def fetch_stock_account_statistics():
    """
    Fetch 股票账户统计 data using AkShare.

    Returns:
        pd.DataFrame: DataFrame containing stock account statistics from 201504 to 202308.
    """
    df = ak.stock_account_statistics_em()

    # Rename columns for clarity
    df.rename(columns={
        '数据日期': 'Data Date',
        '新增投资者-数量': 'New Investors (万户)',
        '新增投资者-环比': 'New Investors MoM',
        '新增投资者-同比': 'New Investors YoY',
        '期末投资者-总量': 'Total Investors (万户)',
        '期末投资者-A股账户': 'A-share Accounts (万户)',
        '期末投资者-B股账户': 'B-share Accounts (万户)',
        '沪深总市值': 'Total Market Cap',
        '沪深户均市值': 'Average Market Cap per Investor (万)',
        '上证指数-收盘': 'SSE Index Close',
        '上证指数-涨跌幅': 'SSE Index Change',
    }, inplace=True)

    # Convert 'Data Date' to datetime
    # Corrected the format to '%Y-%m' to match "YYYY-MM"
    df['Data Date'] = pd.to_datetime(df['Data Date'], format='%Y-%m')

    # Sort by date
    df = df.sort_values('Data Date')

    return df
    
def plot_new_investors(df):
    """
    Create a Plotly plot for 新增投资者.

    Args:
        df (pd.DataFrame): DataFrame with stock account statistics.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df['Data Date'],
            y=df['New Investors (万户)'],
            name="新增投资者 (万户)",
            line=dict(color='blue')
        )
    )

    fig.update_layout(
        title="新增投资者 (万户) 趋势",
        xaxis_title="日期",
        yaxis_title="新增投资者 (万户)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )

    return fig

def plot_total_investors(df):
    """
    Create a Plotly plot for 期末投资者总量.

    Args:
        df (pd.DataFrame): DataFrame with stock account statistics.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df['Data Date'],
            y=df['Total Investors (万户)'],
            name="期末投资者总量 (万户)",
            line=dict(color='green')
        )
    )

    fig.update_layout(
        title="期末投资者总量 (万户) 趋势",
        xaxis_title="日期",
        yaxis_title="期末投资者总量 (万户)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )

    return fig

def plot_sse_index_and_market_cap(df):
    """
    Create a Plotly plot for 上证指数和户均市值 with dual y-axes.

    Args:
        df (pd.DataFrame): DataFrame with stock account statistics.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = go.Figure()

    # 上证指数-收盘 (Primary Y-Axis)
    fig.add_trace(
        go.Scatter(
            x=df['Data Date'],
            y=df['SSE Index Close'],
            name="上证指数-收盘",
            line=dict(color='red')
        )
    )

    # 沪深户均市值 (万) (Secondary Y-Axis)
    fig.add_trace(
        go.Scatter(
            x=df['Data Date'],
            y=df['Average Market Cap per Investor (万)'],
            name="沪深户均市值 (万)",
            line=dict(color='purple', dash='dash')
        )
    )

    # Update layout for dual y-axes
    fig.update_layout(
        title="上证指数与沪深户均市值走势",
        xaxis=dict(
            title="日期"
        ),
        yaxis=dict(
            title="上证指数",
            titlefont=dict(color='red'),
            tickfont=dict(color='red')
        ),
        yaxis2=dict(
            title="沪深户均市值 (万)",
            titlefont=dict(color='purple'),
            tickfont=dict(color='purple'),
            overlaying='y',
            side='right'
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bordercolor="Black",
            borderwidth=1
        ),
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    # Assign traces to their respective y-axes
    fig.data[0].update(yaxis='y1')  # Primary y-axis
    fig.data[1].update(yaxis='y2')  # Secondary y-axis

    return fig


def display_stock_account_statistics():
    """
    Fetches, filters, and displays the 股票账户统计 data in Streamlit.
    Includes date range filters, raw data display, plots, and statistical summary.
    """
    st.title("股票账户统计 (Stock Account Statistics)")
    st.markdown("""
    本页面展示了从 **2015年4月** 至 **2023年8月** 的股票账户统计数据，包括新增投资者数量、期末投资者总量、A股与B股账户分布、沪深总市值、户均市值以及上证指数表现。
    数据来源于 [东方财富网](https://data.eastmoney.com/cjsj/gpkhsj.html)。
    """)

    # Fetch the data
    data = fetch_stock_account_statistics()

    # Sidebar - Date Range Filter
    st.sidebar.header("筛选参数")
    min_date = data['Data Date'].min().date()
    max_date = data['Data Date'].max().date()

    # Since data is monthly, we'll set default dates to the first day of the month
    start_date = st.sidebar.date_input(
        "开始日期",
        min_value=min_date,
        max_value=max_date,
        value=min_date
    )
    end_date = st.sidebar.date_input(
        "结束日期",
        min_value=min_date,
        max_value=max_date,
        value=max_date
    )

    if start_date > end_date:
        st.sidebar.error("错误: 开始日期必须早于结束日期。")
        return  # Exit the function early

    # Convert selected dates to the first day of the month for consistency
    start_date = pd.to_datetime(start_date.replace(day=1))
    end_date = pd.to_datetime(end_date.replace(day=1))

    # Filter data based on date range
    filtered_data = data[
        (data['Data Date'] >= start_date) &
        (data['Data Date'] <= end_date)
    ]

    if filtered_data.empty:
        st.warning("在所选日期范围内没有数据。请调整日期范围。")
        return

    # Generate and display the plots
    st.header("新增投资者 (万户) 趋势")
    fig_new_investors = plot_new_investors(filtered_data)
    st.plotly_chart(fig_new_investors, use_container_width=True)

    st.header("期末投资者总量 (万户) 趋势")
    fig_total_investors = plot_total_investors(filtered_data)
    st.plotly_chart(fig_total_investors, use_container_width=True)

    st.header("上证指数与沪深户均市值走势")
    fig_sse_market_cap = plot_sse_index_and_market_cap(filtered_data)
    st.plotly_chart(fig_sse_market_cap, use_container_width=True)

    # Display raw data (optional)
    with st.expander("查看原始数据"):
        st.dataframe(filtered_data)

