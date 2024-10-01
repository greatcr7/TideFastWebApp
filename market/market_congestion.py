import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

@st.cache_data
def fetch_market_congestion_data():
    """
    Fetch 大盘拥挤度 data using AkShare.
    Returns:
        pd.DataFrame: DataFrame containing date, close, and congestion.
    """
    df = ak.stock_a_congestion_lg()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def plot_market_congestion(df):
    """
    Create Plotly plot for 大盘拥挤度.
    
    Args:
        df (pd.DataFrame): DataFrame with 'date', 'close', and 'congestion' columns.
    
    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add closing price trace
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['close'], name="收盘价", line=dict(color='blue')),
        secondary_y=False,
    )

    # Add congestion trace
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['congestion'], name="拥挤度", line=dict(color='red')),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="日期")

    # Set y-axes titles
    fig.update_yaxes(title_text="收盘价", secondary_y=False)
    fig.update_yaxes(title_text="拥挤度", secondary_y=True)

    # Update layout
    fig.update_layout(
        title_text="大盘收盘价与拥挤度走势",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white",
    )

    return fig

def display_market_congestion():
    """
    Fetches, filters, and displays the 大盘拥挤度 data in Streamlit.
    Includes date range filters, raw data display, plot, and statistical summary.
    """
    st.header("大盘拥挤度 (Market Congestion)")

    # Fetch the data
    congestion_data = fetch_market_congestion_data()

    # Sidebar - Date Range Filter specific to Market Congestion
    st.sidebar.header("大盘拥挤度参数")
    min_date = congestion_data['date'].min().date()
    max_date = congestion_data['date'].max().date()

    # Calculate default_start as max_date minus 4 years
    try:
        default_start = max_date.replace(year=max_date.year - 4)
    except ValueError:
        # Handle leap year or other date issues
        default_start = max_date - pd.DateOffset(years=4)
        default_start = default_start.date()

    # Ensure default_start is not before min_date
    if default_start < min_date:
        default_start = min_date

    start_date = st.sidebar.date_input(
        "开始日期",
        min_value=min_date,
        max_value=max_date,
        value=default_start
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

    # Filter data based on date range
    filtered_data = congestion_data[
        (congestion_data['date'] >= pd.to_datetime(start_date)) &
        (congestion_data['date'] <= pd.to_datetime(end_date))
    ]

    # Display raw data (optional)
    with st.expander("查看原始数据"):
        st.dataframe(filtered_data)

    # Generate and display the plot
    congestion_plot = plot_market_congestion(filtered_data)
    st.plotly_chart(congestion_plot, use_container_width=True)

    # Display statistical summary
    st.markdown("### 统计摘要")
    st.write(filtered_data.describe())