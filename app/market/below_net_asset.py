import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


@st.cache_data
def fetch_below_net_asset_data(symbol: str) -> pd.DataFrame:
    """
    Fetch 破净股统计 data using AkShare.

    Args:
        symbol (str): The stock index symbol. Choices are {"全部A股", "沪深300", "上证50", "中证500"}.

    Returns:
        pd.DataFrame: DataFrame containing date, below_net_asset, total_company, and below_net_asset_ratio.
    """
    df = ak.stock_a_below_net_asset_statistics(symbol=symbol)
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])
    # Sort by date
    df = df.sort_values('date')
    return df

def plot_below_net_asset(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Create Plotly plot for 破净股统计.

    Args:
        df (pd.DataFrame): DataFrame with 'date', 'below_net_asset', 'total_company', and 'below_net_asset_ratio'.
        symbol (str): The stock index symbol for the title.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Below Net Asset trace
    fig.add_trace(
        go.Bar(
            x=df['date'], 
            y=df['below_net_asset'], 
            name="破净股家数", 
            marker_color='rgba(255, 99, 71, 0.6)'  # Tomato color with some transparency
        ),
        secondary_y=False,
    )

    # Add Below Net Asset Ratio trace
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['below_net_asset_ratio'], 
            name="破净股比率", 
            line=dict(color='RoyalBlue', width=2)
        ),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="日期", showgrid=False)

    # Set y-axes titles and remove grid lines
    fig.update_yaxes(
        title_text="破净股家数", 
        secondary_y=False, 
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="破净股比率", 
        secondary_y=True, 
        tickformat=".2%", 
        showgrid=False,
        zeroline=False
    )

    # Update layout
    fig.update_layout(
        title_text=f"{symbol} 破净股统计",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bordercolor="black", borderwidth=1),
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    return fig


def display_below_net_asset_statistics():
    """
    Fetches, filters, and displays the 破净股统计 data in Streamlit.
    Includes symbol selection, date range filters, raw data display, plot, and statistical summary.
    """
    st.header("破净股统计 (Below Net Asset Statistics)")

    # Sidebar - Symbol Selection
    st.sidebar.header("选择股票指数")
    symbol_options = ["全部A股", "沪深300", "上证50", "中证500"]
    selected_symbol = st.sidebar.selectbox("选择指数", options=symbol_options, index=0)

    # Fetch the data
    with st.spinner("正在获取数据..."):
        data = fetch_below_net_asset_data(symbol=selected_symbol)

    # Sidebar - Date Range Filter
    st.sidebar.header("日期范围选择")
    min_date = data['date'].min().date()
    max_date = data['date'].max().date()

    # Default date range: Entire available data
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

    # Filter data based on date range
    mask = (data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))
    filtered_data = data.loc[mask]


    # Generate and display the plot
    fig = plot_below_net_asset(filtered_data, selected_symbol)
    st.plotly_chart(fig, use_container_width=True)

    # Display raw data (optional)
    with st.expander("查看原始数据"):
        st.dataframe(filtered_data)


