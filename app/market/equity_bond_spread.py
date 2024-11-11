# equity_bond_spread.py

import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def fetch_equity_bond_spread_data():
    """
    Fetch 股债利差 data using AkShare.
    
    Returns:
        pd.DataFrame: DataFrame containing date, CSI 300 Index, Equity-Bond Spread, and its moving average.
    """
    df = ak.stock_ebs_lg()
    # Rename columns for clarity
    df.rename(columns={
        '日期': 'date', 
        '沪深300指数': 'CSI 300 Index', 
        '股债利差': 'Equity-Bond Spread', 
        '股债利差均线': 'Equity-Bond Spread MA'
    }, inplace=True)
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])
    # Sort by date
    df = df.sort_values('date')
    return df

def plot_equity_bond_spread(df):
    """
    Create Plotly plot for 股债利差.
    
    Args:
        df (pd.DataFrame): DataFrame with 'date', 'CSI 300 Index', 'Equity-Bond Spread', and 'Equity-Bond Spread MA'.
    
    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add CSI 300 Index trace
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['CSI 300 Index'], name="沪深300指数", line=dict(color='green')),
        secondary_y=False,
    )

    # Add Equity-Bond Spread trace
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['Equity-Bond Spread'], name="股债利差", line=dict(color='orange')),
        secondary_y=True,
    )

    # Add Equity-Bond Spread MA trace
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['Equity-Bond Spread MA'], name="股债利差均线", line=dict(color='purple', dash='dash')),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="日期")

    # Set y-axes titles
    fig.update_yaxes(title_text="沪深300指数", secondary_y=False)
    fig.update_yaxes(title_text="股债利差", secondary_y=True)

    # Update layout
    fig.update_layout(
        title_text="股债利差与沪深300指数走势",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        template="plotly_white",
    )

    return fig

def display_equity_bond_spread():
    """
    Fetches, filters, and displays the 股债利差 data in Streamlit.
    Includes date range filters, raw data display, plot, and statistical summary.
    """
    st.header("股债利差 (Equity-Bond Spread)")

    # Fetch the data
    ebs_data = fetch_equity_bond_spread_data()

    # Sidebar - Date Range Filter specific to Equity-Bond Spread
    st.sidebar.header("股债利差参数")
    min_date_ebs = ebs_data['date'].min().date()
    max_date_ebs = ebs_data['date'].max().date()

    # Default date range: Entire available data
    start_date_ebs = st.sidebar.date_input(
        "开始日期",
        min_value=min_date_ebs,
        max_value=max_date_ebs,
        value=min_date_ebs
    )
    end_date_ebs = st.sidebar.date_input(
        "结束日期",
        min_value=min_date_ebs,
        max_value=max_date_ebs,
        value=max_date_ebs
    )

    if start_date_ebs > end_date_ebs:
        st.sidebar.error("错误: 开始日期必须早于结束日期。")
        return  # Exit the function early

    # Filter data based on date range
    filtered_ebs_data = ebs_data[
        (ebs_data['date'] >= pd.to_datetime(start_date_ebs)) &
        (ebs_data['date'] <= pd.to_datetime(end_date_ebs))
    ]


    # Generate and display the plot
    ebs_plot = plot_equity_bond_spread(filtered_ebs_data)
    st.plotly_chart(ebs_plot, use_container_width=True)


    # Display raw data (optional)
    with st.expander("查看原始数据"):
        st.dataframe(filtered_ebs_data)
