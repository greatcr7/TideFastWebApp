import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

@st.cache_data
def fetch_sector_change_data():
    """
    Fetch 板块异动详情 data using AkShare.

    Returns:
        pd.DataFrame: DataFrame containing sector change details for the latest trading day.
    """
    df = ak.stock_board_change_em()
    
    # Rename columns for clarity
    df.rename(columns={
        '板块名称': 'Sector Name',
        '涨跌幅': 'Change Percentage (%)',
        '主力净流入': 'Main Net Inflow (10k CNY)',
        '板块异动总次数': 'Total Sector Changes',
        '板块异动最频繁个股及所属类型-股票代码': 'Top Stock Code',
        '板块异动最频繁个股及所属类型-股票名称': 'Top Stock Name',
        '板块异动最频繁个股及所属类型-买卖方向': 'Top Stock Direction',
        '板块具体异动类型列表及出现次数': 'Detailed Change Types',
    }, inplace=True)
    
    # Convert numerical columns to appropriate data types
    df['Change Percentage (%)'] = pd.to_numeric(df['Change Percentage (%)'], errors='coerce')
    df['Main Net Inflow (10k CNY)'] = pd.to_numeric(df['Main Net Inflow (10k CNY)'], errors='coerce')
    df['Total Sector Changes'] = pd.to_numeric(df['Total Sector Changes'], errors='coerce')
    
    # Handle missing values if any
    df.fillna(0, inplace=True)
    
    return df

def plot_change_percentage(df):
    """
    Create a Plotly bar chart for Change Percentage per Sector.

    Args:
        df (pd.DataFrame): DataFrame with sector change details.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=df['Sector Name'],
            y=df['Change Percentage (%)'],
            name="Change Percentage (%)",
            marker_color='#1f77b4'  # Blue
        )
    )
    
    fig.update_layout(
        title="涨跌幅 (Change Percentage) per Sector",
        xaxis_title="Sector Name",
        yaxis_title="Change Percentage (%)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    
    return fig

def plot_main_net_inflow(df):
    """
    Create a Plotly bar chart for Main Net Inflow per Sector.

    Args:
        df (pd.DataFrame): DataFrame with sector change details.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=df['Sector Name'],
            y=df['Main Net Inflow (10k CNY)'],
            name="Main Net Inflow (10k CNY)",
            marker_color='#2ca02c'  # Green
        )
    )
    
    fig.update_layout(
        title="主力净流入 (Main Net Inflow) per Sector",
        xaxis_title="Sector Name",
        yaxis_title="Main Net Inflow (10k CNY)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    
    return fig

def plot_total_sector_changes(df):
    """
    Create a Plotly bar chart for Total Sector Changes per Sector.

    Args:
        df (pd.DataFrame): DataFrame with sector change details.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=df['Sector Name'],
            y=df['Total Sector Changes'],
            name="Total Sector Changes",
            marker_color='#d62728'  # Red
        )
    )
    
    fig.update_layout(
        title="板块异动总次数 (Total Sector Changes) per Sector",
        xaxis_title="Sector Name",
        yaxis_title="Total Sector Changes",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    
    return fig

def plot_change_percentage_vs_inflow(df):
    """
    Create a Plotly scatter plot for Change Percentage vs. Main Net Inflow.

    Args:
        df (pd.DataFrame): DataFrame with sector change details.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df['Change Percentage (%)'],
            y=df['Main Net Inflow (10k CNY)'],
            mode='markers',
            marker=dict(
                size=10,
                color=df['Change Percentage (%)'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Change Percentage (%)")
            ),
            text=df['Sector Name'],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Change Percentage: %{x}%<br>"
                "Main Net Inflow: %{y} 10k CNY<br>"
                "<extra></extra>"
            )
        )
    )
    
    fig.update_layout(
        title="涨跌幅 vs. 主力净流入 (Change Percentage vs. Main Net Inflow)",
        xaxis_title="Change Percentage (%)",
        yaxis_title="Main Net Inflow (10k CNY)",
        hovermode="closest",
        template="plotly_white",
        height=600,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    
    return fig

def plot_treemap_detailed_changes(df):
    """
    Create a Plotly treemap for Detailed Change Types per Sector.

    Args:
        df (pd.DataFrame): DataFrame with sector change details.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    # Expand the 'Detailed Change Types' into separate rows
    detailed_changes = []
    for index, row in df.iterrows():
        sector = row['Sector Name']
        change_types = row['Detailed Change Types']
        for change in change_types:
            change_type = change.get('t', 'Unknown')
            count = change.get('ct', 0)
            detailed_changes.append({
                'Sector Name': sector,
                'Change Type': change_type,
                'Count': count
            })
    
    detailed_df = pd.DataFrame(detailed_changes)
    
    # Group by Sector and Change Type
    treemap_df = detailed_df.groupby(['Sector Name', 'Change Type']).sum().reset_index()
    
    # Create treemap
    fig = go.Figure(go.Treemap(
        labels=treemap_df['Change Type'],
        parents=treemap_df['Sector Name'],
        values=treemap_df['Count'],
        marker=dict(colors=treemap_df['Count'], colorscale='Blues'),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Sector: %{parent}<br>"
            "Count: %{value}<br>"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title="板块具体异动类型及出现次数 (Detailed Change Types and Counts per Sector)",
        margin = dict(t=50, l=25, r=25, b=25),
        template="plotly_white",
    )
    
    return fig

def display_sector_change_details():
    """
    Fetches, processes, and displays the 板块异动详情 data in Streamlit.
    Includes interactive plots and data tables.
    """
    st.title("板块异动详情 (Sector Change Details)")
    st.markdown("""
    本页面展示了 **东方财富网** [行情中心](https://quote.eastmoney.com/changes/) 提供的当日板块异动详情数据。您可以通过以下可视化图表深入了解各板块的表现和异动情况。
    """)
    
    # Fetch data
    data = fetch_sector_change_data()
    
    # Display total number of sectors
    st.subheader(f"总板块数: {data.shape[0]}")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("涨跌幅 (Change Percentage) per Sector")
        fig_change_percentage = plot_change_percentage(data)
        st.plotly_chart(fig_change_percentage, use_container_width=True)
    
    with col2:
        st.header("主力净流入 (Main Net Inflow) per Sector")
        fig_main_net_inflow = plot_main_net_inflow(data)
        st.plotly_chart(fig_main_net_inflow, use_container_width=True)
    
    # Total Sector Changes Bar Chart
    st.header("板块异动总次数 (Total Sector Changes) per Sector")
    fig_total_sector_changes = plot_total_sector_changes(data)
    st.plotly_chart(fig_total_sector_changes, use_container_width=True)
    
    # Scatter Plot: Change Percentage vs. Main Net Inflow
    st.header("涨跌幅 vs. 主力净流入 (Change Percentage vs. Main Net Inflow)")
    fig_scatter = plot_change_percentage_vs_inflow(data)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Treemap: Detailed Change Types per Sector
    st.header("板块具体异动类型及出现次数 (Detailed Change Types and Counts per Sector)")
    fig_treemap = plot_treemap_detailed_changes(data)
    st.plotly_chart(fig_treemap, use_container_width=True)
        
    # Display Top 10 Sectors by Change Percentage
    st.header("涨跌幅排名前10的板块 (Top 10 Sectors by Change Percentage)")
    top10_change = data.sort_values(by='Change Percentage (%)', ascending=False).head(10)
    st.dataframe(top10_change[['Sector Name', 'Change Percentage (%)', 'Main Net Inflow (10k CNY)', 'Total Sector Changes']])
    
    # Display Top 10 Sectors by Main Net Inflow
    st.header("主力净流入排名前10的板块 (Top 10 Sectors by Main Net Inflow)")
    top10_inflow = data.sort_values(by='Main Net Inflow (10k CNY)', ascending=False).head(10)
    st.dataframe(top10_inflow[['Sector Name', 'Main Net Inflow (10k CNY)', 'Change Percentage (%)', 'Total Sector Changes']])
    
    # Display Raw Data
    with st.expander("查看原始数据"):
        st.dataframe(data)
