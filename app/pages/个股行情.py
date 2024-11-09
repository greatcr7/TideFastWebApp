import json
import streamlit as st
import akshare as ak
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data.stock import get_stock_prices

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="个股行情 - 历澜投资",
    layout="wide",
    page_icon="images/logo.png"
)

# ---------------------------
# Custom CSS for Styling (Optional)
# ---------------------------
def local_css():
    st.markdown("""
    <style>
    /* Style for all buttons */
    div.stButton > button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        font-weight: bold;
        border: 1px solid #333; /* Added border (rim) */
        border-radius: 12px; /* Rounded corners */
        transition: background-color 0.3s, color 0.3s, border-color 0.3s; /* Smooth transition */
        cursor: pointer;
    }
    /* Additional styling for better aesthetics */
    .metric-box {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ---------------------------
# Load Stock Data from JSON
# ---------------------------
@st.cache_data
def load_stock_data(json_path='stocks.json'):
    """
    Load stock data from a JSON file and filter by market.
    
    Args:
        json_path (str): Path to the JSON file containing stock data.
    
    Returns:
        list: List of stock dictionaries with 'cname', 'ticker', and 'market'.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            stocks = json.load(f)
        # Filter stocks where market is 'china'
        china_stocks = [stock for stock in stocks if stock.get('market', '').lower() == 'china']
        return china_stocks
    except FileNotFoundError:
        st.error(f"股票数据文件未找到: {json_path}")
        return []
    except json.JSONDecodeError:
        st.error("股票数据文件格式错误。")
        return []

stocks = load_stock_data()

if not stocks:
    st.stop()  # Stop the app if no stocks are loaded

# Create a mapping from display name to ticker
stock_display_to_ticker = {f"{stock['cname']} ({stock['ticker']})": stock['ticker'] for stock in stocks}
stock_display_names = list(stock_display_to_ticker.keys())

# ---------------------------
# Market Quotes Analysis Function
# ---------------------------
@st.cache_data
def fetch_market_quotes(stock_symbol):
    """
    Fetch bid-ask data for the given stock symbol using akshare.
    
    Args:
        stock_symbol (str): The stock ticker symbol with market prefix (e.g., "sz000001").
    
    Returns:
        pd.DataFrame: DataFrame containing bid-ask data.
    """
    try:
        df = ak.stock_bid_ask_em(symbol=stock_symbol[:-3])
        return df
    except Exception as e:
        st.error(f"获取行情报价数据时出错: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def display_latest_price_and_change(metrics):
    """
    Display the latest price and change in Robinhood style at the top of the app using Streamlit's native components.

    Args:
        metrics (dict): Dictionary containing key metrics including '最新', '涨跌', and '涨幅'.
    """
    latest_price = metrics.get("最新", "N/A")
    change = metrics.get("涨跌", "0")
    change_percent = metrics.get("涨幅", "0")

    # Attempt to convert change values to float for formatting
    try:
        change_float = float(change)
        change_percent_float = float(change_percent)
    except ValueError:
        change_float = 0
        change_percent_float = 0

    # Format the change string with absolute and percentage changes
    delta = f"{change_float:.2f} ({change_percent_float})%"

    # Determine delta_color based on the value of change
    if change_float > 0:
        delta_color = "inverse"  # Green
    elif change_float < 0:
        delta_color = "normal"  # Red
    else:
        delta_color = "off"  # Black or default

    st.metric(label="最新价", value=f"¥ {latest_price}", delta=delta, delta_color=delta_color)


def display_market_quotes(stock_symbol, df):
    """
    Display the fetched market quotes data and visualize it.
    
    Args:
        stock_symbol (str): Stock symbol with market prefix.
        df (pd.DataFrame): Bid-ask data DataFrame.
    """
    if df.empty:
        st.warning("没有可显示的数据。")
        return
        
    # Extract key metrics
    metrics_items = [
        "最新", "均价", "涨幅", "涨跌", "总手", "金额", 
        "换手", "量比", "最高", "最低", "今开", "昨收", 
        "涨停", "跌停", "外盘", "内盘"
    ]
    metrics = df[df['item'].isin(metrics_items)].set_index('item')['value'].to_dict()
    
    # Display Latest Price and Change at the Top
    display_latest_price_and_change(metrics)
    
    # ---------------------------
    # Candlestick Chart Section
    # ---------------------------

    # Fetch historical stock prices using the get_stock_prices function
    historical_data = get_stock_prices(ticker=stock_symbol)

    if historical_data is not None and not historical_data.empty:
        # Create a Plotly Candlestick chart
        fig_candlestick = go.Figure(data=[go.Candlestick(
            x=historical_data['date'],
            open=historical_data['open'],
            high=historical_data['high'],
            low=historical_data['low'],
            close=historical_data['close'],
            name='Candlestick'
        )])

        fig_candlestick.update_layout(
            title=f"{stock_symbol} 的K线图",
            xaxis_title='日期',
            yaxis_title='价格 (¥)',
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )

        st.plotly_chart(fig_candlestick, use_container_width=True)
    else:
        st.warning("没有可用的历史价格数据来绘制K线图。")

    
    st.markdown("---")

    # Extract buy and sell data
    buy_df = df[df['item'].str.startswith('buy')].copy()
    sell_df = df[df['item'].str.startswith('sell')].copy()
    
    # Extract levels
    buy_df['Level'] = buy_df['item'].str.extract(r'buy[_]?(\d+)').astype(int)
    sell_df['Level'] = sell_df['item'].str.extract(r'sell[_]?(\d+)').astype(int)
    
    # Separate prices and volumes
    buy_prices = buy_df[~buy_df['item'].str.contains('vol')][['Level', 'value']]
    buy_volumes = buy_df[buy_df['item'].str.contains('vol')][['Level', 'value']]
    sell_prices = sell_df[~sell_df['item'].str.contains('vol')][['Level', 'value']]
    sell_volumes = sell_df[sell_df['item'].str.contains('vol')][['Level', 'value']]
    
    # Rename columns for merging
    buy_prices.rename(columns={'value': '价格'}, inplace=True)
    buy_volumes.rename(columns={'value': '成交量'}, inplace=True)
    sell_prices.rename(columns={'value': '价格'}, inplace=True)
    sell_volumes.rename(columns={'value': '成交量'}, inplace=True)
    
    # Merge buy and sell data on Level
    buy_merged = pd.merge(buy_prices, buy_volumes, on='Level')
    sell_merged = pd.merge(sell_prices, sell_volumes, on='Level')
    
    # Sort levels
    buy_merged.sort_values('Level', ascending=False, inplace=True)
    sell_merged.sort_values('Level', ascending=True, inplace=True)
    
    # Visualization: Buy and Sell Depth
    st.markdown("### 买卖盘深度")
    fig = go.Figure()
    
    # Add Buy Orders
    fig.add_trace(go.Bar(
        x=buy_merged['价格'],
        y=buy_merged['成交量'],
        name='买单量',
        marker_color='green',
        opacity=0.6
    ))
    
    # Add Sell Orders
    fig.add_trace(go.Bar(
        x=sell_merged['价格'],
        y=sell_merged['成交量'],
        name='卖单量',
        marker_color='red',
        opacity=0.6
    ))
    
    fig.update_layout(
        barmode='overlay',
        title='买卖盘价格与成交量',
        xaxis_title='价格',
        yaxis_title='成交量',
        legend=dict(x=0.7, y=1.0),
        template='plotly_white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Visualization: Volume Distribution Pie Chart
    st.markdown("### 成交量分布")
    volume_df = pd.DataFrame({
        '类型': ['买单量', '卖单量'],
        '总成交量': [buy_merged['成交量'].sum(), sell_merged['成交量'].sum()]
    })
    
    fig2 = px.pie(
        volume_df, 
        names='类型', 
        values='总成交量', 
        title='买卖单总成交量分布', 
        hole=0.3,
        color='类型',
        color_discrete_map={'买单量':'green', '卖单量':'red'}
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Display Detailed Buy and Sell Data
    st.markdown("### 买卖盘详细数据")
    
    combined_df = pd.merge(
        buy_merged, 
        sell_merged, 
        on='Level', 
        how='outer', 
        suffixes=('_买', '_卖')
    ).fillna(0)
    
    combined_df.sort_values('Level', ascending=False, inplace=True)
    
    # Rename columns for clarity
    combined_df = combined_df.rename(columns={
        'Level': '层级',
        '价格_买': '买价格',
        '成交量_买': '买成交量',
        '价格_卖': '卖价格',
        '成交量_卖': '卖成交量'
    })
    
    # Prepare data for Plotly Grouped Bar Chart
    bar_df = combined_df.copy()
    
    # Ensure '层级' is treated as a categorical variable for ordering
    bar_df['层级'] = bar_df['层级'].astype(int)
    bar_df = bar_df.sort_values('层级', ascending=False)
    
    # Create Plotly Grouped Bar Chart
    fig_table = go.Figure()

    fig_table.add_trace(go.Bar(
        x=bar_df['层级'],
        y=bar_df['买成交量'],
        name='买成交量',
        marker_color='green'
    ))

    fig_table.add_trace(go.Bar(
        x=bar_df['层级'],
        y=bar_df['卖成交量'],
        name='卖成交量',
        marker_color='red'
    ))

    fig_table.update_layout(
        barmode='group',
        title='详细买卖盘成交量分布',
        xaxis_title='层级',
        yaxis_title='成交量',
        legend_title='类型',
        template='plotly_white',
        hovermode='closest'
    )

    st.plotly_chart(fig_table, use_container_width=True)

    # Display Detailed Buy and Sell Data
    st.markdown("### 买卖盘详细数据")
    
    # Merge the buy and sell DataFrames
    combined_df = pd.merge(
        buy_merged, 
        sell_merged, 
        on='Level', 
        how='outer', 
        suffixes=('_买', '_卖')
    ).fillna(0)
    
    # Sort the DataFrame by 'Level' in descending order
    combined_df.sort_values('Level', ascending=False, inplace=True)
    
    # Create a copy for display purposes
    display_df = combined_df.copy()
    
    # Columns to format
    format_columns = ['价格_买', '成交量_买', '价格_卖', '成交量_卖']
    
    # Apply formatting: format numbers and replace 0 with '-'
    for col in format_columns:
        display_df[col] = display_df[col].apply(
            lambda x: f"{float(x):,.2f}" if isinstance(x, (int, float)) and x != 0 else '-'
        )
    
    # Rename columns for display
    display_df.rename(columns={
        'Level': '层级',
        '价格_买': '买价格',
        '成交量_买': '买成交量',
        '价格_卖': '卖价格',
        '成交量_卖': '卖成交量'
    }, inplace=True)
    
    # Display the formatted DataFrame
    st.dataframe(display_df, width=800, hide_index=True, use_container_width=True)

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['指标', '数值'])

    # Display key metrics in columns
    st.markdown("#### 关键指标")
    num_cols = 4
    metric_cols = st.columns(num_cols)
    for idx, (key, value) in enumerate(metrics.items()):
        col = metric_cols[idx % num_cols]
        with col:
            # Format numbers with commas and appropriate decimal places
            try:
                numeric_value = float(value)
                if '幅' in key:
                    display_value = f"{numeric_value:.3}%"
                elif '金额' in key:
                    display_value = f"¥{int(numeric_value):,}"
                else:
                    display_value = f"{numeric_value:,.2f}"
            except ValueError:
                display_value = value
            st.metric(label=key, value=display_value)

def main():
    st.title("个股行情 📊")

    st.logo(
        "images/logo.png",
        link="https://platform.tidefast.com",
        size="large", 
        icon_image="images/logo.png",
    )

    # ---------------------------
    # Selection Bar (Fixed at Top)
    # ---------------------------
    selection_container = st.container()
    with selection_container:

        selected_stock_display = st.selectbox(
            "搜索并选择股票",
            ["请选择一个股票"] + stock_display_names,
            key='selected_stock_display',
            help="输入股票名称或代码以搜索并选择股票",
        )

        if selected_stock_display != "请选择一个股票":
            selected_stock = stock_display_to_ticker[selected_stock_display]
            st.session_state.selected_stock = selected_stock
        else:
            st.session_state.selected_stock = None

            # Define popular stocks (subset of stocks)
    popular_stocks = [
        {"cname": "中国中免", "ticker": "601888.SH", "market": "china"},
        {"cname": "恒生电子", "ticker": "600570.SH", "market": "china"},
        {"cname": "甘李药业", "ticker": "603087.SH", "market": "china"},
        {"cname": "长电科技", "ticker": "600584.SH", "market": "china"},
        {"cname": "达仁堂", "ticker": "600329.SH", "market": "china"},
        {"cname": "以岭药业", "ticker": "002603.SZ", "market": "china"},
        {"cname": "泸州老窖", "ticker": "000568.SZ", "market": "china"},
        {"cname": "晶合集成", "ticker": "688249.SH", "market": "china"}
    ]
    
    # Popular Stocks Buttons
    button_cols = st.columns(4)  # Create a column for each popular stock

    for idx, stock in enumerate(popular_stocks):
        with button_cols[idx % 4]:
            stock_display = f"{stock['cname']} ({stock['ticker']})"
            if st.button(stock_display, key=f"button_{stock['ticker']}"):
                st.session_state.selected_stock = stock_display_to_ticker[stock_display]
    
    if st.session_state.selected_stock != None:
        st.success(f"已选择股票: {st.session_state.selected_stock}")

    # ---------------------------
    # Display Market Quotes Automatically
    # ---------------------------
    if st.session_state.selected_stock:
        with st.spinner('正在获取行情报价数据...'):
            df_quotes = fetch_market_quotes(st.session_state.selected_stock)
        display_market_quotes(st.session_state.selected_stock, df_quotes)

if __name__ == "__main__":
    main()