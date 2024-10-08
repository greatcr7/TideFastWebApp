import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from market.below_net_asset import display_below_net_asset_statistics
from market.board_change import display_stock_board_change
from market.cctv_news import display_news_cctv
from market.equity_bond_spread import display_equity_bond_spread
from market.market_congestion import display_market_congestion
from market.stock_change import display_stock_changes
from market.valuation import display_valuation_statistics

# Set the page configuration
st.set_page_config(
    page_title="首页",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("首页")

# Sidebar - Dataset Selection
st.sidebar.header("首页")
dataset_choice = st.sidebar.radio(
    "请选择要查看的数据集",
    ("股债利差", "大盘拥挤度", "破净股统计", "估值水平", "新闻联播全文", "盘口异动", "板块异动")
)


if dataset_choice == "大盘拥挤度":
    display_market_congestion()

elif dataset_choice == "股债利差":
    display_equity_bond_spread()

elif dataset_choice == "破净股统计":
    display_below_net_asset_statistics()

elif dataset_choice == "估值水平":
    display_valuation_statistics()

elif dataset_choice == "新闻联播全文":
    display_news_cctv()

elif dataset_choice == "盘口异动":
    display_stock_changes()

elif dataset_choice == "板块异动":
    display_stock_board_change()