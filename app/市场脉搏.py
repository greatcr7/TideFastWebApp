import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from market.account_number import display_stock_account_statistics
from market.below_net_asset import display_below_net_asset_statistics
from market.board_change import display_stock_board_change
from market.cctv_news import display_news_cctv
from market.equity_bond_spread import display_equity_bond_spread
from market.market_congestion import display_market_congestion
from market.stock_change import display_stock_changes

# Set the page configuration
st.set_page_config(
    page_title="市场脉搏 - 历澜投资",
    layout="wide",
    page_icon="images/logo.png",
    initial_sidebar_state="expanded",
)

st.logo(
    "images/logo.png",
    link="https://platform.tidefast.com",
    size="large", 
    icon_image="images/logo.png",
)

# Sidebar - Dataset Selection
st.sidebar.header("市场脉搏")
dataset_choice = st.sidebar.radio(
    "请选择要查看的数据集",
    ("概念板块异动", "股票盘口异动", "新闻联播全文", "股债利差", "股票账户数", "大盘拥挤度", "破净股统计",)
)

if dataset_choice == "大盘拥挤度":
    display_market_congestion()

elif dataset_choice == "股债利差":
    display_equity_bond_spread()

elif dataset_choice == "破净股统计":
    display_below_net_asset_statistics()

elif dataset_choice == "新闻联播全文":
    display_news_cctv()

elif dataset_choice == "股票盘口异动":
    display_stock_changes()

elif dataset_choice == "概念板块异动":
    display_stock_board_change()

elif dataset_choice == "股票账户数":
    display_stock_account_statistics()

