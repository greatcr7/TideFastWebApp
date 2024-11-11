import json
import streamlit as st
from indicators.emd_macd import emd_macd_analysis
from indicators.fib_retracement import fibonacci_retracement_analysis
from indicators.alma import alma_analysis
from indicators.bollinger import bollinger_band_analysis
from indicators.choppiness import choppiness_analysis
from indicators.cks import cks_analysis
from indicators.cmf import cmf_analysis
from indicators.hurst import hurst_analysis
from indicators.ichimoku import ichimoku_analysis
from indicators.kama import kama_analysis
from indicators.kc import keltner_channel_analysis
from indicators.parabolic_sar import parabolic_sar_analysis
from indicators.rsi import rsi_analysis
from indicators.stc import stc_analysis
from indicators.supertrend import supertrend_analysis
# from indicators.financials import financials_analysis
# from indicators.ml_predictions import ml_predictions_analysis

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="技术分析 - 历澜投资",
    layout="wide",
    page_icon="images/logo.png"
)

# ---------------------------
# Custom CSS for Button Styling
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
        background-color: transparent; /* Transparent background */
        border: 2px solid 
        border-radius: 12px; /* Rounded corners */
        transition: background-color 0.3s, color 0.3s; /* Smooth transition */
        cursor: pointer;
    }
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ---------------------------
# Initialize Session State
# ---------------------------
if 'selected_indicator' not in st.session_state:
    st.session_state.selected_indicator = None
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None

# ---------------------------
# Callback Functions
# ---------------------------

def set_selected_indicator(indicator):
    st.session_state.selected_indicator = indicator

def set_selected_indicator_dropdown():
    selected_indicator = st.session_state.dropdown_selection
    if selected_indicator != "请选择一个指标":
        st.session_state.selected_indicator = selected_indicator

def indicator_return_home():
    st.session_state.selected_indicator = None
    st.session_state.selected_stock = None
    st.session_state.selected_stock_display = "请选择一个股票"
    st.session_state.dropdown_selection = "请选择一个指标"

# Load stock data from JSON
def load_stock_data(json_path='stocks.json'):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            stocks = json.load(f)
        return stocks
    except FileNotFoundError:
        st.error(f"股票数据文件未找到: {json_path}")
        return []
    except json.JSONDecodeError:
        st.error("股票数据文件格式错误。")
        return []

stocks = load_stock_data()

# Create a mapping from display name to ticker
stock_display_to_ticker = {f"{stock['cname']} ({stock['ticker']})": stock['ticker'] for stock in stocks}
stock_display_names = list(stock_display_to_ticker.keys())

# ---------------------------
# Indicator Mapping
# ---------------------------
page_names_to_funcs = {
    "EMD-MACD均线 💰": emd_macd_analysis,
    "RSI指标 📈": rsi_analysis,
    "KAMA均线 💼": kama_analysis,
    "Hurst指标 📊": hurst_analysis,
    "超级趋势 📈": supertrend_analysis,
    "斐波那契回调线 💰": fibonacci_retracement_analysis,
    "布林带指标 📊": bollinger_band_analysis,
    "波动指数 💼": choppiness_analysis,
    "蔡金资金流量 💰": cmf_analysis,
    "抛物线转向指标 📈": parabolic_sar_analysis,
    "一目云均衡图 💼": ichimoku_analysis,
    "肯特纳通道 📊": keltner_channel_analysis, 
    "CKS指数 📈": cks_analysis,
    "ALMA移动平均数 📊": alma_analysis,
    "沙夫趋势周期 💼": stc_analysis,
}

# ---------------------------
# Home Page Function (Recommended Indicators Grid)
# ---------------------------
def home():
    # st.markdown("### 推荐指标")
    categories = {
        "技术指标 🔍": ["RSI指标 📈", "KAMA均线 📉", "MACD指标 📊", "布林带指标 📈"],
        "基本面分析 💼": ["财务数据分析 💼", "盈利能力分析 💰", "市盈率分析 📉", "资产负债分析 📊"],
        "机器学习预测 🤖": ["股票价格预测 🤖", "趋势分析 📈", "风险评估 🔍", "波动率预测 📉"]
    }
    # for category, indicators in categories.items():
    #     st.markdown(f"#### {category}")
    #     indicators_per_row = 4
    #     for i in range(0, len(indicators), indicators_per_row):
    #         cols = st.columns(indicators_per_row)
    #         row_indicators = indicators[i:i + indicators_per_row]
    #         for col, indicator in zip(cols, row_indicators):
    #             with col:
    #                 st.button(indicator, key=indicator, on_click=set_selected_indicator_dropdown)



# ---------------------------
# Main App Execution
# ---------------------------
def main():
    st.title("技术分析 📈")
    
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
        selection_cols = st.columns(2)

        with selection_cols[0]:
            selected_stock_display = st.selectbox(
                "搜索并选择股票",
                ["请选择一个股票"] + stock_display_names,
                key='selected_stock_display',
                help="输入股票名称或代码以搜索并选择股票",
                on_change=set_selected_indicator_dropdown,
            )

            if selected_stock_display != "请选择一个股票":
                selected_stock = stock_display_to_ticker[selected_stock_display]
                st.session_state.selected_stock = selected_stock
                st.success(f"已选择股票: {selected_stock_display}")
            else:
                st.session_state.selected_stock = None

        with selection_cols[1]:
            dropdown_selection = st.selectbox(
                "选择指标",
                ["请选择一个指标"] + list(page_names_to_funcs.keys()),
                key='dropdown_selection',
                on_change=set_selected_indicator_dropdown,
                help="在此输入并选择您想要查看的指标"
            )

            if st.session_state.selected_indicator:
                st.success(f"已选择指标: {st.session_state.selected_indicator}")

    st.markdown("---")  # Separator

    # ---------------------------
    # Display Content Below Selection Bar
    # ---------------------------
    if st.session_state.selected_indicator:
        if st.session_state.selected_stock:
            indicator = st.session_state.selected_indicator
            stock = st.session_state.selected_stock
            analysis_func = page_names_to_funcs.get(indicator)

            if analysis_func:
                analysis_func(stock)
            else:
                st.error("所选指标的分析功能尚未实现。")
                # if st.button("返回主页"):
                #     indicator_return_home()
        else:
            st.warning("请先选择一个股票，然后再选择一个指标进行分析。")
            # if st.button("返回主页"):
            #     indicator_return_home()
    else:
        home()

main()