import streamlit as st
from indicators.rsi import rsi_analysis

# Set the page configuration
st.set_page_config(page_title="历澜投资", layout="wide", page_icon="images/logo.png")

# Sidebar for logo and demo selection
logo_path = "images/logo.png"  # Update with your actual logo path

# Display the logo at 25% width of the sidebar
sidebar_width = 500  # Estimated sidebar width (in pixels)
logo_width = sidebar_width * 0.1  # Set logo width to 25% of the sidebar width

st.sidebar.image(logo_path, width=int(logo_width))  # Set the image width to 25% of the sidebar width

# ---------------------------
# Home Page
# ---------------------------

def home():
    st.write("# 欢迎来到历澜投资! 👋")
    st.sidebar.success("☝️ 试试看你的股票！")

    st.markdown(
        """
        历澜投资致力于为交易者提供全面的股票分析工具。通过结合**技术指标**、**基本面分析**和**机器学习模型**，帮助您做出更明智的投资决策。

        ### 我们的功能

        - **技术指标分析**：利用如相对强弱指数（RSI）、KAMA均线等技术指标评估股票的买卖信号。
        - **基本面分析**：深入查看公司的财务数据、盈利能力和市场表现，了解其内在价值。
        - **机器学习预测**：应用先进的机器学习算法预测股票价格走势，提升分析的准确性。

        ### 如何使用

        从左侧的下拉菜单中选择您感兴趣的分析工具，开始探索股票数据的不同方面。无论您是经验丰富的交易者还是刚入门的新手，历澜投资都能为您提供有价值的洞见。

        ### 资源与支持

        - 了解更多关于我们的信息，请访问 [历澜投资官网](https://tidefast.com)

        ---
        ### Legal Disclaimer

        **This platform is intended for informational purposes only and does not constitute investment advice. Always conduct your own research or consult with a qualified financial advisor before making any investment decisions.**

        ---
        ### 法律免责声明

        **本平台仅供参考，並不构成投资建议。请在做出任何投资决定之前，务必自行研究或咨询专业的金融顾问。**
        """
    )

# ---------------------------
# Mapping Demos to Names
# ---------------------------

page_names_to_funcs = {
    "首页": home,           # Added Home option
    "RSI指标": rsi_analysis  # RSI Analysis
}

# ---------------------------
# Render Selected Indicator
# ---------------------------

indicator_name = st.sidebar.selectbox("选择指标", page_names_to_funcs.keys())
page_names_to_funcs[indicator_name]()