import streamlit as st
import datetime
import pandas as pd
import plotly.express as px

# Set the page configuration
st.set_page_config(
    page_title="历澜指数 - 历澜投资",
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

# Logo and Header
st.title("历澜指数™")

# Current Date and Time (Asia/Shanghai Timezone)
st.markdown(f"**{datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')}**（北京时间）")

# Top Holdings
st.markdown("###### 历澜指数™ - 持仓明细")
top_holdings = pd.DataFrame({
    "股票名称": ["李宁", "中芯国际", "小米集团", "达仁堂", "康方生物" ],
    "持仓比例": [25, 20, 15, 10, 15]
})
st.table(top_holdings)

# Create a DataFrame
portfolio_df = pd.DataFrame(top_holdings)

# Generate a pie chart using Plotly
fig = px.pie(portfolio_df, names="股票名称", values="持仓比例", title="历澜指数™ - 资产分布")

# Display the pie chart in Streamlit
st.plotly_chart(fig)

# # Sector Breakdown (Pie Chart)
# st.markdown("### 行业分布")
# sector_breakdown = pd.DataFrame({
#     "行业": ["科技", "金融", "消费"],
#     "比例": [40, 30, 30]
# })
# st.bar_chart(sector_breakdown.set_index("行业"))

# # Performance Chart (Line Chart)
# st.markdown("### 历史表现")
# performance_data = pd.DataFrame({
#     "日期": pd.date_range(start="2023-01-01", periods=10, freq="ME"),
#     "指数值": [1000, 1010, 1025, 1015, 1030, 1040, 1025, 1050, 1075, 1025]
# }).set_index("日期")
# st.line_chart(performance_data)


# Footer
st.markdown("---")
st.write("###### 风险免责声明")
st.write(
    "本页面所提供的历澜指数及其相关信息仅供参考，"
    "不构成投资建议或要约。投资涉及风险，"
    "包括可能损失本金。投资者应根据自身的投资目标、"
    "财务状况和风险承受能力独立评估并作出决策。"
    "历澜投资及其关联方对因使用本页面内容所导致的任何损失不承担责任。"
)
st.write("联系: tidefast@protonmail.com")

# Customize the styling
st.markdown(
    """
    <style>
    .css-1d391kg { font-size: 16px; }
    .stButton>button { background-color: lightgrey; }
    </style>
    """,
    unsafe_allow_html=True
)