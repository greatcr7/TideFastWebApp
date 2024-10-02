import streamlit as st
import akshare as ak
import pandas as pd
from datetime import datetime
import pytz


@st.cache_data
def fetch_stock_changes_data(symbol: str) -> pd.DataFrame:
    """
    使用 AkShare 获取盘口异动数据。
    
    Args:
        symbol (str): 盘口异动类型，如 "大笔买入"。
        
    Returns:
        pd.DataFrame: 包含时间、代码、名称、板块和相关信息的 DataFrame。
    """
    try:
        df = ak.stock_changes_em(symbol=symbol)
        return df
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

def get_latest_trading_day():
    """
    获取最新的交易日日期。
    
    Returns:
        datetime.date: 最新的交易日日期。
    """
    # 这里可以根据需要实现获取最新交易日的逻辑
    # 暂时返回今天的日期
    return datetime.today().date()

def display_stock_changes():
    """
    在 Streamlit 中获取、过滤并展示盘口异动数据。
    包括 symbol 选择、数据表展示和相关信息解析。
    """
    st.title("盘口异动可视化")
    st.markdown("""
    本页面展示来自东方财富行情中心的盘口异动数据。您可以通过侧边栏选择不同的异动类型来查看最新的交易数据。
    """)
    
    # 侧边栏 - Symbol 选择
    st.sidebar.header("查询参数")
    
    # 定义可选的 symbol 列表
    symbol_options = [
        '火箭发射', '快速反弹', '大笔买入', '封涨停板', '打开跌停板', '有大买盘',
        '竞价上涨', '高开5日线', '向上缺口', '60日新高', '60日大幅上涨', '加速下跌',
        '高台跳水', '大笔卖出', '封跌停板', '打开涨停板', '有大卖盘', '竞价下跌',
        '低开5日线', '向下缺口', '60日新低', '60日大幅下跌'
    ]
    
    # 默认选择 "大笔买入"
    selected_symbol = st.sidebar.selectbox(
        "选择盘口异动类型",
        options=symbol_options,
        index=symbol_options.index("大笔买入") if "大笔买入" in symbol_options else 0
    )
    
    st.sidebar.markdown(f"**选择的盘口异动类型:** {selected_symbol}")
    
    # 获取最新交易日
    latest_date = get_latest_trading_day()
    st.sidebar.markdown(f"**最新交易日:** {latest_date}")
    
    # 获取数据
    with st.spinner("正在获取数据..."):
        changes_data = fetch_stock_changes_data(symbol=selected_symbol)
    
    if changes_data.empty:
        st.warning("没有找到对应的盘口异动数据。")
        return
    
    st.success(f"找到 {len(changes_data)} 条盘口异动记录。")
    
    # 解析 '相关信息' 列，根据 symbol 类型不同可能需要不同的解析方式
    # 这里假设 '相关信息' 为 "数量,价格,涨跌幅" 的格式
    if '相关信息' in changes_data.columns:
        related_info_split = changes_data['相关信息'].str.split(',', expand=True)
        if related_info_split.shape[1] >= 3:
            changes_data['数量'] = related_info_split[0]
            changes_data['价格'] = related_info_split[1]
            changes_data['涨跌幅'] = related_info_split[2]
            changes_data = changes_data.drop(columns=['相关信息'])
        else:
            # 如果格式不符合预期，则保留原始 '相关信息'
            changes_data['相关信息'] = changes_data['相关信息']
    
    # 显示数据表
    with st.expander("查看原始数据"):
        st.dataframe(changes_data)
    
    st.markdown("### 盘口异动详情")
    
    # 展示每条盘口异动记录
    for idx, row in changes_data.iterrows():
        details = f"**时间:** {row['时间']}  \n" \
                  f"**代码:** {row['代码']}  \n" \
                  f"**名称:** {row['名称']}  \n" \
                  f"**板块:** {row['板块']}  \n"
        if '数量' in row and '价格' in row and '涨跌幅' in row:
            details += f"**数量:** {row['数量']}  \n" \
                       f"**价格:** {row['价格']}  \n" \
                       f"**涨跌幅:** {row['涨跌幅']}"
        else:
            details += f"**相关信息:** {row['相关信息']}"
        
        with st.expander(f"{row['名称']} ({row['代码']})"):
            st.markdown(details)
