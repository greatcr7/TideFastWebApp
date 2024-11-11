import pytz
import streamlit as st
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

def fetch_news_cctv_data(date: str) -> pd.DataFrame:
    """
    使用 AkShare 获取新闻联播文字稿数据。
    
    Args:
        date (str): 要获取新闻的日期，格式为 'YYYYMMDD'。
        
    Returns:
        pd.DataFrame: 包含新闻日期、标题和内容的 DataFrame。
    """
    try:
        df = ak.news_cctv(date=date)
        return df
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

def get_latest_available_date():
    """
    获取最新可用的新闻日期，基于北京时间 19:00。
    
    Returns:
        datetime.date: 最新可用的日期。
    """
    # 定义北京时间时区
    bj_timezone = pytz.timezone("Asia/Shanghai")
    now_bj = datetime.now(bj_timezone)
    
    # 如果当前时间晚于或等于19:00，则最新日期为今天
    if now_bj.hour >= 19:
        latest_date = now_bj.date()
    else:
        # 否则，最新日期为昨天
        latest_date = (now_bj - timedelta(days=1)).date()
    return latest_date

def display_news_cctv():
    """
    在 Streamlit 中获取、过滤并展示新闻联播文字稿数据。
    包括日期选择、数据表展示和单条新闻内容展开查看。
    """
    st.title("新闻联播全文")
    st.markdown("""
    您可以通过侧边栏选择日期来查看当天的新闻内容。
    """)
    
    # 侧边栏 - 日期选择
    st.sidebar.header("查询参数")
    
    # 定义日期范围
    min_date = datetime.strptime("20160330", "%Y%m%d").date()
    max_date = get_latest_available_date()
    
    # 日期输入，默认选择最新可用日期
    selected_date = st.sidebar.date_input(
        "选择日期",
        min_value=min_date,
        max_value=max_date,
        value=max_date
    )
    
    # 检查选择的日期是否在有效范围内
    if selected_date < min_date or selected_date > max_date:
        st.sidebar.error(f"请选择 {min_date} 到 {max_date} 之间的日期。")
        return
    
    # 将选择的日期转换为字符串格式 'YYYYMMDD'
    selected_date_str = selected_date.strftime("%Y%m%d")
        
    # 获取数据
    with st.spinner("正在获取数据..."):
        news_data = fetch_news_cctv_data(date=selected_date_str)
    
    if news_data.empty:
        st.warning("没有找到对应日期的新闻数据。")
        return
    
    st.success(f"找到 {len(news_data)} 条新闻记录。")
        
    st.markdown("### 新闻列表")
    
    # 展示每条新闻内容
    for idx, row in news_data.iterrows():
        with st.expander(f"{row['title']}"):
            st.write(row['content'])

    # 可展开查看原始数据
    with st.expander("查看原始数据"):
        st.dataframe(news_data)