import streamlit as st
import akshare as ak
import pandas as pd
from datetime import datetime
import plotly.express as px
import json
import plotly.graph_objs as go

@st.cache_data
def fetch_stock_board_change_data() -> pd.DataFrame:
    """
    使用 AkShare 获取当日板块异动详情数据。
    
    Returns:
        pd.DataFrame: 包含板块名称、涨跌幅、主力净流入、板块异动总次数、
                      板块异动最频繁个股及所属类型-股票代码、名称、买卖方向、
                      以及板块具体异动类型列表及出现次数的 DataFrame。
    """
    try:
        df = ak.stock_board_change_em()
        return df
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

def parse_board_specific_changes(row):
    """
    解析 '板块具体异动类型列表及出现次数' 列，将其转换为易读的格式。
    
    Args:
        row (pd.Series): DataFrame 的一行数据。
    
    Returns:
        str: 格式化的板块具体异动类型及出现次数。
    """
    try:
        changes = row['板块具体异动类型列表及出现次数']
        if isinstance(changes, str):
            changes = json.loads(changes.replace("'", '"'))  # 将字符串转换为列表
        formatted_changes = ""
        for change in changes:
            formatted_changes += f"类型代码: {change.get('t')}, 次数: {change.get('ct')}\n"
        return formatted_changes
    except Exception:
        return "无法解析数据"
    

def display_stock_board_change():
    """
    在 Streamlit 中获取、过滤并展示板块异动详情数据。
    包括数据表展示、数据统计、可视化图表以及数据导出功能。
    """
    st.title("概念板块异动")
    st.markdown("""
    您可以在左边的输入栏选择想查看的板块。
    """)
    
    # 侧边栏 - 信息展示
    st.sidebar.header("信息")
    latest_date = datetime.today().date()
    st.sidebar.markdown(f"**最新交易日:** {latest_date}")
    
    # 侧边栏 - 过滤选项
    st.sidebar.header("过滤选项")
    search_term = st.sidebar.text_input("搜索板块名称", "")
    
    # 获取数据
    with st.spinner("正在获取数据..."):
        df = fetch_stock_board_change_data()
    
    if df.empty:
        st.warning("没有找到当日的板块异动数据。")
        return
    
    # 解析 '板块具体异动类型列表及出现次数'
    df['板块具体异动类型及次数'] = df.apply(parse_board_specific_changes, axis=1)
    
    # 转换数据类型
    for col in ['涨跌幅', '主力净流入', '板块异动总次数']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 过滤数据
    if search_term:
        df = df[df['板块名称'].str.contains(search_term)]
        st.sidebar.info(f"过滤后找到 {len(df)} 个板块的异动记录。")
    
    st.success(f"找到 {len(df)} 个板块的异动记录。")
        
    # 数据统计
    st.markdown("### 数据统计")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_boards = len(df)
        st.metric("板块总数", f"{total_boards}")
    with col2:
        avg_change = df['涨跌幅'].mean()
        st.metric("平均涨跌幅", f"{avg_change:.2f}%")
    with col3:
        total_net_flow = df['主力净流入'].sum()
        st.metric("主力净流入总额", f"{total_net_flow / 10000:.2f} 亿元")    


    # 可视化: 板块涨跌幅前20
    st.markdown("### 板块涨跌幅前20")
    # 选择涨跌幅前20的板块
    top20_change = df.nlargest(20, '涨跌幅').copy()
    # 确保 '涨跌幅' 数据格式正确，保留两位小数
    top20_change['涨跌幅_百分比'] = top20_change['涨跌幅'].round(2)
    # 创建水平条形图
    fig_change = px.bar(
        top20_change.sort_values('涨跌幅_百分比'),
        x='涨跌幅_百分比',  # 使用格式化后的涨跌幅
        y='板块名称',
        orientation='h',
        labels={
            '涨跌幅_百分比': '涨跌幅 (%)',  # 更新标签
            '板块名称': '板块名称'
        },
        title='涨跌幅前20的板块',
        color='涨跌幅_百分比',  # 使用格式化后的涨跌幅进行颜色映射
        color_continuous_scale='RdYlGn',
    )
    # 优化布局（可选）
    fig_change.update_layout(
        xaxis_title='涨跌幅 (%)',
        yaxis_title='板块名称',
        yaxis=dict(categoryorder='total ascending'),  # 按涨跌幅从低到高排序
        margin=dict(l=150, r=20, t=50, b=20)
    )
    # 显示图表
    st.plotly_chart(fig_change, use_container_width=True, config={'displayModeBar': False})    


    # 可视化: 主力净流入前20
    st.markdown("### 主力净流入前20")

    # 选取主力净流入前20的板块
    top20_net_flow = df.nlargest(20, '主力净流入').copy()

    # 将 '主力净流入' 从 万元 转换为 亿元
    top20_net_flow['主力净流入_亿元'] = top20_net_flow['主力净流入'] / 10000

    # 创建水平条形图
    fig_net_flow = px.bar(
        top20_net_flow.sort_values('主力净流入_亿元'),
        x='主力净流入_亿元',  # 使用转换后的单位
        y='板块名称',
        orientation='h',
        labels={
            '主力净流入_亿元': '主力净流入 (亿元)',  # 更新标签单位
            '板块名称': '板块名称'
        },
        title='主力净流入前20的板块',
        color='主力净流入_亿元',  # 使用转换后的单位进行颜色映射
        color_continuous_scale='Blues',
    )

    # 更新布局以优化显示效果（可选）
    fig_net_flow.update_layout(
        xaxis_title='主力净流入 (亿元)',
        yaxis_title='板块名称',
        yaxis=dict(categoryorder='total ascending'),  # 按净流入从低到高排序
        margin=dict(l=100, r=20, t=50, b=20)
    )

    # 显示图表
    st.plotly_chart(fig_net_flow, use_container_width=True, config={'displayModeBar': False})    

    st.markdown("### 板块异动总次数前20")
    # 选择板块异动总次数前20的板块
    top20_change_times = df.nlargest(20, '板块异动总次数').copy()

    # 确保 '板块异动总次数' 为整数，并进行必要的格式化
    top20_change_times['板块异动总次数'] = top20_change_times['板块异动总次数'].astype(int)

    # 创建水平条形图
    fig_change_times = px.bar(
        top20_change_times.sort_values('板块异动总次数'),
        x='板块异动总次数',
        y='板块名称',
        orientation='h',
        labels={
            '板块异动总次数': '异动次数',
            '板块名称': '板块名称'
        },
        title='板块异动总次数前20',
        color='板块异动总次数',
        color_continuous_scale='Oranges',
    )

    # 优化布局（可选）
    fig_change_times.update_layout(
        xaxis_title='异动次数',
        yaxis_title='板块名称',
        yaxis=dict(categoryorder='total ascending'),  # 按异动次数从低到高排序
        margin=dict(l=150, r=20, t=50, b=20)
    )

    # 显示图表
    st.plotly_chart(fig_change_times, use_container_width=True, config={'displayModeBar': False})

    # 显示数据表
    st.markdown("### 板块异动数据表")
    st.dataframe(df[['板块名称', '涨跌幅', '主力净流入', '板块异动总次数']].reset_index(drop=True), use_container_width=True)
