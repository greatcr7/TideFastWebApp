import streamlit as st

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    if st.button("Log in"):
        st.session_state.logged_in = True
        st.rerun()

def logout():
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

stock_move = st.Page(
    "page/个股行情.py", title="个股行情", icon=":material/candlestick_chart:"
)
market_move = st.Page("page/市场脉搏.py", title="市场脉搏", icon=":material/vital_signs:")
technical_analysis = st.Page(
    "page/技术分析.py", title="技术分析", icon=":material/precision_manufacturing:"
)

home = st.Page("page/历澜助手.py", title="历澜助手", icon=":material/robot_2:")
tidefast_index = st.Page("page/历澜指数.py", title="历澜指数™", icon=":material/monitoring:")


if not st.session_state.logged_in:
    pg = st.navigation(
        {
            "历澜投资": [home, tidefast_index],
            "研究": [technical_analysis, market_move, stock_move],
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()