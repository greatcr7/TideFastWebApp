import streamlit as st
import hmac

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        st.title("欢迎来到历澜投资")
        st.text("请输入您的账号信息")
        
        with st.form("Credentials"):
            st.text_input("用户名", key="username")
            st.text_input("密码", type="password", key="password") 
            st.form_submit_button("登录", on_click=password_entered)


    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 用户名/密码错误。请再试一遍。")
    return False


if not check_password():
    st.stop()


stock_move = st.Page(
    "page/个股行情.py", title="个股行情", icon=":material/candlestick_chart:"
)
market_move = st.Page("page/市场脉搏.py", title="市场脉搏", icon=":material/vital_signs:")
technical_analysis = st.Page(
    "page/技术分析.py", title="技术分析", icon=":material/precision_manufacturing:"
)

home = st.Page("page/历澜助手.py", title="历澜助手", icon=":material/robot_2:")
tidefast_index = st.Page("page/历澜指数.py", title="历澜指数™", icon=":material/monitoring:")


pg = st.navigation(
    {
        "历澜投资": [home, tidefast_index],
        "研究": [technical_analysis, market_move, stock_move],
    }
)

pg.run()