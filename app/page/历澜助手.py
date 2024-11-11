import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="首页 - 历澜投资",
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

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

# Create a main navigation section on the page
def home():
    st.title("历澜助手")

    # Display the disclaimer
    if "agreed_to_disclaimer" not in st.session_state:
        st.session_state["agreed_to_disclaimer"] = False

    if not st.session_state["agreed_to_disclaimer"]:
        st.markdown("---")
        st.write("###### 风险免责声明")
        st.write(
            "历澜助手所提供的相关信息仅供参考，"
            "不构成投资建议或要约。投资涉及风险，"
            "包括可能损失本金。投资者应根据自身的投资目标、"
            "财务状况和风险承受能力独立评估并作出决策。"
            "AI有可能犯错，历澜投资及其关联方对因使用本页面内容所导致的任何损失不承担责任。"
        )
        agree_button = st.button("我已阅读并同意免责声明")

        if agree_button:
            st.session_state["agreed_to_disclaimer"] = True
            st.rerun()
    else:
        # Show the chatbot interface only if the user agrees to the disclaimer
        client = OpenAI(api_key=openai_key)

        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-4o-mini"

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Hi！我是历澜AI助手？有什么问题能帮到您吗？"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

home()