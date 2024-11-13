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

    if 'name' not in st.session_state:
        st.session_state['name'] = 'John Doe'

    st.header(st.session_state['name'])

    if st.button('Jane'):
        st.session_state['name'] = 'Jane Doe'

    if st.button('John'):
        st.session_state['name'] = 'John Doe'

    st.header(st.session_state['name'])

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
        # Instructions for developers: Make sure the OpenAI API key is set correctly in the environment variables
        # and that the 'openai_model' session state is initialized before starting the conversation.
        
        # Initialize the OpenAI client and model
        client = OpenAI(api_key=openai_key)

        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-4o-mini"

        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.markdown("""
            **使用提示：**
            - 请在下方的输入框中输入您的问题或想要了解的投资信息。
            - 历澜AI助手将为您提供实时解答，但请记住，这些建议仅供参考，不构成投资建议。
            - 使用前，请先阅读并同意免责声明，以确保您了解使用本助手的相关风险。
            - 随时开始对话，AI助手会尽力为您提供有用的信息！
            """)
            
        # Display all chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle new user input
        if prompt := st.chat_input("Hi！我是历澜AI助手？有什么问题能帮到您吗？"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Generate a response from OpenAI's GPT model and display it
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