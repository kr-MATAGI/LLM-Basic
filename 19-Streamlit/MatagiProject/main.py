import streamlit as st
from langchain_core.messages.chat import ChatMessage

# 타이틀
st.title("나의 ChatGPT 테스트")


# 함수
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_messages():
    for chat_msg in st.session_state["messages"]:
        st.chat_message(chat_msg.role).write(chat_msg.content)


# 대화 기록을 저장해주는 용도
# 페이지가 새로고침이 일어나도 데이터가 지워지지 않는다.
if "messages" not in st.session_state:  # 처음 한 번만 실행되게
    st.session_state["messages"] = []
else:
    print_messages()


# 사용자 질의
user_input = st.chat_input("무엇이든 물어보세요!")
if user_input:
    # 웹에 대화를 출력
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        st.write(user_input)

    # 대회 기록을 저장
    add_message("user", user_input)
    add_message("assistant", user_input)
