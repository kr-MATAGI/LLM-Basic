import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages.chat import ChatMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# 초기 셋팅
load_dotenv()

# 타이틀
st.title("나의 ChatGPT 테스트")


# 함수
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_messages():
    for chat_msg in st.session_state["messages"]:
        st.chat_message(chat_msg.role).write(chat_msg.content)


def create_chain():
    """
    체인 생성
    """
    # prompt | llm | output_parser
    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절한 AI 상담원 입니다."),
            ("user", "#Question:\n{question}"),
        ]
    )

    # GPT
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Parser
    output_parser = StrOutputParser()

    # Chain
    chain = prompt | llm | output_parser

    return chain


# 사이드 바
with st.sidebar:
    clear_btn = st.button("대화 초기화")


# 대화 기록을 저장해주는 용도
# 페이지가 새로고침이 일어나도 데이터가 지워지지 않는다.
if "messages" not in st.session_state:  # 처음 한 번만 실행되게
    st.session_state["messages"] = []
else:
    if clear_btn:
        st.session_state["messages"] = []
    print_messages()

# 사용자 질의
user_input = st.chat_input("무엇이든 물어보세요!")
if user_input:
    # 웹에 대화를 출력
    with st.chat_message("user"):
        st.write(user_input)
    # chain을 생성
    chain = create_chain()

    # 한 번에 출력
    # ai_answer = chain.invoke({"question": user_input})

    # chatGPT처럼 출력
    ai_response = chain.stream({"question": user_input})
    ai_answer = ""
    with st.chat_message("assistant"):
        # 빈 공간(컨터에너)을 만들어서, 여기에 토큰 스트리밍을 출력하낟.
        container = st.empty()

        for token in ai_response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대회 기록을 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)
