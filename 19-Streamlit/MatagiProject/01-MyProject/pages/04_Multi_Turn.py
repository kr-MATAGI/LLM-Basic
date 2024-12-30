import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote.models import MultiModal
from langchain_teddynote import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser


# 초기 셋팅
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] Multi Turn 챗봇")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


# 타이틀
st.title("대화 내용을 기억하는 챗봇")


# 세션 기록을 저장할 딕셔너리
if not "store" in st.session_state:
    st.session_state["store"] = {}

# 함수
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_messages():
    for chat_msg in st.session_state["messages"]:
        st.chat_message(chat_msg.role).write(chat_msg.content)

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
        
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

def create_chain(
    model_name="gpt-4o"
):
    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요.",
            ),
            # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
            MessagesPlaceholder(variable_name="chat_history"), # 대화의 내용이 쌓임. (필수 요소)
            ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용 (필수 요소)
        ]
    )

    # llm 생성
    llm = ChatOpenAI(model_name=model_name)

    # 일반 Chain 생성
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )

    return chain_with_history


# 이미지를 캐시 저장 (시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다...")
def process_image(file):
    # 업로드한 파일을 캐시 디렉토리에 저장한다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # Create Retriever
    return file_path


# 사이드 바
with st.sidebar:
    # 초기화 버튼
    clear_btn = st.button("대화 초기화")

    # LLM 모델 선택
    selected_model = st.selectbox(
        "LLM 선택", 
        ["gpt-4o", "gpt-4o-mini"], 
        index=0
    )

    # 세션 ID를 지정하는 메뉴
    session_id = st.text_input("세션 ID를 입력하세요", "abc123")


# 대화 기록을 저장해주는 용도
# 페이지가 새로고침이 일어나도 데이터가 지워지지 않는다.
if "messages" not in st.session_state:  # 처음 한 번만 실행되게
    st.session_state["messages"] = []
else:
    if clear_btn:
        st.session_state["messages"] = []
    print_messages()


if not "chain" in st.session_state:
    st.session_state["chain"] = create_chain(model_name=selected_model)

# 사용자 질의
warning_msg = st.empty()  # 경고 메시지를 띄우기 위한 빈 영역 생성
user_input = st.chat_input("무엇이든 물어보세요!")

if user_input:
    # 답변 요청   

    # chain을 생성
    chain = st.session_state["chain"]
    if chain:
        response = chain.stream(
            # 질문 입력
            {"question": user_input},
            # 세션 ID 기준으로 대화를 기록합니다.
            config={"configurable": {"session_id": session_id}},
        )

        # 웹에 대화를 출력
        with st.chat_message("user"):
            st.write(user_input)

        # chatGPT처럼 출력 (streaming)
        ai_answer = ""
        with st.chat_message("assistant"):
            # 빈 공간(컨터에너)을 만들어서, 여기에 토큰 스트리밍을 출력하낟.
            container = st.empty()

            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대회 기록을 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else: 
        warning_msg.error("Hi.")