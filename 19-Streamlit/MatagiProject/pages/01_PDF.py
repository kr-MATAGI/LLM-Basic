import streamlit as st
import glob
import os
from dotenv import load_dotenv

from langchain_core.messages.chat import ChatMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt

# 초기 셋팅
load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")



# 타이틀
st.title("PDF 기반 QA")


# 함수
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_messages():
    for chat_msg in st.session_state["messages"]:
        st.chat_message(chat_msg.role).write(chat_msg.content)


def create_chain(prompt_filepath):
    """
    체인 생성
    """
    prompt = load_prompt(prompt_filepath, encoding="utf-8")

    # GPT
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Parser
    output_parser = StrOutputParser()

    # Chain
    chain = prompt | llm | output_parser

    return chain

# 파일을 캐시 저장 (시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장한다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)


# 사이드 바
with st.sidebar:
    # 초기화 버튼
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # 프롬프트 선택
    selected_prompt = "prompts/pdf-rag.yaml"


# 대화 기록을 저장해주는 용도
# 페이지가 새로고침이 일어나도 데이터가 지워지지 않는다.
if "messages" not in st.session_state:  # 처음 한 번만 실행되게
    st.session_state["messages"] = []
else:
    if clear_btn:
        st.session_state["messages"] = []
    print_messages()

# 파일이 업로드 되었을 때
if uploaded_file:
    embed_file(uploaded_file)


# 사용자 질의
user_input = st.chat_input("무엇이든 물어보세요!")
if user_input:
    # 웹에 대화를 출력
    with st.chat_message("user"):
        st.write(user_input)
    # chain을 생성
    chain = create_chain(prompt_filepath=selected_prompt)

    # 한 번에 출력
    # ai_answer = chain.invoke({"question": user_input})

    # chatGPT처럼 출력 (streaming)
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
