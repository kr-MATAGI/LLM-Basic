import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough
from langchain_teddynote import logging

from langchain_ollama import ChatOllama

from retriever import create_retriever

# 초기 셋팅
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] PDF-RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


# 타이틀
st.title("Local 모델 기반 RAG")


# 함수
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_messages():
    for chat_msg in st.session_state["messages"]:
        st.chat_message(chat_msg.role).write(chat_msg.content)


def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])


def create_chain(retriever, model_name="ollama"):
    """
    체인 생성
    """
    if "ollama" == model_name:
        
        # HuggingFace는 PC사양이 매우 많이 필요
        # ollama는 개인이 사용할 수 있는 정도도 지원한다.
        #     - But 성능이 제한적 (7~10B 모델을 사용)
        #     - Context 정보도 작아짐
        
        prompt = load_prompt("prompts/pdf-rag-ollama.yaml", encoding="utf-8")

        # Ollama 모델을 불러옵니다.
        llm = ChatOllama(model="hf.co/teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf:Q4_0")
    else:
        # Default
        prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

        llm = ChatOpenAI(model_name=model_name, temperature=0)
    
    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever | format_doc, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# 파일을 캐시 저장 (시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장한다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # Create Retriever
    return create_retriever(file_path)


# 사이드 바
with st.sidebar:
    # 초기화 버튼
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # LLM 모델 선택
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "ollama", "gpt-4o-mini"], index=0
    )


# 대화 기록을 저장해주는 용도
# 페이지가 새로고침이 일어나도 데이터가 지워지지 않는다.
if "messages" not in st.session_state:  # 처음 한 번만 실행되게
    st.session_state["messages"] = []
else:
    if clear_btn:
        st.session_state["messages"] = []
    print_messages()

if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 파일이 업로드 되었을 때
if uploaded_file:
    # 파입 업로드 후 retriever 생성 (작업시간 오래 걸릴 예정 ...)
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever=retriever, model_name=selected_model)
    st.session_state["chain"] = chain

# 사용자 질의
warning_msg = st.empty()  # 경고 메시지를 띄우기 위한 빈 영역 생성
user_input = st.chat_input("무엇이든 물어보세요!")
if user_input:

    # chain을 생성
    chain = st.session_state["chain"]
    if chain:
        # 웹에 대화를 출력
        with st.chat_message("user"):
            st.write(user_input)
        # 한 번에 출력
        # ai_answer = chain.invoke({"question": user_input})

        # chatGPT처럼 출력 (streaming)
        ai_response = chain.stream(user_input)
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
    else:
        warning_msg.error("파일을 업로드 해주세요.")
