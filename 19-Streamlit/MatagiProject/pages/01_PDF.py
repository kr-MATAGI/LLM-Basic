import streamlit as st
import glob
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

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


def create_chain(retriever):
    """
    체인 생성
    """
    # prompt = load_prompt(prompt_filepath, encoding="utf-8")

    # # GPT
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Parser
    # output_parser = StrOutputParser()

    # Chain
    # chain = prompt | llm | output_parser
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Answer in Korean.

    #Context: 
    {context}

    #Question:
    {question}

    #Answer:"""
    )

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
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

    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()

    return retriever


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

if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 파일이 업로드 되었을 때
if uploaded_file:
    # 파입 업로드 후 retriever 생성 (작업시간 오래 걸릴 예정 ...)
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever=retriever)
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
