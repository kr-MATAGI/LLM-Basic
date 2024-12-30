import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote.models import MultiModal
from langchain_teddynote import logging

from retriever import create_retriever

# 초기 셋팅
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] 이미지 인식")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


# 타이틀
st.title("이미지 인식 기반 ChatBot")

# 탭 생성
main_tab1, main_tab2 = st.tabs(["이미지", "대화내용"])


# 함수
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_messages():
    for chat_msg in st.session_state["messages"]:
        main_tab2.chat_message(chat_msg.role).write(chat_msg.content)



def generate_answer(
    img_filepath, 
    system_prompt, 
    user_prompt, 
    model_name="gpt-4o"
):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0.1,  # 창의성 (0.0 ~ 2.0)
        model_name=model_name,
    )

    # 역할 부여 (전역 설정, 페로소나 및 임무 설정)
    # system_prompt = """당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다.
    # 당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다."""

    # # 원하는 목표를 디테일 하게
    # user_prompt = """당신에게 주어진 표는 회사의 재무제표 입니다. 흥미로운 사실을 정리하여 답변하세요."""

    # 멀티모달 객체 생성
    multi_modal = MultiModal(
        llm, system_prompt=system_prompt, user_prompt=user_prompt
    )

    answer = multi_modal.stream(img_filepath)
    return answer


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

    # 이미지 업로드
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

    # LLM 모델 선택
    selected_model = st.selectbox(
        "LLM 선택", 
        ["gpt-4o", "gpt-4o-mini"], 
        index=0
    )

    # 시스템 프롬프트 추가
    system_prompt = st.text_area(
        "시스템 프롬프트", 
        "당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다. 당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다.",
        height=200,
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

# 사용자 질의
warning_msg = main_tab2.empty()  # 경고 메시지를 띄우기 위한 빈 영역 생성

# 이미지가 업로드 -> 미리 보기
if uploaded_file:
    imagefile_path = process_image(uploaded_file)
    main_tab1.image(imagefile_path)

user_input = st.chat_input("무엇이든 물어보세요!")
if user_input:
    if uploaded_file:
        # imagefile_path = process_image(uploaded_file)
        # 답변 요청
        response = generate_answer(
            img_filepath=imagefile_path,
            system_prompt=system_prompt,
            user_prompt=user_input,
            model_name=selected_model,
        )

    # chain을 생성
    # 웹에 대화를 출력
    with main_tab2.chat_message("user"):
        st.write(user_input)
    # 한 번에 출력
    # ai_answer = chain.invoke({"question": user_input})

    # chatGPT처럼 출력 (streaming)
    ai_answer = ""
    with main_tab2.chat_message("assistant"):
        # 빈 공간(컨터에너)을 만들어서, 여기에 토큰 스트리밍을 출력하낟.
        container = st.empty()

        for token in response:
            ai_answer += token.content
            container.markdown(ai_answer)

    # 대회 기록을 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)
else: 
    warning_msg.error("이미지를 업로드 해주세요.")
