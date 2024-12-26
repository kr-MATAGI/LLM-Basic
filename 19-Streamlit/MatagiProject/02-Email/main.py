import streamlit as st
import glob
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages.chat import ChatMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SerpAPIWrapper


# 초기 셋팅
load_dotenv()

# 타이틀
st.title("이메일 요약기")


# OutputParser에 들어가는 것
class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    company: str = Field(description="메일을 보낸 사람의 회사 정보")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")


# 함수
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_messages():
    for chat_msg in st.session_state["messages"]:
        st.chat_message(chat_msg.role).write(chat_msg.content)


def create_email_paring_chain():
    """
    체인 생성
    """
    output_parser = PydanticOutputParser(pydantic_object=EmailSummary)

    prompt = PromptTemplate.from_template(
        """
    You are a helpful assistant. Please answer the following questions in KOREAN.

    #QUESTION:
    다음의 이메일 내용 중에서 주요 내용을 추출해 주세요.

    #EMAIL CONVERSATION:
    {email_conversation}

    #FORMAT:
    {format}
    """
    )

    # format 에 PydanticOutputParser의 부분 포맷팅(partial) 추가
    prompt = prompt.partial(format=output_parser.get_format_instructions())

    # Chain
    chain = (
        prompt
        | ChatOpenAI(
            model_name="gpt-4o",
            temperature=0,
        )
        | output_parser
    )

    return chain


def create_report_chain():
    report_prompt = load_prompt("prompts/email.yaml", encoding="utf-8")

    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    chain = (
        report_prompt
        | ChatOpenAI(
            model="gpt-4o",
            temperature=0,
        )
        | output_parser
    )

    return chain


# 사이드 바
with st.sidebar:
    # 초기화 버튼
    clear_btn = st.button("대화 초기화")

    task_input = st.text_input("TASK 입력", "")


# 대화 기록을 저장해주는 용도
# 페이지가 새로고침이 일어나도 데이터가 지워지지 않는다.
if "messages" not in st.session_state:  # 처음 한 번만 실행되게
    st.session_state["messages"] = []
else:
    if clear_btn:
        st.session_state["messages"] = []
    print_messages()


# 사용자 질의
# gl: Country
# hl: UI Language
# num: 검색 결과 몇 건을 가지고 올 것인가?
params = {"engine": "google", "gl": "kr", "hl": "ko", "num": "3"}
search = SerpAPIWrapper(params=params)
user_input = st.chat_input("무엇이든 물어보세요!")
if user_input:
    # 웹에 대화를 출력
    with st.chat_message("user"):
        st.write(user_input)

    # 1) Email 파싱하는 chain을 생성
    email_chain = create_email_paring_chain()
    answer = email_chain.invoke({"email_conversation": user_input})

    # 2) 보낸 사람의 추가 정보 수집 (검색)
    search_query = f"{answer.person} {answer.company} {answer.email}"
    search_result = eval(search.run(search_query))
    search_result_string = "\n".join(search_result)

    # 3) 이메일 요약 chain 생성
    report_chain = create_report_chain()
    report_chain_input = {
        "sender": answer.person,
        "additional_information": search_result_string,
        "company": answer.company,
        "email": answer.email,
        "subject": answer.subject,
        "summary": answer.summary,
        "date": answer.date,
    }
    report_response = report_chain.stream(report_chain_input)

    ai_answer = ""
    with st.chat_message("assistant"):
        # 빈 공간(컨터에너)을 만들어서, 여기에 토큰 스트리밍을 출력한다.
        container = st.empty()

        for token in report_response:
            ai_answer += token
            container.markdown(ai_answer)

    # 한 번에 출력
    # ai_answer = chain.invoke({"question": user_input})

    # chatGPT처럼 출력 (streaming)
    # ai_response = chain.stream({"question": user_input})
    # ai_answer = ""
    # with st.chat_message("assistant"):
    #     # 빈 공간(컨터에너)을 만들어서, 여기에 토큰 스트리밍을 출력하낟.
    #     container = st.empty()

    #     for token in ai_response:
    #         ai_answer += token
    #         container.markdown(ai_answer)

    # 대회 기록을 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)
