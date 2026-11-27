"""10 기본 챗봇과 문서 RAG 챗봇 — 학생이 구현하는 파일

화면에서 [기본 챗봇]을 고르고 보내면 answer_basic(question), [문서 RAG]를 고르면 answer_rag(question)이 불립니다.
두 함수 모두 dict를 돌려줍니다. 최소 형태는 {"reply": "답변 문장"} 입니다.

사용할 수 있는 제공 코드
  llm.get_chat_model()            채팅 모델 gpt-6-luna(두 모드가 같은 모델·같은 설정)
  prompts.BASIC_SYSTEM            기본 모드 지시문
  prompts.RAG_SYSTEM              RAG 모드 지시문(JSON으로 답하게 함)
  pdf_rag_tools.load_pdf_page()   → split_into_chunks(page) → build_index(chunks) → search(store, question, k=3)
                                  → format_context(hits) → read_json(text) → make_rag_reply(data, hits)
모델 호출: model.invoke([("system", 지시문), ("human", 사람 메시지)]).content 가 답변 문자열입니다.
"""
import pdf_rag_tools as tools
from llm import get_chat_model
from prompts import BASIC_SYSTEM, RAG_SYSTEM


def answer_basic(question: str) -> dict:
    # TODO 1 · 기본 챗봇: 문서 검색 없이 question을 채팅 모델에 보내고 {"reply": 답변 문자열}을 돌려준다.
    model = get_chat_model()
    response = model.invoke([("system", BASIC_SYSTEM), ("human", question)])
    return {"reply": response.content}


def answer_rag(question: str) -> dict:
    # TODO 2 · 문서 RAG: PDF 쪽 읽기 → 청킹 → 색인 → 검색(k=3) → 검색 원문과 질문을 모델에 전달
    #          → 응답 JSON 읽기 → make_rag_reply(data, hits)의 결과를 그대로 돌려준다.
    page = tools.load_pdf_page()
    chunks = tools.split_into_chunks(page)
    store = tools.build_index(chunks)
    hits = tools.search(store, question, k=3)
    model = get_chat_model()
    human = "[질문]\n" + question + "\n\n[검색된 원문]\n" + tools.format_context(hits)
    response = model.invoke([("system", RAG_SYSTEM), ("human", human)])
    data = tools.read_json(response.content)
    return tools.make_rag_reply(data, hits)
