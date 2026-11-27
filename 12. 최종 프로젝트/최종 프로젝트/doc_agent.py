"""[2단계] 공정 문서 담당 에이전트 — 검색 도구로 찾은 원문만 근거로 답하고 [n] 인용·쪽수를 붙인다.

search_process_doc(query: str) -> (str, list)   @tool(response_format="content_and_artifact"): 모델에는 doc_rag.format_context 문자열,
                                                기록에는 검색 결과(hits) — record.current()["hits"]에도 추가
build_doc_agent()                               create_agent(채팅 모델, [search_process_doc], 지시문)
                                                지시문: 검색한 원문만 근거, JSON {"answerable", "sentences": [{"text", "chunk_id", "quote"}], "missing"}
finish_doc_answer(result) -> str                담당의 마지막 메시지 JSON에서 인용구가 이번 검색 원문에 글자 그대로 있는 문장만
                                                "문장 [n]"(record.add_source로 문서 카드 등록), 하나도 없으면 "제공된 문서에서 확인할 수 없습니다."
ask_doc_agent(question: str) -> str             @tool: 메인이 공정 질문을 담당에게 맡기는 위임 도구(4단계에서 메인에 등록)
"""
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.tools import tool

import settings
from doc_rag import build_index, format_context, load_pdf_page, search, split_into_chunks
from llm import get_chat_model
from record import DOC_LOG, add_source, current

DOC_PROMPT = """너는 반도체 공정 문서 담당 에이전트다.
- 답하기 전에 반드시 search_process_doc 도구로 질문과 관련된 가이드북 원문 조각을 검색한다.
- 답 문장은 도구 결과로 나온 검색 원문에만 근거를 둔다. 원문에 없는 내용은 문장에 쓰지 않는다.
- 최종 답변은 다른 설명 없이 아래 형식의 JSON 하나만 출력한다.
  {"answerable": true, "sentences": [{"text": "답 문장", "chunk_id": "pwc-p8-c01", "quote": "그 문장의 근거인 검색 원문 구절"}], "missing": []}
- 각 문장의 quote는 도구 결과 원문에서 글자 그대로 복사한 구절이어야 하고, chunk_id는 그 원문 조각의 [조각ID]다.
- 원문으로 뒷받침되지 않는 부분은 sentences에 넣지 않고 missing에 적는다.
- 문서로 답할 수 없는 질문이면 {"answerable": false, "sentences": [], "missing": ["이유"]}로 답한다."""


@tool(response_format="content_and_artifact")
def search_process_doc(query: str):
    """반도체 공정 가이드북(책자 6쪽 = PDF 8번째 페이지)에서 질의와 관련된 원문 조각을 검색해 반환한다. 공정 문서 질문에 답하기 전 반드시 먼저 호출한다."""
    page = load_pdf_page()
    store = build_index(split_into_chunks(page))
    hits = search(store, query, k=settings.SEARCH_K)
    current()["hits"].extend(hits)
    return format_context(hits), hits


def build_doc_agent():
    return create_agent(get_chat_model(), [search_process_doc], system_prompt=DOC_PROMPT, middleware=[DOC_LOG])


def _final_json(result) -> dict:
    messages = result.get("messages") if isinstance(result, dict) else None
    last = messages[-1] if messages else result
    try:
        # 랭체인 출력 파서: StrOutputParser(블록 콘텐츠 → 텍스트) → JsonOutputParser(코드펜스·JSON 복원)
        return JsonOutputParser().invoke(StrOutputParser().invoke(last))
    except Exception:
        return {}


def _contains(text: str, quote: str) -> bool:
    if quote in text:
        return True
    squash = lambda s: "".join(s.split())  # 공백 차이만 허용한 글자 그대로 확인
    return squash(quote) in squash(text)


def _hit_for(hits: list, chunk_id, quote: str):
    if chunk_id:
        for hit in hits:
            if hit["chunk_id"] == chunk_id and _contains(hit["text"], quote):
                return hit
    for hit in hits:
        if _contains(hit["text"], quote):
            return hit
    return None


def finish_doc_answer(result) -> str:
    hits = current()["hits"]
    lines = []
    for sentence in _final_json(result).get("sentences", []):
        text = str(sentence.get("text") or "").strip()
        quote = str(sentence.get("quote") or "").strip()
        if not text or not quote:
            continue
        hit = _hit_for(hits, sentence.get("chunk_id"), quote)
        if hit is None:
            # 타당성 검증 실패: 출처 없이 답에 쓰지 않고 검증 안 된 문장으로 남긴다
            current()["unsupported"].append(text)
            continue
        n = add_source(
            {
                "kind": "doc",
                "chunk_id": hit["chunk_id"],
                "source": hit["source"],
                "book_page": hit["book_page"],
                "pdf_page": hit["pdf_page"],
                "url": hit["url"],
                "quotes": [quote],
                "text": hit["text"],
                "rank": hit["rank"],
            },
            key=hit["chunk_id"],
        )
        lines.append(f"{text} [{n}]")
    return "\n".join(lines) if lines else "제공된 문서에서 확인할 수 없습니다."


_doc_agent = None


@tool
def ask_doc_agent(question: str) -> str:
    """반도체 공정 문서 확인이 필요한 질문을 공정 문서 담당 에이전트에게 맡긴다(공정·개념·재료·장비 설명). 답은 인용 [n]이 붙어 돌아온다."""
    global _doc_agent
    if _doc_agent is None:
        _doc_agent = build_doc_agent()
    result = _doc_agent.invoke({"messages": [HumanMessage(content=question)]})
    return finish_doc_answer(result)
