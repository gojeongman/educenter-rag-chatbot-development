"""[4단계] 메인 에이전트 — 복합 질문을 나누어 문서 담당과 측정 도구를 부르고 결과를 합친다.

MAIN_PROMPT: str   인사·일반 질문(두 도구 어디에도 해당하지 않음)은 직접 / 공정 개념 → ask_doc_agent / 실험 ID 수치 비교 → compare_etch_locations /
                   두 결과의 문장과 [n]을 바꾸지 않고 이어 붙임 / 도구 결과에 근거 없는 원인은
                   "주어진 자료로는 원인을 특정할 수 없습니다."(출처 없이) / '조회 결과 없음'은 그대로
build_main_agent() create_agent(채팅 모델, [ask_doc_agent, compare_etch_locations], MAIN_PROMPT, middleware=[record.MAIN_LOG])
finish_main_answer(result) -> str   메인의 최종 답(출력 파서)을 꺼내고 [n] 타당성을 검증한다:
                                    이번 요청에 등록된 출처 번호만 남기고 가짜 [n]은 떼어낸다.
"""
import re

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser

import record
from doc_agent import ask_doc_agent
from llm import get_chat_model
from measure import compare_etch_locations

MAIN_PROMPT = """너는 반도체 공정 어시스턴트의 메인 에이전트다. 두 도구 중 알맞은 것을 골라 부르고 결과를 합쳐 답한다.
- 인사·잡담이나 아래 두 도구 어디에도 해당하지 않는 일반 질문(일반 지식·일상·프로그래밍 등)은
  도구를 부르지 말고 네 지식으로 바로 직접 답한다. 억지로 도구를 골라 부르지 않는다.
- 공정·개념·재료·장비 등 가이드북 문서 확인이 필요하면 ask_doc_agent 도구로 공정 문서 담당에게 맡긴다.
- 실험 ID(예: 2024-07-02_01)가 나온 수치 비교 질문은 compare_etch_locations 도구로 측정값을 비교한다.
- 단, 이 프로젝트의 실험 데이터 안의 구체적 수치나 데이터 분석은 도구 결과 없이 지어내지 않는다.
  실험 ID 없는 데이터 질문에는 실험 ID를 지정해 비교해 달라고 안내한다.
- 복합 질문이면 필요한 도구를 알아서 여러 개 부른다.
- 도구가 돌려준 문장과 [n] 표기는 바꾸거나 지우지 않고 그대로 이어 붙여 답한다. 도구 결과에 없는 내용을 덧붙이지 않는다.
- "왜 그런지"처럼 원인을 묻는 부분은 도구 결과에 근거가 없으면 추측하지 말고
  "주어진 자료로는 원인을 특정할 수 없습니다."라고 [n] 없이 답한다.
- 도구가 '조회 결과 없음'을 돌려주면 그 사실을 그대로 알리고 값을 만들어내지 않는다."""


def build_main_agent():
    return create_agent(
        get_chat_model(),
        [ask_doc_agent, compare_etch_locations],
        system_prompt=MAIN_PROMPT,
        middleware=[record.MAIN_LOG],
    )


def finish_main_answer(result) -> str:
    last = result["messages"][-1] if isinstance(result, dict) else result
    reply = StrOutputParser().invoke(last)
    registered = {card["n"] for card in record.current()["sources"]}
    reply = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in registered else "", reply)
    reply = re.sub(r"[ \t]+\.", ".", reply)  # 떼어낸 [n] 자리의 공백 정리
    return reply.strip()
