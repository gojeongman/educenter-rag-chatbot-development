"""질문 하나를 처리해 화면 결과 dict를 만든다(단계마다 이 함수의 처리를 바꾼다).

answer(question: str, mode: str = "main") -> dict
  돌려줄 dict: {"reply": 답변 문자열, "route": 처리 경로, "sources": [출처 카드], "searched": bool}
  1단계: 채팅 모델에 바로 질문
  2단계: 공정 문서 담당 에이전트(doc_agent) → 인용 [n]이 붙은 답과 문서 출처
  3단계: 질문에 실험 ID(예: 2024-07-02_01)가 있으면 측정 도구(measure) 결과,
         그 외 데이터 질문은 pandas 서브에이전트(measure.build_data_agent)로 분석,
         데이터로 확인할 수 없는 질문은 답변이 제한된다
  4단계: 메인 에이전트(main_agent)가 두 도구를 골라 부름
"""
from langchain_core.messages import HumanMessage

import main_agent
import record

ROUTE_MAIN = "메인 에이전트"

_main_agent = None


def answer(question: str, mode: str = "main") -> dict:
    record.start_request()
    global _main_agent
    if _main_agent is None:
        _main_agent = main_agent.build_main_agent()
    result = _main_agent.invoke({"messages": [HumanMessage(content=question)]})
    return record.report(main_agent.finish_main_answer(result), route=ROUTE_MAIN)
