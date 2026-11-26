"""대화 기록 — 대화 기록 실습의 기준 코드(제공, 고치지 않음)

build_chat_agent()  기록 저장소(InMemorySaver)를 연결한 대화 에이전트(도구 없음)
config_for(room)    대화방 이름 → 실행 설정 {"configurable": {"thread_id": room}}
호출할 때는 새 사용자 메시지 하나만 보낸다. 이전 메시지는 기록 저장소가 같은 thread_id의 기록에서 붙인다.
"""
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from llm import get_model

CHAT_RULES = """당신은 제조 현장의 교대 인계 도우미입니다.
이 대화에서 사용자가 알려준 내용만 근거로 답하세요.
같은 항목이 정정되면 가장 최근에 알려준 값을 따르세요.
대화에 없는 정보는 추측하지 말고 확인을 요청하세요."""


def build_chat_agent():
    return create_agent(model=get_model(), tools=[], system_prompt=CHAT_RULES, checkpointer=InMemorySaver())


def config_for(room: str) -> dict:
    return {"configurable": {"thread_id": room}}
