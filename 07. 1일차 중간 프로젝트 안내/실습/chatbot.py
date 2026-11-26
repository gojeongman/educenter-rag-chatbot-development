"""교대 인계 챗봇 — 수강생이 HelpyCode로 채우는 파일

화면에서 대화방(첫 번째 대화방·두 번째 대화방)을 고르고 메시지를 보내면 서버가 ask_agent(room, question)를 부르고,
돌려준 메시지(방금 보낸 질문에서 새로 생긴 것)를 ‘호출 기록’에, 그 대화방의 전체 기록을 ‘저장된 기록’에 보여 줍니다.
room에는 화면에서 고른 대화방을 가리키는 값이 들어오고, 그대로 thread_id로 씁니다.
[연속 시험]은 첫 번째 대화방 생산량 등록 → 같은 방 정지 손실 → 두 번째 대화방 같은 질문을 차례로 보내고 채점합니다.

읽어 둘 제공 코드(고치지 않음): baseline/handover_chain.py · baseline/chat_memory.py · llm.py
단계: TODO 1 정지 손실 도구 만들기(baseline/calc_tools.py) → TODO 2 대화 기록 + 두 계산 도구 연결(build_shift_agent)
      → TODO 3 방금 보낸 질문의 메시지만 돌려주기(ask_agent)
"""
from langchain.agents import create_agent  # noqa: F401  (TODO 2에서 사용)
from langgraph.checkpoint.memory import InMemorySaver  # noqa: F401  (TODO 2에서 사용)

from baseline.calc_tools import TOOLS  # noqa: F401  (TODO 2에서 사용 — 1단계에서 완성한 도구 목록)
from baseline.chat_memory import config_for
from llm import get_model  # noqa: F401  (TODO 2에서 사용)

SHIFT_RULES = """당신은 제조 현장의 교대 인계 도우미입니다.
이 대화에서 사용자가 알려준 내용만 근거로 답하고, 값이 정정되면 가장 최근 값을 따르세요.
수량 계산은 직접 하지 말고 반드시 알맞은 도구를 사용하세요.
계산에 필요한 값(시간당 생산량, 시간)이 이 대화에 없으면 도구를 부르지 말고 먼저 물어보세요.
답에는 단위 '개'를 붙이세요."""


def build_shift_agent():
    """대화 기록과 계산 도구를 함께 쓰는 에이전트를 만든다. 서버가 [새로 시작]·연속 시험 때 부른다."""
    return create_agent(
        model=get_model(),
        tools=TOOLS,
        system_prompt=SHIFT_RULES,
        checkpointer=InMemorySaver(),
    )


def ask_agent(room: str, question: str) -> list:
    """같은 대화방에 새 메시지 하나를 보내고, 방금 보낸 질문에서 새로 생긴 메시지 목록만 돌려준다."""
    config = config_for(room)
    before = len(get_agent().get_state(config).values.get("messages", []))
    result = get_agent().invoke({"messages": [{"role": "user", "content": question}]}, config)
    return result["messages"][before:]


# ── 제공: 서버가 부르는 함수(고치지 않음) ──
AGENT = None


def reset():
    """에이전트를 새로 만든다(모든 대화방 기록이 0개에서 시작)."""
    global AGENT
    AGENT = build_shift_agent()


def get_agent():
    if AGENT is None:
        reset()
    return AGENT


def history(room: str) -> list:
    """그 대화방에 실제로 저장된 메시지 목록."""
    return get_agent().get_state(config_for(room)).values.get("messages", [])
