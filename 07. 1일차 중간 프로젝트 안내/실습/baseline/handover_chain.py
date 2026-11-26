"""인계문 체인 — 첫 모델 호출·프롬프트 체인 실습의 기준 코드(제공, 고치지 않음)

make_handover(memo) → 완료·예정·확인 세 줄 인계문(str). 프롬프트 → 모델 → 파서. 대화 기록은 저장하지 않는다.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import get_model

HANDOVER_RULES = """당신은 제조 현장의 교대 인계 도우미입니다.
메모를 정확히 세 줄로만 정리하세요.
1줄: '완료: '로 시작하고 끝난 작업을 씁니다.
2줄: '예정: '로 시작하고 앞으로 할 작업과 시각을 씁니다.
3줄: '확인: '로 시작하고 다음 근무자가 확인할 일을 씁니다.
메모에 없는 내용은 만들지 말고, 해당 내용이 없으면 '기재 없음'이라고 씁니다.
'예정'인 일을 '완료'로 바꾸지 마세요."""


def make_handover(memo: str) -> str:
    prompt = ChatPromptTemplate.from_messages([("system", HANDOVER_RULES), ("human", "{memo}")])
    return (prompt | get_model() | StrOutputParser()).invoke({"memo": memo})
