"""계산 도구 — 계산 도구 실습의 기준 코드 + 1단계(TODO 1)에서 수강생이 정지 손실 도구를 추가하는 파일

expected_output()            제공 · 시간당 생산량 × 시간(분) ÷ 60
calculate_production         제공 · 완성 예제: 가동한 시간의 예상 생산량(개)
calculate_production_loss    TODO 1 · 멈춘(정지한) 시간의 생산 손실(개) — 수강생이 추가
TOOLS                        에이전트에 넘기는 도구 목록(TODO 1에서 정지 손실 도구를 등록)

직접 시험(모델·API 호출 없음, starter 폴더에서): python checks.py
"""
from langchain.tools import tool


def expected_output(units_per_hour: int, operating_minutes: int) -> int:
    """시간당 생산량 × 가동 시간(분) ÷ 60 = 예상 생산량(개, 반올림)"""
    return round(units_per_hour * operating_minutes / 60)


@tool
def calculate_production(units_per_hour: int, operating_minutes: int) -> int:
    """라인이 '가동한' 시간 동안 만든 예상 생산량(개)을 계산합니다.
    units_per_hour는 시간당 생산량(개/시간), operating_minutes는 가동 시간(분)입니다.
    정지·고장으로 못 만든 수량에는 사용하지 않습니다."""
    return expected_output(units_per_hour, operating_minutes)


@tool
def calculate_production_loss(units_per_hour: int, stopped_minutes: int) -> int:
    """라인이 '멈춘(정지한)' 시간 때문에 못 만든 생산 손실량(개)을 계산합니다.
    units_per_hour는 시간당 생산량(개/시간), stopped_minutes는 정지 시간(분)입니다.
    가동 중 생산량 계산에는 사용하지 않습니다."""
    return expected_output(units_per_hour, stopped_minutes)


TOOLS = [calculate_production, calculate_production_loss]
