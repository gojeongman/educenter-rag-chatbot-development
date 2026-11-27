"""서버 설정 읽기(제공, 고치지 않음) — 이 폴더의 .env 파일에서 읽습니다.

.env 에 넣는 값(따옴표 없이 = 뒤에 붙여 씁니다)
  OPENAI_API_KEY=수업용 API 키(채팅·임베딩 공용, 하나)
  OPENAI_BASE_URL=채팅 모델 주소(https://...)
  EMBEDDING_BASE_URL=임베딩 모델 주소(https://..., 채팅 주소와 다름)
모델 이름은 확정값이라 .env에 넣지 않습니다: 채팅 gpt-6-luna, 임베딩 text-embedding-3-small.
이 폴더 .env의 값이 비었거나 자리표시자이면 홈 폴더 .env(~/.env)에 같은 이름의 값이 있는지 봅니다.
임베딩 주소가 없을 때 채팅 주소로 대신하지 않습니다. 키는 화면으로 보내지 않습니다.
"""
import os
from pathlib import Path

from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
ENV_FILES = [p for p in (HERE / ".env", Path.home() / ".env") if p.is_file()]
_FILE_VALUES = [dotenv_values(p) for p in ENV_FILES]

MODEL_NAME = "gpt-6-luna"                         # 채팅·답변 생성(기본 모드와 RAG 모드가 같은 모델·같은 설정)
EMBEDDING_MODEL_NAME = "text-embedding-3-small"   # 문서·질문 임베딩


def _real(v):
    v = str(v or "").strip()
    return "" if (not v or v.startswith("여기에")) else v


def value(name):
    """설정값: 환경 변수 → 이 폴더 .env → 홈 .env 순서로 첫 실제 값. 없으면 빈 문자열."""
    for v in [os.environ.get(name)] + [vals.get(name) for vals in _FILE_VALUES]:
        if _real(v):
            return _real(v)
    return ""


def need(*names):
    """필요한 값이 없으면 모델을 부르기 전에 멈추고 무엇을 넣을지 알려 준다."""
    missing = [n for n in names if not value(n)]
    if missing:
        raise RuntimeError(f"프로젝트 폴더 .env의 {', '.join(missing)}에 안내받은 값을 넣고 서버를 다시 켜세요. "
                           "(모델을 호출하기 전에 멈췄습니다)")


def status():
    """화면 표시용 — 값 자체는 보내지 않고 준비 여부만."""
    return {"chat_ready": bool(value("OPENAI_API_KEY") and value("OPENAI_BASE_URL")),
            "embedding_ready": bool(value("OPENAI_API_KEY") and value("EMBEDDING_BASE_URL")),
            "chat_model": MODEL_NAME, "embedding_model": EMBEDDING_MODEL_NAME}

SEARCH_K = 3   # 11·12 검색 도구(search_process_doc)가 돌려줄 조각 수. 12 4단계에서 하나만 바꿔 전후를 비교합니다.
