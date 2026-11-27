"""[1단계] 설정 읽기 — 프로젝트 폴더의 .env(= .env.example을 복사해 값 입력)

HERE: Path                      이 폴더
PDF_FILE: Path                  data/samilpwc_semicon-guidebook.pdf (첨부자료로 별도 배포, 없으면 2단계에서 발행처에서 받음)
CSV_FILE: Path                  data/Si_Oxide_etch_9_points.csv (첨부자료로 별도 배포)
SEARCH_K: int = 3               문서 검색 조각 수
MODEL_NAME · EMBEDDING_MODEL_NAME   .env의 값(없으면 gpt-6-luna · text-embedding-3-small)
value(name) -> str              환경 변수 → .env 순서로 찾은 값. 비었거나 '여기에'로 시작하면 ""
need(*names) -> None            값이 없으면 RuntimeError("…에 값을 넣고 서버를 다시 켜세요") — 모델 호출 전에 멈춤
status() -> dict                {"chat_ready", "embedding_ready", "chat_model", "embedding_model"} (키·주소 값은 넣지 않음)
"""
import os
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
PDF_FILE = HERE / "data" / "samilpwc_semicon-guidebook.pdf"
CSV_FILE = HERE / "data" / "Si_Oxide_etch_9_points.csv"
SEARCH_K = 3

# 환경 변수 → .env 순서: 이미 선언된 환경 변수를 .env 값이 덮지 않게 한다
load_dotenv(HERE / ".env", override=False)


def value(name: str) -> str:
    raw = (os.environ.get(name) or "").strip()
    if not raw or raw.startswith("여기에"):
        return ""
    return raw


MODEL_NAME = value("MODEL_NAME") or "gpt-6-luna"
EMBEDDING_MODEL_NAME = value("EMBEDDING_MODEL_NAME") or "text-embedding-3-small"


def need(*names: str) -> None:
    missing = [name for name in names if not value(name)]
    if missing:
        raise RuntimeError(f"{', '.join(missing)}에 값을 넣고 서버를 다시 켜세요")


def status() -> dict:
    return {
        "chat_ready": bool(value("OPENAI_API_KEY") and value("OPENAI_BASE_URL")),
        "embedding_ready": bool(value("OPENAI_API_KEY") and value("EMBEDDING_BASE_URL")),
        "chat_model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }
