"""모델 접속(제공) — 같은 폴더의 .env 파일(API_KEY·BASE_URL)로 ChatOpenAI를 한 번 만들어 재사용합니다.

.env 파일(같은 폴더, 없으면 만드세요)
    안내받은 API_KEY와 채팅 모델 주소 BASE_URL을 KEY=VALUE 형식으로 적습니다. 주석·빈 줄은 무시하고,
    값을 바꾼 뒤에는 서버를 다시 켜세요. 이 파일은 공유·제출 대상에서 빼고, 키를 HelpyCode나 채팅창에 붙여넣지 마세요.
get_model()  수업용 채팅 모델(ChatOpenAI, gpt-6-luna). API_KEY나 BASE_URL이 비었거나 자리표시자면 모델을 호출하기 전에 이유를 알려 주는 오류를 냅니다.
"""
from pathlib import Path

MODEL_NAME = "gpt-6-luna"   # 수업에서 쓰는 채팅·도구 호출 모델(확정값, 바꾸지 않음)

_PLACEHOLDER = "여기에_수업용_API_키_입력"


def _read_env(path: Path) -> dict:
    """같은 폴더의 .env 파일(KEY=VALUE 형식)을 읽어 dict로 돌려준다. 주석·빈 줄은 무시."""
    values = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_ENV = _read_env(Path(__file__).resolve().parent / ".env")
API_KEY = _ENV.get("API_KEY") or _PLACEHOLDER
BASE_URL = _ENV.get("BASE_URL") or _PLACEHOLDER

_cache = {}


def get_model():
    if "model" not in _cache:
        missing = [name for name, value in (("API_KEY", API_KEY), ("BASE_URL", BASE_URL))
                   if str(value).startswith("여기에")]
        if missing:
            raise RuntimeError(".env의 " + ", ".join(missing) + "에 안내받은 값을 넣고 서버를 다시 켜세요. (모델을 호출하기 전에 멈췄습니다)")
        from langchain_openai import ChatOpenAI
        # 도구 사용 시 응답 API로 자동 전환되면 content가 추론 블록(16진수 id) 목록으로 내려와
        # 화면·채점이 오염된다. 제공사 안내(gpt-6-luna + 도구 = /v1/responses 또는 reasoning_effort 'none')
        # 에 따라 채팅 완성 API를 고정하고 추론을 끈다.
        _cache["model"] = ChatOpenAI(model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL,
                                     use_responses_api=False, reasoning_effort="none")
    return _cache["model"]
