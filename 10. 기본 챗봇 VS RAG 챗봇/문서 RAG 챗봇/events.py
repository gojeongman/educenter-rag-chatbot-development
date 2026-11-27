"""처리 과정 기록(제공, 고치지 않음) — 함수·도구가 실제로 실행될 때만 화면의 ‘실행 기록’에 한 줄씩 보냅니다.

with step("문서 검색", detail="상위 3개") as s:   # 시작하면 ‘진행 중’
    ...                                         # 끝나면 소요 시간과 s["result"] 요약, 예외가 나면 ‘실패’
    s["result"] = "3개 반환"                     # s["error"] = "…" 로 실패를 직접 표시할 수도 있음
요청마다 따로 기록합니다(contextvars). 동시에 여러 질문이 와도 다른 요청의 줄이 섞이지 않습니다.
타이머로 단계를 꾸며 내지 않습니다. 기본 모드처럼 아무 함수도 부르지 않으면 줄이 생기지 않습니다.
"""
import contextvars
import time
from contextlib import contextmanager

_sink = contextvars.ContextVar("events_sink", default=None)
_counter = contextvars.ContextVar("events_counter", default=None)


def bind(sink):
    """현재 요청에 이벤트를 받을 함수를 연결(app.py가 요청마다 호출)."""
    _sink.set(sink)
    _counter.set([0])


def unbind():
    _sink.set(None)


def emit(event):
    sink = _sink.get()
    if sink:
        sink({"type": "step", **event})


@contextmanager
def step(label, detail="", who=""):
    counter = _counter.get()
    if counter is None:
        counter = [0]
        _counter.set(counter)
    counter[0] += 1
    sid = counter[0]
    t0 = time.perf_counter()
    box = {"result": "", "error": ""}
    emit({"id": sid, "label": label, "status": "running", "detail": detail, "who": who})
    try:
        yield box
    except Exception as e:
        emit({"id": sid, "label": label, "status": "error", "who": who, "detail": f"{type(e).__name__}: {str(e)[:160]}",
              "ms": round((time.perf_counter() - t0) * 1000)})
        raise
    emit({"id": sid, "label": label, "status": "error" if box["error"] else "done", "who": who,
          "detail": box["error"] or box["result"] or detail, "ms": round((time.perf_counter() - t0) * 1000)})
