"""요청 단위 기록 — 출처 카드 등록과 실행 이벤트를 이 요청에만 묶어 둔다.

start_request() -> dict        새 요청 기록을 시작한다: {"calls", "hits", "sources", "unsupported"}
current() -> dict              지금 요청의 기록(contextvar — 요청별로 독립)
add_source(card: dict, key: str) -> int   출처 카드를 등록하고 번호(n)를 돌려준다.
                               같은 key로 다시 등록하면 같은 번호(인용구는 합쳐짐) — 같은 근거는 같은 번호
report(reply: str, route=None) -> dict    화면 결과: {"reply", "route", "sources", "searched", "calls", "hits", "unsupported"}

[4단계] bind(sink) / unbind()               이 요청의 이벤트를 받을 함수 연결·해제
        emit(event: dict)                   {"type": "step", …}을 sink로 보냄(연결이 없으면 버림)
        ToolTrace(who)                      langchain AgentMiddleware: wrap_tool_call에서 시작 running → 끝 done/error·ms·요약 emit,
                                            current()["calls"]에 기록. 라벨: ask_doc_agent=담당 에이전트 호출,
                                            search_process_doc=검색 도구 호출, compare_etch_locations=측정 비교
        MAIN_LOG = ToolTrace("메인"), DOC_LOG = ToolTrace("공정 문서 담당")
"""
import contextvars
import time

from langchain.agents.middleware import AgentMiddleware

from llm import text_of

_request: contextvars.ContextVar = contextvars.ContextVar("record_request", default=None)
_sink: contextvars.ContextVar = contextvars.ContextVar("record_sink", default=None)


def start_request() -> dict:
    rec = {"calls": [], "hits": [], "sources": [], "unsupported": []}
    _request.set(rec)
    return rec


def current() -> dict:
    rec = _request.get()
    if rec is None:
        rec = start_request()
    return rec


def add_source(card: dict, key: str) -> int:
    rec = current()
    for s in rec["sources"]:
        if s.get("key") == key:
            for q in card.get("quotes", []):
                if q not in s["quotes"]:
                    s["quotes"].append(q)
            return s["n"]
    n = len(rec["sources"]) + 1
    entry = {"key": key, "n": n, **card}
    rec["sources"].append(entry)
    return n


def report(reply: str, route=None) -> dict:
    rec = current()
    return {
        "reply": reply,
        "route": route or "채팅 모델 직접 답변",
        "sources": [{k: v for k, v in s.items() if k != "key"} for s in rec["sources"]],
        "searched": bool(rec["hits"]),
        "calls": rec["calls"],
        "hits": rec["hits"],
        "unsupported": list(rec["unsupported"]),
    }


def bind(sink) -> None:
    """이 요청의 emit(event)을 sink로 보낸다(스트림 경로가 요청마다 연결)."""
    _sink.set(sink)


def unbind() -> None:
    _sink.set(None)


def emit(event: dict) -> None:
    sink = _sink.get()
    if sink is None:
        return
    try:
        sink(event)
    except Exception:
        pass  # 이벤트 전달 실패가 답변 처리를 막지 않게 한다


LABELS = {
    "ask_doc_agent": "담당 에이전트 호출",
    "search_process_doc": "검색 도구 호출",
    "compare_etch_locations": "측정 비교",
}


def _args_preview(name: str, args: dict) -> str:
    if name == "search_process_doc":
        return f"검색어 '{args.get('query', '')}'"
    if name == "compare_etch_locations":
        return f"실험 {args.get('experiment_key', '')}"
    if name == "ask_doc_agent":
        return str(args.get("question", ""))[:60]
    return str(args)[:80]


def _result_text(result) -> str:
    content = getattr(result, "content", result)
    return content if isinstance(content, str) else text_of(content)


def _result_summary(name: str, result) -> str:
    text = _result_text(result).strip()
    if name == "search_process_doc":
        blocks = text.count("\n\n") + 1 if text else 0
        return f"관련 조각 {blocks}개 검색" if text else "검색 결과 없음"
    if name == "compare_etch_locations":
        return text.splitlines()[0][:90] if text else "빈 결과"
    if name == "ask_doc_agent":
        lines = [l for l in text.splitlines() if l.strip()]
        return f"답변 {len(lines)}문장: {lines[0][:60]}…" if lines else "빈 답변"
    return text[:90]


class ToolTrace(AgentMiddleware):
    """도구 실행을 이벤트로 알린다: 시작 running → 끝 done/error(ms·요약) + current()["calls"] 기록"""

    def __init__(self, who: str):
        super().__init__()
        self.who = who

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        name = call.get("name", "?")
        label = LABELS.get(name, name)
        step_id = call.get("id") or f"{name}:{time.perf_counter_ns()}"
        args = call.get("args") or {}
        emit({"type": "step", "id": step_id, "label": label, "status": "running",
              "detail": _args_preview(name, args), "who": self.who})
        started = time.perf_counter()
        try:
            result = handler(request)
        except Exception as e:
            ms = round((time.perf_counter() - started) * 1000)
            detail = f"{type(e).__name__}: {e}"
            emit({"type": "step", "id": step_id, "label": label, "status": "error",
                  "detail": detail, "who": self.who, "ms": ms})
            current()["calls"].append({"who": self.who, "tool": name, "label": label,
                                       "status": "error", "ms": ms, "detail": detail, "args": args})
            raise
        ms = round((time.perf_counter() - started) * 1000)
        detail = _result_summary(name, result)
        emit({"type": "step", "id": step_id, "label": label, "status": "done",
              "detail": detail, "who": self.who, "ms": ms})
        current()["calls"].append({"who": self.who, "tool": name, "label": label,
                                   "status": "done", "ms": ms, "detail": detail, "args": args})
        return result


MAIN_LOG = ToolTrace("메인")
DOC_LOG = ToolTrace("공정 문서 담당")
