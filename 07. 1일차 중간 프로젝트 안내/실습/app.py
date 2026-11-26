"""교대 인계 챗봇 — 종합 실습 — 웹 서버(제공, 고치지 않음)

실행(이 폴더에서):  python -m uvicorn app:app --host 0.0.0.0 --port 8037
화면: 서버를 켠 뒤 브라우저에서 8037번 포트 주소를 엽니다(내 PC에서 실행했다면 http://127.0.0.1:8037/).
수강생은 baseline/calc_tools.py(1단계)와 chatbot.py(2·3단계)만 고칩니다. 고친 뒤에는 터미널에서 Ctrl+C로 서버를 끄고 다시 켭니다.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import checks
import chatbot
from baseline import handover_chain

HERE = Path(__file__).resolve().parent
app = FastAPI(title="교대 인계 챗봇 — 종합 실습")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


def fail(e):
    """오류를 화면에 보여 줄 형태로 바꾼다(서버는 계속 동작)."""
    if isinstance(e, NotImplementedError):
        return {"ok": False, "error": f"[미구현] {e} — 안내문의 해당 TODO를 구현하고 서버를 다시 켜세요."}
    return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}


def msg_json(m):
    """LangChain 메시지를 화면용 dict로 바꾼다."""
    return {"type": m.type, "content": str(m.content), "name": getattr(m, "name", None),
            "tool_calls": [{"name": c["name"], "args": c["args"]} for c in (getattr(m, "tool_calls", None) or [])]}


QUESTION = '그 라인이 15분 멈추면 생산 손실은 몇 개야?'


class AskIn(BaseModel):
    room: str
    message: str


class MemoIn(BaseModel):
    memo: str


@app.post("/api/handover")
def handover(body: MemoIn):
    try:
        return {"ok": True, "reply": handover_chain.make_handover(body.memo)}
    except Exception as e:
        return fail(e)


@app.post("/api/toolcheck")
def toolcheck():
    """1단계 시험(모델·API 호출 없음): 정지 손실 도구를 직접 실행하고 TOOLS 등록을 확인한다."""
    try:
        from baseline import calc_tools
        return {"ok": True, "check": checks.run_check(checks.check_loss_tool, calc_tools)}
    except Exception as e:
        return fail(e)


@app.post("/api/reset")
def reset():
    try:
        chatbot.reset()
        return {"ok": True}
    except Exception as e:
        return fail(e)


@app.post("/api/ask")
def ask(body: AskIn):
    try:
        new = chatbot.ask_agent(body.room, body.message.strip())
        saved = chatbot.history(body.room)
    except Exception as e:
        return fail(e)
    return {"ok": True, "new": [msg_json(m) for m in new], "saved": [msg_json(m) for m in saved]}


@app.get("/api/history")
def history(room: str):
    try:
        return {"ok": True, "saved": [msg_json(m) for m in chatbot.history(room)]}
    except Exception as e:
        return fail(e)


@app.post("/api/sequence")
def sequence():
    """연속 시험: 첫 번째 대화방(검증-A) 생산량 등록 → 같은 방 정지 손실 → 두 번째 대화방(검증-B) 같은 질문.
    화면 선택 메뉴의 내부 값이 검증-A·검증-B이고, 화면에는 ‘첫 번째 대화방’·‘두 번째 대화방’으로 보인다."""
    try:
        chatbot.reset()                                    # 두 대화방을 포함한 모든 기록을 비운 뒤 시작
        checks.shift_agent, checks.config_for = chatbot.AGENT, chatbot.config_for
        turns, results = [], []
        plan = [("검증-A", "A라인은 시간당 240개를 생산해.", dict(expected_tools=[])),
                ("검증-A", QUESTION, dict(expected_tools=[("calculate_production_loss", {"units_per_hour": 240, "stopped_minutes": 15}, 60)], expected_qty=60)),
                ("검증-B", QUESTION, dict(expected_tools=[], ask_missing=True, question=QUESTION))]
        for room, text, kw in plan:
            new = chatbot.ask_agent(room, text)
            c = checks.run_check(checks.check_turn, new, **kw)
            results.append(c["result"])
            turns.append({"room": room, "message": text, "new": [msg_json(m) for m in new], "check": c})
        a, b = checks.count_saved("검증-A"), checks.count_saved("검증-B")
        ok4 = a == 6 and b == 2
        results.append(ok4)
    except Exception as e:
        return fail(e)
    if all(r is True for r in results):
        verdict = "→ 전체 통과"
    elif any(r is False for r in results):
        verdict = "→ ❌ 항목을 확인하고 TODO를 고쳐 서버를 다시 켠 뒤 다시 실행하세요"
    else:
        verdict = "→ 자동 판정 항목은 통과했습니다. 🔎 학생 확인 항목을 직접 읽고 판단해 결과 기록에 적으세요. (전체 통과 아님)"
    return {"ok": True, "turns": turns, "saved": {"검증-A": a, "검증-B": b}, "saved_ok": ok4, "verdict": verdict}
