"""10 기본 챗봇과 문서 RAG 챗봇 — 웹 서버(제공, 고치지 않음)

실행(이 폴더에서):  python -m uvicorn app:app --host 0.0.0.0 --port 8043
화면: 서버를 켠 뒤 브라우저에서 8043번 포트 주소를 엽니다(내 PC라면 http://127.0.0.1:8043/).
chatbot.py를 고친 뒤에는 터미널에서 Ctrl+C로 서버를 끄고 다시 켭니다.

화면이 보내는 요청   POST api/chat/stream {"message": "<질문>", "mode": "basic" | "rag"}
받는 응답(한 줄에 JSON 하나)  {"type": "step", ...} 처리 과정(보조 함수가 실제로 실행될 때만)
                              {"type": "result", "ok": true, "reply", "mode", "searched", "segments", "sources", ...}
                              또는 {"type": "result", "ok": false, "error"}
같은 내용을 한 번에 받는 POST api/chat 도 있습니다(처리 과정은 "steps" 목록). API 키는 화면으로 보내지 않습니다.
"""
import json
import os
import queue
import threading
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import chatbot
import events
import pdf_rag_tools
import settings

HERE = Path(__file__).resolve().parent
app = FastAPI(title="10 기본 챗봇과 문서 RAG 챗봇")
EXAMPLES = ["포토 공정과 식각 공정의 역할은 어떻게 달라?",
            "웨이퍼에서 원하는 무늬만 남기고 필요 없는 부분을 깎는 공정은 뭐야?",
            "우리 회사 A라인의 어제 식각 불량 원인은 뭐야?"]


class ChatIn(BaseModel):
    message: str
    mode: str = "basic"


def run(body):
    """질문 하나를 처리해 화면용 dict를 만든다(예외도 dict로)."""
    question, mode = body.message.strip(), body.mode
    if not question:
        return {"ok": False, "error": "빈 질문입니다."}
    if mode not in ("basic", "rag"):
        return {"ok": False, "error": "mode는 basic 또는 rag입니다."}
    try:
        result = chatbot.answer_basic(question) if mode == "basic" else chatbot.answer_rag(question)
    except NotImplementedError as e:
        return {"ok": False, "error": f"[미구현] {e} — 안내문의 해당 단계를 chatbot.py에 구현하고 서버를 다시 켜세요."}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    if not isinstance(result, dict) or not pdf_rag_tools.content_text(result.get("reply", "")).strip():
        return {"ok": False, "error": "함수는 {'reply': '답변 문장'} 형태의 dict를 돌려줘야 합니다. 빈 답변은 성공으로 보지 않습니다."}
    result = {"segments": [], "sources": [], "unsupported": [], "missing": [], "searched": False, **result,
              "reply": pdf_rag_tools.content_text(result["reply"])}
    if mode == "basic":
        result["searched"] = False
    return {"ok": True, "mode": mode, "question": question, "chat_model": settings.MODEL_NAME, **result}


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/info")
def info():
    return {"ok": True, "title": "10 기본 챗봇과 문서 RAG 챗봇", "chapter": "10", "examples": EXAMPLES,
            "modes": [{"id": "basic", "label": "기본 챗봇", "hint": "문서 검색 없이 채팅 모델만"},
                      {"id": "rag", "label": "문서 RAG", "hint": "가이드북 6쪽 검색 후 출처와 함께"}],
            "settings": settings.status(), "document": pdf_rag_tools.status()}


@app.post("/api/chat")
def chat(body: ChatIn):
    steps = []
    events.bind(steps.append)
    try:
        result = run(body)
    finally:
        events.unbind()
    return {**result, "steps": steps}


@app.post("/api/chat/stream")
def chat_stream(body: ChatIn):
    q = queue.Queue()

    def work():
        events.bind(q.put)
        try:
            result = run(body)
        finally:
            events.unbind()
        q.put({"type": "result", **result})
        q.put(None)

    threading.Thread(target=work, daemon=True).start()

    def lines():
        while True:
            item = q.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")
