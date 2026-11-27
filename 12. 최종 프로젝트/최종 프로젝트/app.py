"""웹 서버 — 화면(static/index.html, 제공)과 주고받는 경로(1단계에서 만들고 4단계에서 실행 기록을 잇는다)

실행: python -m uvicorn app:app --host 0.0.0.0 --port 8046
GET  /                 static/index.html 파일
GET  /api/info         {"ok": true, "title", "chapter": "12", "examples": [예시 질문…], "modes": [],
                       "settings": settings.status(),
                       "document": {"document": 문서명·쪽, "pdf_ready": bool, "pdf_url", "chunk_size", "chunk_overlap"},
                       "data": {"name", "url", "credit"}}
POST /api/chat/stream  요청 {"message": 질문, "mode": "main"} → 응답은 한 줄에 JSON 하나(media_type="application/x-ndjson")
                       실행 중(4단계): {"type": "step", "id", "label", "status": "running"|"done"|"error", "detail", "who", "ms"}
                       마지막:        {"type": "result", "ok": true, "question", "chat_model", "reply", "route", "sources", "searched"}
                                      실패하면 {"type": "result", "ok": false, "error": 이유}
출처 카드: 문서 {"n", "kind": "doc", "chunk_id", "source", "book_page", "pdf_page", "url", "quotes", "text", "rank"}
         측정 {"n", "kind": "measure", "dataset", "experiment_key", "columns", "points", "si_max_loc", "si_max",
               "oxide_max_loc", "oxide_max", "table", "url"}
답변 속 [n]은 같은 n의 출처 카드와 연결된다. 완성 답변을 화면 코드에 미리 넣지 않는다. 키·주소는 화면으로 보내지 않는다.

TODO 4 완료: 스트림 경로에서 요청마다 record.bind로 실행 이벤트를 받아 결과보다 먼저 한 줄씩 보낸다
"""
import json
import queue
import threading

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import chatbot
import record
import settings

PDF_URL = "https://www.pwc.com/kr/ko/insights/industry-focus/samilpwc_semicon-guidebook.pdf#page=8"
DATA_URL = "https://zenodo.org/records/17122442"

app = FastAPI(title="반도체 공정 RAG 챗봇")


class ChatRequest(BaseModel):
    message: str
    mode: str = "main"


@app.get("/")
def index():
    return FileResponse(settings.HERE / "static" / "index.html")


@app.get("/api/info")
def info():
    return {
        "ok": True,
        "title": "반도체 공정 RAG 챗봇",
        "chapter": "12",
        "examples": [
            "반도체 8대 공정을 순서대로 설명해 줘",
            "식각 공정이란 무엇인가요?",
            "포토 공정에서 포토레지스트(PR)는 왜 필요한가요?",
        ],
        "modes": [],
        "settings": settings.status(),
        "document": {
            "document": "삼일PwC 반도체 산업 가이드북 · 책자 6쪽(PDF 8번째 페이지)",
            "pdf_ready": settings.PDF_FILE.exists(),
            "pdf_url": PDF_URL,
            "chunk_size": 200,
            "chunk_overlap": 60,
        },
        "data": {"name": "Si_Oxide_etch_9_points.csv", "url": DATA_URL, "credit": "Zenodo 17122442 · CC BY 4.0"},
    }


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    events: queue.Queue = queue.Queue()

    def sink(event: dict) -> None:
        events.put(event)

    def work() -> None:
        try:
            record.bind(sink)
            result = chatbot.answer(req.message, req.mode)
            events.put({"type": "result", "ok": True, "question": req.message,
                        "chat_model": settings.MODEL_NAME, "reply": result["reply"],
                        "route": result["route"], "sources": result["sources"],
                        "searched": result["searched"],
                        "unsupported": result.get("unsupported", [])})
        except Exception as e:
            events.put({"type": "result", "ok": False, "error": str(e)})
        finally:
            record.unbind()

    def lines():
        threading.Thread(target=work, daemon=True).start()
        while True:
            event = events.get()
            yield json.dumps(event, ensure_ascii=False) + "\n"
            if event.get("type") == "result":
                break

    return StreamingResponse(lines(), media_type="application/x-ndjson")
