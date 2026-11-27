"""PDF 읽기·청킹·임베딩·Chroma 검색 보조 함수(제공, 고치지 않음)

문서: 삼일PwC 『반도체 산업 Business Guidebook』(2025.12) 책자 6쪽 = PDF 8번째 페이지(코드의 페이지 번호 7)
      ‘2. 반도체 Value Chain Overview (8대 공정)’. 원본 PDF는 이 프로젝트에 들어 있지 않습니다.
      data/samilpwc_semicon-guidebook.pdf 가 없으면 발행처 주소에서 한 번 내려받아 data/에 둡니다.

TODO 2(chatbot.py)에서 차례로 부르는 함수
  load_pdf_page()                       → page(dict): 제목·본문·쪽 정보
  split_into_chunks(page, size, overlap) → 조각(Document) 목록. 조각 ID 예: pwc-p8-c03
  build_index(chunks)                   → Chroma 저장소(같은 설정이면 다시 만들지 않고 재사용)
  search(store, question, k=3)          → 검색 결과(hits) 목록: 조각 ID·원문·쪽·거리
  format_context(hits)                  → 답변 모델에 넣을 [검색된 원문] 문자열
  read_json(text)                       → 모델 응답 문자열에서 JSON 객체(dict)
  make_rag_reply(data, hits)            → 화면에 보낼 dict(답변 문장·[n] 표시·출처 카드)
각 함수는 실제로 실행될 때 화면의 ‘처리 과정’에 한 줄을 남깁니다(events.step).
"""
import hashlib
import json
import os
import threading
import urllib.request
from pathlib import Path

from langchain_core.documents import Document

import events
import llm

HERE = Path(__file__).resolve().parent
PDF_URL = "https://www.pwc.com/kr/ko/insights/industry-focus/samilpwc_semicon-guidebook.pdf"
PDF_FILE = HERE / "data" / "samilpwc_semicon-guidebook.pdf"
PUBLISHER, DOC_TITLE = "삼일PwC", "반도체 산업 Business Guidebook"
PAGE_INDEX = 7                     # 0부터 세는 번호 → PDF 뷰어의 8번째 페이지
PDF_PAGE, BOOK_PAGE = PAGE_INDEX + 1, 6
VIEW_URL = f"{PDF_URL}#page={PDF_PAGE}"   # #page=8은 뷰어 이동용(코드는 PAGE_INDEX로 페이지를 고름)
CHUNK_SIZE, CHUNK_OVERLAP = 200, 60
NOT_FOUND = "제공된 문서에서 확인할 수 없습니다."

_lock = threading.Lock()           # 서버를 켜자마자 질문이 와도 색인을 한 번만 만든다
_stores = {}


def ensure_pdf():
    """원본 PDF 경로. 없으면 발행처 주소에서 내려받는다."""
    path = Path(os.environ.get("PDF_PATH", "") or PDF_FILE)
    if not path.is_absolute():
        path = HERE / path
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(PDF_URL, timeout=60) as r:   # 기본 요청(브라우저 흉내 헤더를 붙이면 발행처가 403으로 거절)
            path.write_bytes(r.read())
    except Exception as e:
        raise RuntimeError(f"PDF가 없습니다. 발행처 원문({PDF_URL})을 내려받아 {path} 에 두세요. ({type(e).__name__})")
    return path


def page_runs(page):
    """페이지의 글자 묶음과 글자 크기·세로 위치."""
    runs = []
    page.extract_text(visitor_text=lambda t, cm, tm, fd, fs: runs.append((fs * (tm[0] or 1), tm[5], t)))
    return runs


def load_pdf_page():
    """PDF 8번째 페이지에서 제목(20pt)과 본문(10pt, 그림 영역 아래)만 읽는다. 그림 속 짧은 이름표는 뺀다."""
    with events.step("PDF 읽기", detail=f"PDF {PDF_PAGE}번째 페이지(책자 {BOOK_PAGE}쪽)") as s:
        from pypdf import PdfReader
        path = ensure_pdf()
        runs = page_runs(PdfReader(str(path)).pages[PAGE_INDEX])
        title = "".join(t for size, y, t in runs if size >= 19).replace("\n", "").strip()
        body = "".join(t for size, y, t in runs if 9.5 <= size <= 10.5 and y < 360).replace("\n", "").strip()
        if not body:
            raise RuntimeError("본문을 읽지 못했습니다. PDF 파일이 발행처 원본인지 확인하세요.")
        s["result"] = f"본문 {len(body)}자 · 제목 ‘{title}’"
    return {"title": title, "text": body, "file": path.name, "pdf_page": PDF_PAGE, "book_page": BOOK_PAGE,
            "url": VIEW_URL, "source": f"{PUBLISHER} 『{DOC_TITLE}』"}


def split_into_chunks(page, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """정해진 글자 수마다 자르되 앞 조각의 끝 overlap 글자를 다음 조각 앞에 겹친다."""
    with events.step("청킹", detail=f"{size}자 · 겹침 {overlap}자") as s:
        if not 0 <= overlap < size:
            raise ValueError("겹침은 0 이상, 조각 크기보다 작아야 합니다.")
        text, chunks, start = page["text"], [], 0
        while True:
            end = min(start + size, len(text))
            n = len(chunks) + 1
            chunks.append(Document(page_content=text[start:end], metadata={
                "chunk_id": f"pwc-p{page['pdf_page']}-c{n:02d}", "source": page["source"], "title": page["title"],
                "file": page["file"], "book_page": page["book_page"], "pdf_page": page["pdf_page"],
                "url": page["url"], "start": start, "end": end}))
            if end >= len(text):
                break
            start += size - overlap
        s["result"] = f"{len(chunks)}개 조각 ({chunks[0].metadata['chunk_id']} ~ {chunks[-1].metadata['chunk_id']})"
    return chunks


def collection_name(chunks):
    """임베딩 모델·조각 내용이 같으면 같은 이름 → 같은 설정으로 다시 실행해도 중복 저장하지 않는다."""
    key = llm.embedding_name() + "".join(c.page_content for c in chunks)
    m = chunks[0].metadata
    size = max(c.metadata["end"] - c.metadata["start"] for c in chunks)
    overlap = chunks[0].metadata["end"] - chunks[1].metadata["start"] if len(chunks) > 1 else 0
    return f"pwc-p{m['pdf_page']}-{size}-{overlap}-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:8]}"


def build_index(chunks):
    """조각을 임베딩해 로컬 Chroma(chroma_db/)에 저장. 이미 있는 조각 ID는 다시 넣지 않는다."""
    with events.step("임베딩·Chroma 색인", detail=llm.embedding_name()) as s:
        with _lock:
            name = collection_name(chunks)
            if name not in _stores:
                from langchain_chroma import Chroma
                _stores[name] = Chroma(collection_name=name, embedding_function=llm.get_embeddings(),
                                       persist_directory=str(HERE / "chroma_db"),
                                       collection_metadata={"hnsw:space": "cosine"})
            store = _stores[name]
            ids = [c.metadata["chunk_id"] for c in chunks]
            have = set(store.get(ids=ids)["ids"])
            new = [c for c in chunks if c.metadata["chunk_id"] not in have]
            if new:
                store.add_documents(new, ids=[c.metadata["chunk_id"] for c in new])
            s["result"] = f"새로 저장 {len(new)}개 · 저장된 조각 {len(store.get()['ids'])}개 · 컬렉션 {name}"
    return store


def search(store, question, k=3):
    """질문을 같은 임베딩 모델로 바꿔 코사인 거리가 가까운 조각 k개를 찾는다."""
    with events.step("문서 검색", detail=f"상위 {k}개") as s:
        found = store.similarity_search_with_score(question, k=k)
        hits = [{"rank": i, "chunk_id": d.metadata["chunk_id"], "text": d.page_content,
                 "distance": round(float(dist), 4), **{key: d.metadata[key] for key in
                 ("source", "title", "book_page", "pdf_page", "url", "start", "end")}}
                for i, (d, dist) in enumerate(found, start=1)]
        s["result"] = f"{len(hits)}개 반환: " + ", ".join(h["chunk_id"] for h in hits)
    return hits


def format_context(hits):
    """[조각ID: …] 출처 한 줄 + 원문 묶음을 빈 줄로 이어 붙인다."""
    return "\n\n".join(f"[조각ID: {h['chunk_id']}] {h['source']} 책자 {h['book_page']}쪽(PDF {h['pdf_page']}쪽)\n{h['text']}"
                       for h in hits)


def content_text(content):
    """모델 응답 content를 글자로. 응답이 블록 목록(추론·텍스트 블록)이면 텍스트 블록만 이어 붙인다."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def read_json(text):
    """모델 응답에서 첫 '{'부터 마지막 '}'까지를 JSON으로 읽는다. 읽지 못하면 빈 dict."""
    text = content_text(text).strip()
    a, b = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[a:b + 1]) if a >= 0 and b > a else {}
    except json.JSONDecodeError:
        return {}


def squash(text):
    return "".join(str(text).split())


def make_rag_reply(data, hits):
    """모델이 고른 근거(조각 ID·인용구)가 실제 검색 원문에 있는지 확인하고, 확인된 문장에만 [n] 출처를 붙인다."""
    with events.step("근거 확인", detail="인용구가 검색 원문에 그대로 있는지") as s:
        by_id = {h["chunk_id"]: h for h in hits}
        sources, segments, unsupported = [], [], []
        for item in data.get("sentences", []) if isinstance(data, dict) else []:
            text = str(item.get("text", "")).strip()
            cid, quote = str(item.get("chunk_id", "")).strip(), str(item.get("quote", "")).strip()
            if not text:
                continue
            hit = by_id.get(cid)
            if hit and quote and squash(quote) in squash(hit["text"]):
                card = next((c for c in sources if c["chunk_id"] == cid), None)
                if card is None:
                    card = {"n": len(sources) + 1, "kind": "doc", "chunk_id": cid, "quotes": [], **{key: hit[key] for key in
                            ("source", "title", "book_page", "pdf_page", "url", "text", "rank", "distance")}}
                    sources.append(card)
                if quote not in card["quotes"]:
                    card["quotes"].append(quote)
                segments.append({"text": text, "refs": [card["n"]]})
            else:
                unsupported.append(text)
        missing = [str(m) for m in (data.get("missing") or [])] if isinstance(data, dict) else []
        if not segments:
            reply = NOT_FOUND
        else:
            reply = " ".join(f"{seg['text']} [{seg['refs'][0]}]" for seg in segments)
        s["result"] = f"근거 확인 {len(segments)}문장 · 확인 실패 {len(unsupported)}문장 · 출처 {len(sources)}개"
    return {"reply": reply, "segments": segments, "sources": sources, "unsupported": unsupported,
            "missing": missing, "searched": True,
            "hits": [{key: h[key] for key in ("rank", "chunk_id", "distance", "book_page", "pdf_page")} for h in hits]}


def status():
    """화면 표시용 상태(색인을 만들지 않음)."""
    path = Path(os.environ.get("PDF_PATH", "") or PDF_FILE)
    return {"pdf_ready": (path if path.is_absolute() else HERE / path).is_file(), "pdf_url": VIEW_URL,
            "document": f"{PUBLISHER} 『{DOC_TITLE}』 책자 {BOOK_PAGE}쪽 · PDF {PDF_PAGE}번째 페이지",
            "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP}
