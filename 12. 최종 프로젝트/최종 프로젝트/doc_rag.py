"""[2단계] 문서 준비와 검색 — 삼일PwC 『반도체 산업 Business Guidebook』 책자 6쪽 = PDF 8번째 페이지(pages[7])

ensure_pdf() -> Path                       settings.PDF_FILE, 없으면 발행처 PDF(https://www.pwc.com/kr/ko/insights/industry-focus/samilpwc_semicon-guidebook.pdf)를 받아 저장
load_pdf_page() -> dict                    {"title", "text", "file", "pdf_page": 8, "book_page": 6, "url": 원문#page=8, "source"}
                                           본문 = 글자 크기 9.5~10.5pt·세로 위치 y<360 글자(그림 속 이름표 제외), 제목 = 19pt 이상
split_into_chunks(page, size=200, overlap=60) -> list[Document]
                                           langchain RecursiveCharacterTextSplitter(단어 경계에서 자름),
                                           ID "pwc-p8-c01"…, metadata: chunk_id·source·title·book_page·pdf_page·url·start·end
get_store() -> Chroma                      벡터 DB 캐싱 — store·조각·BM25 색인을 프로세스당 한 번만 만들어 재사용
build_index(chunks) -> Chroma              text-embedding-3-small, 코사인(collection_metadata={"hnsw:space": "cosine"}), chroma_db/에 저장,
                                           이미 있는 조각 ID는 다시 넣지 않음(디스크 캐시)
search(store, question, k=3) -> list[dict] 하이브리드: MMR(벡터·다양성) + BM25(키워드) 순위를 RRF로 융합 → 상위 k
                                           [{"rank", "chunk_id", "text", "distance", "source", "title", "book_page", "pdf_page", "url"}]
format_context(hits) -> str                "[조각ID: …] 출처 책자 6쪽(PDF 8쪽)\n원문" 블록을 빈 줄로 이음
"""
import shutil
import urllib.request

import pypdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import settings
from llm import get_embeddings

PUBLISHER_PDF = "https://www.pwc.com/kr/ko/insights/industry-focus/samilpwc_semicon-guidebook.pdf"
PDF_PAGE = 8
BOOK_PAGE = 6
SOURCE = "삼일PwC 반도체 산업 Business Guidebook"
RRF_K = 60  # RRF 완화 계수: 순위 → 점수(1/(RRF_K+순위)) 환산

_state = None  # (Chroma store, chunks, BM25Okapi)


def ensure_pdf():
    if settings.PDF_FILE.exists():
        return settings.PDF_FILE
    settings.PDF_FILE.parent.mkdir(parents=True, exist_ok=True)
    local = settings.HERE / settings.PDF_FILE.name  # 첨부자료: 프로젝트 루트에 놓인 원본
    if local.exists():
        shutil.copyfile(local, settings.PDF_FILE)
    else:
        urllib.request.urlretrieve(PUBLISHER_PDF, settings.PDF_FILE)
    return settings.PDF_FILE


def load_pdf_page() -> dict:
    path = ensure_pdf()
    page = pypdf.PdfReader(str(path)).pages[PDF_PAGE - 1]
    fragments = []

    def visit(text, cm, tm, font_dict, font_size):
        if not text.strip():
            return
        matrix = tm if tm is not None else cm
        # 이 PDF는 font_size가 항상 1.0이고 실제 크기가 텍스트 행렬 스케일에 들어 있다 (본문 10pt·제목 20pt)
        size = float(font_size) * float(matrix[3] or matrix[0] or 1.0)
        fragments.append({"text": text.replace("\n", ""), "size": size, "x": float(matrix[4]), "y": float(matrix[5])})

    page.extract_text(visitor_text=visit)

    title = " ".join(_assemble([f for f in fragments if f["size"] >= 19]).split())
    body = _assemble_columns([f for f in fragments if 9.5 <= f["size"] <= 10.5 and f["y"] < 360])

    return {
        "title": title,
        "text": body,
        "file": str(path),
        "pdf_page": PDF_PAGE,
        "book_page": BOOK_PAGE,
        "url": f"{PUBLISHER_PDF}#page={PDF_PAGE}",
        "source": SOURCE,
    }


def _assemble_columns(fragments: list) -> str:
    # 좌우 2단 본문: x가 가까운 조각끼리 한 단으로 묶어 왼쪽 단부터 읽는다 (줄별로 섞이지 않게)
    if not fragments:
        return ""
    columns = []
    for f in sorted(fragments, key=lambda f: f["x"]):
        best = min(columns, key=lambda col: abs(col[0] - f["x"]), default=None)
        if best is not None and abs(best[0] - f["x"]) < 100:
            best[1].append(f)
        else:
            columns.append([f["x"], [f]])
    columns.sort(key=lambda col: col[0])
    return "\n".join(_assemble(col[1]) for col in columns)


def _assemble(fragments: list) -> str:
    if not fragments:
        return ""
    # 같은 줄(세로 위치 차이 2pt 미만)끼리 모으고, 위→아래·왼→오른쪽 순서로 이어 붙인다
    lines = []
    for f in sorted(fragments, key=lambda f: -f["y"]):
        for line in lines:
            if abs(line[0] - f["y"]) < 2.0:
                line[1].append(f)
                break
        else:
            lines.append([f["y"], [f]])
    lines.sort(key=lambda line: -line[0])

    out = []
    for _, fs in lines:
        fs.sort(key=lambda f: f["x"])
        line, prev = "", None
        for f in fs:
            if prev is not None:
                gap = f["x"] - (prev["x"] + len(prev["text"]) * prev["size"] * 0.55)
                if gap > max(1.5, f["size"] * 0.35):
                    line += " "
            line += f["text"]
            prev = f
        # 줄끝 공백은 단어 경계의 줄바꿈이므로 남기고, 줄 사이는 이어 붙인다(생산하+는=생산하는)
        line = line.lstrip()
        if line:
            out.append(line)
    return "".join(out)


def split_into_chunks(page: dict, size: int = 200, overlap: int = 60) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    text = page["text"]
    chunks = []
    cursor = 0
    for i, piece in enumerate(splitter.split_text(text), 1):
        start = text.find(piece, cursor)
        if start == -1:
            start = text.find(piece)
        cursor = max(cursor, start + 1)
        chunks.append(Document(
            page_content=piece,
            metadata={
                "chunk_id": f"pwc-p{page['pdf_page']}-c{i:02d}",
                "source": page["source"],
                "title": page["title"],
                "book_page": page["book_page"],
                "pdf_page": page["pdf_page"],
                "url": page["url"],
                "start": start,
                "end": start + len(piece),
            },
        ))
    return chunks


def build_index(chunks: list):
    store = Chroma(
        collection_name="process_doc",
        embedding_function=get_embeddings(),
        persist_directory=str(settings.HERE / "chroma_db"),
        collection_metadata={"hnsw:space": "cosine"},
    )
    existing = set(store.get()["ids"])
    new = [c for c in chunks if c.metadata["chunk_id"] not in existing]
    if new:
        store.add_documents(new, ids=[c.metadata["chunk_id"] for c in new])
    return store


PARTICLES = [  # 한국어 조사 — 긴 것부터 한 개만 벗겨 어간을 만든다(공정이/공정을/공정에서 → 공정)
    "에서는", "에게는", "으로는", "이라는", "라는", "에서", "에게", "한테", "으로", "부터", "까지", "처럼", "보다",
    "이라", "라고", "이나", "이", "가", "을", "를", "은", "는", "의", "에", "와", "과", "도", "로", "만", "랑",
]


def _tokens(text: str) -> list:
    tokens = []
    for word in text.lower().split():
        stem = word
        for particle in PARTICLES:
            if len(stem) > len(particle) + 1 and stem.endswith(particle):
                stem = stem[:-len(particle)]
                break
        if stem:
            tokens.append(stem)
    return tokens


def _get_state() -> tuple:
    # 벡터 DB 캐싱: store·조각·BM25 색인을 프로세스당 한 번만 만든다
    global _state
    if _state is None:
        chunks = split_into_chunks(load_pdf_page())
        _state = (build_index(chunks), chunks, BM25Okapi([_tokens(c.page_content) for c in chunks]))
    return _state


def get_store():
    return _get_state()[0]


def search(store, question: str, k: int = 3) -> list:
    _, chunks, bm25 = _get_state()
    fetch_k = max(k * 3, 8)
    by_id = {c.metadata["chunk_id"]: c for c in chunks}

    # 벡터 다리: 코사인 거리표 + MMR 다양성 순서(임베딩 유사도로 재정렬)
    cosine = {doc.metadata["chunk_id"]: float(distance)
              for doc, distance in store.similarity_search_with_score(question, k=fetch_k)}
    mmr_order = [doc.metadata["chunk_id"]
                 for doc in store.max_marginal_relevance_search(question, k=fetch_k, fetch_k=fetch_k)]

    # 키워드 다리: BM25 상위(점수 0 제외)
    scores = bm25.get_scores(_tokens(question))
    bm25_order = [chunks[i].metadata["chunk_id"]
                  for i in sorted(range(len(chunks)), key=lambda i: -scores[i]) if scores[i] > 0]

    # RRF로 벡터+키워드 융합
    rrf = {}
    for leg in (mmr_order, bm25_order):
        for rank, chunk_id in enumerate(leg, 1):
            rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    hits = []
    for rank, chunk_id in enumerate(sorted(rrf, key=lambda cid: -rrf[cid])[:k], 1):
        doc = by_id[chunk_id]
        m = doc.metadata
        hits.append({
            "rank": rank,
            "chunk_id": m["chunk_id"],
            "text": doc.page_content,
            "distance": cosine.get(chunk_id),
            "source": m["source"],
            "title": m["title"],
            "book_page": m["book_page"],
            "pdf_page": m["pdf_page"],
            "url": m["url"],
        })
    return hits


def format_context(hits: list) -> str:
    blocks = [f"[{h['chunk_id']}: {h['title']}] 출처 책자 {h['book_page']}쪽(PDF {h['pdf_page']}쪽)\n{h['text']}"
              for h in hits]
    return "\n\n".join(blocks)
