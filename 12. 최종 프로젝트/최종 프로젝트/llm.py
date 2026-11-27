"""[1단계] 모델 연결

get_chat_model()   -> ChatOpenAI        MODEL_NAME, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL (한 번 만들어 재사용)
get_embeddings()   -> OpenAIEmbeddings  EMBEDDING_MODEL_NAME, 같은 키, base_url=EMBEDDING_BASE_URL — 채팅 주소로 대신하지 않음
text_of(content)   -> str               응답 content가 블록 목록([{"type": "text", "text": …}, …])이어도 텍스트만 이어 붙임
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import settings

_chat_model = None
_embeddings = None


def get_chat_model():
    global _chat_model
    if _chat_model is None:
        settings.need("OPENAI_API_KEY", "OPENAI_BASE_URL")
        _chat_model = ChatOpenAI(
            model=settings.MODEL_NAME,
            api_key=settings.value("OPENAI_API_KEY"),
            base_url=settings.value("OPENAI_BASE_URL"),
            reasoning_effort="none",  # 이 모델은 /v1/chat/completions에서 도구+reasoning 조합을 거부함
        )
    return _chat_model


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        settings.need("OPENAI_API_KEY", "EMBEDDING_BASE_URL")
        _embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            api_key=settings.value("OPENAI_API_KEY"),
            base_url=settings.value("EMBEDDING_BASE_URL"),
        )
    return _embeddings


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = block.get("text") if isinstance(block, dict) else block
            if text:
                parts.append(str(text))
        return "".join(parts)
    return str(content) if content is not None else ""
