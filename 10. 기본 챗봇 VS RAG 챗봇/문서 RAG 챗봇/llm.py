"""모델 연결(제공, 고치지 않음)

get_chat_model()   채팅 모델 gpt-6-luna — OPENAI_API_KEY + 채팅 주소 OPENAI_BASE_URL
get_embeddings()   임베딩 모델 text-embedding-3-small — 같은 OPENAI_API_KEY + 임베딩 주소 EMBEDDING_BASE_URL
                   (임베딩 주소가 없으면 채팅 주소로 대신하지 않고 멈춥니다)
기본 모드와 RAG 모드는 같은 get_chat_model()을 씁니다(같은 모델·같은 생성 설정).
"""
import settings

_cache = {}


def get_chat_model():
    if "chat" not in _cache:
        settings.need("OPENAI_API_KEY", "OPENAI_BASE_URL")
        from langchain_openai import ChatOpenAI
        _cache["chat"] = ChatOpenAI(model=settings.MODEL_NAME, api_key=settings.value("OPENAI_API_KEY"),
                                    base_url=settings.value("OPENAI_BASE_URL"), timeout=60, max_retries=1)
    return _cache["chat"]


def get_embeddings():
    if "emb" not in _cache:
        settings.need("OPENAI_API_KEY", "EMBEDDING_BASE_URL")
        from langchain_openai import OpenAIEmbeddings
        _cache["emb"] = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME, api_key=settings.value("OPENAI_API_KEY"),
                                         base_url=settings.value("EMBEDDING_BASE_URL"),
                                         check_embedding_ctx_length=False)
    return _cache["emb"]


def embedding_name():
    """색인 이름에 넣을 임베딩 모델 이름(키·주소는 넣지 않음)."""
    return settings.EMBEDDING_MODEL_NAME
