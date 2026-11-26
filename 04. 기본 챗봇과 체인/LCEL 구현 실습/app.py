import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

SYSTEM_PROMPT = """당신은 교대 근무 인수인계 요약 전문가입니다.
사용자가 제공하는 교대 근무 인수인계 내용을 읽고, 인사말이나 부가 설명 없이 아래 3줄 형식으로만 요약하세요.

완료: 이번 교대에서 완료된 작업
진행중: 현재 진행 중인 작업
특이사항: 다음 교대가 알아야 할 특이사항

해당되는 내용이 없는 항목은 "없음"이라고 적고, 반드시 3줄로만 출력하세요."""

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-6-luna"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

chain = prompt | llm | StrOutputParser()

app = FastAPI()


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/chat")
def chat(request: ChatRequest):
    answer = chain.invoke({"question": request.question})
    return {"answer": answer}
