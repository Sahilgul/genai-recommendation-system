import os
from langchain_qwq import ChatQwen
from openai import AsyncOpenAI, OpenAI
from config import QWEN_MODEL, EMBEDDING_BASE_URL, EMBEDDING_MODEL

llm = ChatQwen(
    model="qwen-plus",
    dashscope_api_key=os.environ["ALIBABA_API_KEY"],
    temperature=1,
)

embed_client = AsyncOpenAI(base_url=EMBEDDING_BASE_URL, api_key="lm-studio")
embed_client_sync = OpenAI(base_url=EMBEDDING_BASE_URL, api_key="lm-studio")
embed_model = EMBEDDING_MODEL


async def chat(messages):
    return await llm.ainvoke(messages)


async def chat_with_tools(messages, tools):
    bound = llm.bind(tools=tools)
    return await bound.ainvoke(messages)


async def embed(texts):
    resp = await embed_client.embeddings.create(model=embed_model, input=texts)
    return [d.embedding for d in resp.data]


def embed_sync(texts):
    resp = embed_client_sync.embeddings.create(model=embed_model, input=texts)
    return [d.embedding for d in resp.data]
