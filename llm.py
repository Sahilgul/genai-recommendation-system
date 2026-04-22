from langchain_qwq import ChatQwen
from openai import AsyncOpenAI, OpenAI

from config import EMBEDDING_BASE_URL, EMBEDDING_MODEL, QWEN_MODEL

_llms = {
    'router':   ChatQwen(model=QWEN_MODEL, temperature=0,   max_tokens=32),
    'analyzer': ChatQwen(model=QWEN_MODEL, temperature=0.3, max_tokens=512),
    'tool':     ChatQwen(model=QWEN_MODEL, temperature=0.2, max_tokens=2048),
    'writer':   ChatQwen(model=QWEN_MODEL, temperature=0.8, max_tokens=2048),
}

llm = _llms['writer']

embed_client = AsyncOpenAI(base_url=EMBEDDING_BASE_URL, api_key='lm-studio')
embed_client_sync = OpenAI(base_url=EMBEDDING_BASE_URL, api_key='lm-studio')
embed_model = EMBEDDING_MODEL


async def chat(messages, role='analyzer'):
    return await _llms[role].ainvoke(messages)


async def stream_chat(messages, role='writer'):
    async for chunk in _llms[role].astream(messages):
        if chunk.content:
            yield chunk.content


async def chat_with_tools(messages, tools, role='tool'):
    return await _llms[role].bind(tools=tools).ainvoke(messages)


async def embed(texts):
    resp = await embed_client.embeddings.create(model=embed_model, input=texts)
    return [d.embedding for d in resp.data]


def embed_sync(texts):
    resp = embed_client_sync.embeddings.create(model=embed_model, input=texts)
    return [d.embedding for d in resp.data]
