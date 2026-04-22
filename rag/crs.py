import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from config import BASE_DIR
from data_loader import resolve
from llm import embed_sync, stream_chat
from rag.prompts import build_prompt

CHROMA_DIR = str(BASE_DIR / "chroma_db")

collection = None


class LMStudioEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return embed_sync(input)


def load_index():
    global collection
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = LMStudioEmbedding()

    collection = client.get_or_create_collection(
        name="movies",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        raise RuntimeError("ChromaDB index is empty. Run build_index.py first.")

    print(f"ChromaDB index loaded: {collection.count()} movies")
    return collection


def retrieve(query, profile, data, n_results=15):
    history_asins = set(resolve(a, data['alias_map']) for a in profile['history'])

    results = collection.query(query_texts=[query], n_results=n_results + len(history_asins))

    retrieved = []
    for asin, name, dist in zip(results['ids'][0], results['documents'][0], results['distances'][0], strict=False):
        if asin not in history_asins:
            retrieved.append({"asin": asin, "title": name, "score": round(1 - dist, 4)})
        if len(retrieved) >= n_results:
            break

    return retrieved


async def stream_recommendation(profile, data, chat_history, user_message):
    retrieved = retrieve(user_message, profile, data)
    system = build_prompt(profile, data, retrieved)

    messages = [{"role": "system", "content": system}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    async for token in stream_chat(messages):
        yield token
