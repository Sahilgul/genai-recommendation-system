import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from config import BASE_DIR
from llm import embed_sync
from data_loader import load_all, resolve

CHROMA_DIR = str(BASE_DIR / "chroma_db")
BATCH_SIZE = 32
MAX_PARALLEL = 16
MAX_SNIPPETS = 3
MAX_DOC_CHARS = 600


class LMStudioEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return embed_sync(input)


def _embed_batch(texts):
    return embed_sync(texts)


def _build_movie_to_convos(data):
    movie_convos = defaultdict(set)
    for profile in data['profiles']:
        for sess in profile['conversations']:
            cid = sess['conversation_id']
            for asin in sess['user_likes'] + sess['user_dislikes'] + sess['rec_item']:
                real = resolve(asin, data['alias_map'])
                movie_convos[real].add(cid)
    return movie_convos


def _extract_snippets(movie_name, conv_text):
    sentences = re.split(r'(?<=[.!?])\s+', conv_text)
    name_lower = movie_name.lower()
    snippets = []
    for s in sentences:
        s = s.strip()
        if name_lower in s.lower() and len(s) > 20:
            cleaned = re.sub(r'^(User|Assistant|Recommender|Seeker):\s*', '', s).strip()
            if len(cleaned) > 15 and cleaned not in snippets:
                snippets.append(cleaned)
            if len(snippets) >= MAX_SNIPPETS:
                break
    return snippets


def build_enriched_docs(data):
    movie_convos = _build_movie_to_convos(data)
    conversations = data['conversations']
    primary = data['primary_names']

    docs = {}
    enriched_count = 0

    for asin, title in primary.items():
        conv_ids = movie_convos.get(asin, set())
        snippets = []

        for cid in conv_ids:
            if cid not in conversations:
                continue
            found = _extract_snippets(title, conversations[cid])
            snippets.extend(found)
            if len(snippets) >= MAX_SNIPPETS:
                break

        if snippets:
            snippet_text = " ".join(snippets[:MAX_SNIPPETS])
            doc = f"{title} | {snippet_text}"
            enriched_count += 1
        else:
            doc = title

        docs[asin] = doc[:MAX_DOC_CHARS]

    print(f"  enriched {enriched_count}/{len(primary)} movies with conversation context")
    return docs


def build_index(data):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = LMStudioEmbedding()

    collection = client.get_or_create_collection(
        name="movies",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    primary = data['primary_names']
    existing = collection.count()

    if existing >= len(primary):
        print(f"ChromaDB index loaded from disk: {existing} movies")
        return

    if existing > 0:
        collection.delete(ids=collection.get()['ids'])

    print("  building enriched documents from conversations...")
    docs = build_enriched_docs(data)

    asins = list(docs.keys())
    texts = [docs[a] for a in asins]

    batches = []
    for i in range(0, len(asins), BATCH_SIZE):
        batches.append((asins[i:i + BATCH_SIZE], texts[i:i + BATCH_SIZE]))

    total = len(asins)
    total_batches = len(batches)
    t0 = time.time()
    done = 0
    batch_done = 0
    next_idx = 0

    print(f"  embedding {total} movies in {total_batches} batches "
          f"(batch_size={BATCH_SIZE}, max_parallel={MAX_PARALLEL})")

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        in_flight = {}

        for _ in range(min(MAX_PARALLEL, total_batches)):
            batch_ids, batch_docs = batches[next_idx]
            f = pool.submit(_embed_batch, batch_docs)
            in_flight[f] = (batch_ids, batch_docs)
            next_idx += 1

        while in_flight:
            finished = next(as_completed(in_flight))
            batch_ids, batch_docs = in_flight.pop(finished)
            embeddings = finished.result()
            collection.add(ids=batch_ids, documents=batch_docs, embeddings=embeddings)
            done += len(batch_ids)
            batch_done += 1

            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = total - done
            eta = remaining / rate if rate > 0 else 0
            ram = psutil.virtual_memory()
            print(f"  [{batch_done}/{total_batches}] {done}/{total} movies | "
                  f"{rate:.0f}/s | ETA: {eta:.0f}s | "
                  f"RAM: {ram.percent}% ({ram.available / 1024**3:.1f}GB free)",
                  flush=True)

            if next_idx < total_batches:
                batch_ids, batch_docs = batches[next_idx]
                f = pool.submit(_embed_batch, batch_docs)
                in_flight[f] = (batch_ids, batch_docs)
                next_idx += 1

    elapsed = time.time() - t0
    print(f"\n  done — {collection.count()} movies indexed in {elapsed:.1f}s")


if __name__ == '__main__':
    data = load_all()
    print(f"Loaded: {len(data['item_map'])} movies, {len(data['profiles'])} users")
    build_index(data)
