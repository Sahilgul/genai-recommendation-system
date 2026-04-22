from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from data_loader import load_all, get_user_history_names, get_movie_name, search_movies
from rag.crs import load_index
from few_shot.crs import stream_recommendation as few_shot_stream
from agent.crs import stream_recommendation as agent_stream
from rag.crs import stream_recommendation as rag_stream
from multi_agent.crs import stream_recommendation as multi_agent_stream

data = None


@asynccontextmanager
async def lifespan(app):
    global data
    data = load_all()
    print(f"Loaded: {len(data['item_map'])} movies, "
          f"{len(data['profiles'])} users, "
          f"{len(data['conversations'])} conversations")

    print("Loading ChromaDB vector index...")
    load_index()

    yield


app = FastAPI(
    title="Movie CRS API",
    description="Conversational Recommender System for movies",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatMessage(BaseModel):
    role: str
    content: str


class RecommendRequest(BaseModel):
    user_id: str
    message: str
    history: list[ChatMessage] = []
    approach: str = "few_shot"


@app.get("/health")
async def health():
    return {"status": "ok", "movies": len(data['item_map']) if data else 0}


@app.get("/users")
async def list_users(limit: int = 20, offset: int = 0):
    if not data:
        raise HTTPException(503, "Not loaded")
    users = [p['user_id'] for p in data['profiles'][offset:offset + limit]]
    return {"users": users, "total": len(data['profiles'])}


@app.get("/users/{user_id}")
async def get_user(user_id: str):
    if not data:
        raise HTTPException(503, "Not loaded")
    if user_id not in data['user_index']:
        raise HTTPException(404, f"User {user_id} not found")

    profile = data['profiles'][data['user_index'][user_id]]
    history = get_user_history_names(profile, data['item_map'], data['alias_map'])
    might_like = [get_movie_name(a, data['item_map'], data['alias_map']) for a in profile['might_like']]

    return {
        "user_id": user_id,
        "history_count": len(history),
        "history": history[:20],
        "might_like": might_like,
        "conversation_count": len(profile['conversations']),
    }


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    if not data:
        raise HTTPException(503, "Not loaded")
    if req.user_id not in data['user_index']:
        raise HTTPException(404, f"User {req.user_id} not found")

    profile = data['profiles'][data['user_index'][req.user_id]]
    chat_history = [{"role": m.role, "content": m.content} for m in req.history]

    streams = {
        "agent": agent_stream,
        "rag": rag_stream,
        "few_shot": few_shot_stream,
        "multi_agent": multi_agent_stream,
    }
    stream_recommendation = streams.get(req.approach, few_shot_stream)

    async def generate():
        try:
            async for token in stream_recommendation(
                profile=profile,
                data=data,
                chat_history=chat_history,
                user_message=req.message,
            ):
                yield token
        except Exception as e:
            error_msg = str(e)
            print(f"ERROR in /recommend: {error_msg}")
            yield f"\n\n[ERROR] {error_msg}"

    return StreamingResponse(generate(), media_type="text/markdown")


@app.get("/movies/search")
async def search(q: str, limit: int = 10):
    if not data:
        raise HTTPException(503, "Not loaded")
    results = search_movies(q, data['item_map'], data['alias_map'], limit)
    return {"query": q, "results": [{"asin": a, "title": n} for a, n in results]}


if __name__ == "__main__":
    import uvicorn  # noqa: F811
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
