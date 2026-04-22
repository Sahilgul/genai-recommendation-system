# Movie CRS

A conversational recommender for movies, built four different ways so the
trade-offs between them are obvious. Backed by FastAPI with token streaming,
ChromaDB for the RAG index, and LangSmith for tracing.

Dataset: LLM-Redial (Movies split) — available on Kaggle.

## Approaches

There is one folder per approach, each exposing the same `stream_recommendation`
async generator so `app.py` can swap them via the `approach` field in the
request body.

| Approach        | What it does                                                              |
|-----------------|---------------------------------------------------------------------------|
| `few_shot`      | Pure prompting. User profile + a few example dialogues, then ask the LLM. |
| `agent`         | Tool-calling agent. LLM picks tools (`search_catalog`, `get_movie_details`, `get_user_taste`) exposed via an MCP server. |
| `rag`           | Embeds the catalog into ChromaDB, retrieves top-k by similarity, generates from the retrieved set. |
| `multi_agent`   | LangGraph: supervisor -> preference analyzer -> catalog expert (tool-calling) -> composer. |

## Project layout

```
genai-recommendation-system/
├── app.py                    # FastAPI + streaming endpoint
├── llm.py                    # Qwen client + embedding wrappers
├── config.py                 # paths and env
├── data_loader.py            # LLM-Redial loader + canonical title mapping
├── build_index.py            # builds the ChromaDB index for RAG
├── tools.py                  # tool schemas + execution
├── mcp_server.py             # MCP server exposing tools to the agent approach
├── evaluate.py               # batch evaluation script
├── test_batch.py             # quick smoke test against the running server
├── eda.ipynb                 # data exploration
├── prompt_versions.py        # before/after of the prompt that moved the needle
├── few_shot/   agent/   rag/   multi_agent/      # the 4 approaches
└── tests/                    # unit tests
```

## Setup

```bash
git clone https://github.com/Sahilgul/genai-recommendation-system.git
cd genai-recommendation-system

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env          # then fill in keys
```

You also need the dataset. Download the LLM-Redial Movies split from Kaggle
and put the `Movie/` folder at:

```
genai-recommendation-system/LLM_Redial/Movie/
├── Conversation.txt
├── final_data.jsonl
├── item_map.json
└── user_ids.json
```

For the `rag` approach you also need an embedding server running. This project
uses [LM Studio](https://lmstudio.ai/) serving `text-embedding-qwen3-embedding-0.6b`
(Qwen3 Embedding 0.6B) via its OpenAI-compatible endpoint. Update
`EMBEDDING_BASE_URL` in `config.py` to point at your LM Studio instance, then
build the RAG index once:

```bash
python build_index.py
```

## Run

```bash
uvicorn app:app --reload
```

Server boots, loads the catalog, loads ChromaDB, and is ready on
`http://localhost:8000`.

### Example requests

Pick any user ID returned by `GET /users?limit=20`. The examples below use
`A30Q8X8B1S3GGT`. The `-N` flag on curl is important — it disables buffering
so you can actually see tokens streaming in.

`few_shot` — pure prompting, fastest:

```bash
curl -N -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "A30Q8X8B1S3GGT",
    "message": "Looking for something dark and atmospheric, similar to Blade Runner.",
    "approach": "few_shot"
  }'
```

`agent` — tool-calling agent over MCP:

```bash
curl -N -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "A30Q8X8B1S3GGT",
    "message": "Recommend a sci-fi movie",
    "approach": "agent"
  }'
```

`rag` — ChromaDB retrieval + generation (needs the index built and the
embedding server reachable):

```bash
curl -N -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "A30Q8X8B1S3GGT",
    "message": "Recommend a sci-fi movie",
    "approach": "rag"
  }'
```

`multi_agent` — LangGraph pipeline (analyzer -> catalog expert -> composer):

```bash
curl -N -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "A30Q8X8B1S3GGT",
    "message": "Looking for something dark and atmospheric, similar to Blade Runner.",
    "approach": "multi_agent"
  }'
```

`approach` accepts `few_shot`, `agent`, `rag`, `multi_agent`. Default is
`few_shot`.

### Other endpoints

```
GET  /health                   # liveness + catalog size
GET  /users?limit=20           # list user IDs
GET  /users/{user_id}          # profile + history + might_like
GET  /movies/search?q=batman   # title search
```

## Tests

```bash
pytest
```

174 tests covering the four approaches, the FastAPI app, MCP server, RAG index
builder, data loader, prompts, and tools. Pytest is configured in
`pyproject.toml` to only collect from `tests/`, so the benchmark script
`test_batch.py` (which hits the running server) is correctly skipped.

## Linting

```bash
ruff check .          # lint
ruff check . --fix    # auto-fix what's safely fixable
ruff format .         # apply formatting
```

Configured in `pyproject.toml` — pyflakes, pycodestyle, isort, bugbear, and
pyupgrade rules enabled. Notebooks and tests have looser rules
(`per-file-ignores`).

## Prompt iterations

The brief asks for two prompt changes that improved recommendation accuracy.
See `prompt_versions.py` for the full before/after of the `few_shot` system
prompt (V1 = first commit, V2 = current) plus a short note on the two changes
that mattered: hard negative constraints on the watch history, and treating
dislikes as a similarity-avoid signal rather than just a list.

## Tracing

If `LANGCHAIN_API_KEY` is set, every LLM call is traced to LangSmith under the
project name in `LANGCHAIN_PROJECT`. Useful for inspecting token usage and
debugging the multi-agent graph.

## Notes

- The data loader does canonical-title mapping (`build_alias_map` in
  `data_loader.py`). Same movie released as DVD, Blu-ray, Steelbook etc. has
  multiple ASINs in the raw catalog. Without the mapping, RAG and search return
  duplicates.
- The MCP server (`mcp_server.py`) is launched as a subprocess by the `agent`
  approach. It reads the user ID from `MCP_USER_ID` so tool calls can be
  user-scoped without passing the profile through every tool call.
- LLM: Qwen (`qwen-plus`) via DashScope, configured by
  `DASHSCOPE_API_KEY` and `QWEN_MODEL` in `.env`. The same client is used for
  plain chat, streaming, and tool-calling.
- Embeddings: `text-embedding-qwen3-embedding-0.6b` (Qwen3 Embedding 0.6B)
  served locally via [LM Studio](https://lmstudio.ai/)'s OpenAI-compatible
  endpoint. Base URL is set in `config.py` (`EMBEDDING_BASE_URL`). Used by
  `build_index.py` and the `rag` approach — both will fail with a connection
  error if LM Studio isn't running and serving this model.
