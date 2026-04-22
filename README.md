# Movie CRS

A conversational recommender for movies, built four different ways so the
trade-offs between them are obvious. Backed by FastAPI with token streaming,
ChromaDB for the RAG index, and LangSmith for tracing.

Dataset: [LLM-Redial](https://github.com/LiuTongyang/LLM-REDIAL) (Movies split).

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
Submit_Folder/
├── app.py                    # FastAPI + streaming endpoint
├── llm.py                    # Qwen / Groq client wrappers
├── config.py                 # paths and env
├── data_loader.py            # LLM-Redial loader + canonical title mapping
├── build_index.py            # builds the ChromaDB index for RAG
├── tools.py                  # tool schemas + execution
├── mcp_server.py             # MCP server exposing tools to the agent approach
├── evaluate.py               # batch evaluation script
├── test_batch.py             # quick smoke test against the running server
├── eda.ipynb                 # data exploration
├── few_shot/   agent/   rag/   multi_agent/      # the 4 approaches
└── tests/                    # 135 unit tests
```

## Setup

```bash
git clone <repo-url>
cd Submit_Folder

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env          # then fill in keys
```

You also need the dataset. Download LLM-Redial and put the `Movie/` folder at:

```
Submit_Folder/LLM_Redial/Movie/
├── Conversation.txt
├── final_data.jsonl
├── item_map.json
└── user_ids.json
```

Then build the RAG index once:

```bash
python build_index.py
```

## Run

```bash
uvicorn app:app --reload
```

Server boots, loads the catalog, loads ChromaDB, and is ready on
`http://localhost:8000`.

### Example request

```bash
curl -N -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "A1234567890",
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
pytest -q
```

135 tests covering the four approaches, the FastAPI app, MCP server, RAG index
builder, data loader, prompts, and tools.

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
- Default LLM is Qwen via DashScope. Tool-calling paths use Groq's
  `llama-3.3-70b-versatile` because Qwen tool-calling on DashScope was less
  reliable in testing.
