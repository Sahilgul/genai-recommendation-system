from llm import stream_chat
from multi_agent.graph import build_graph

compiled_graph = None


def _get_graph():
    global compiled_graph
    if compiled_graph is None:
        compiled_graph = build_graph()
    return compiled_graph


async def stream_recommendation(profile, data, chat_history, user_message):
    graph = _get_graph()

    initial_state = {
        "user_message": user_message,
        "chat_history": chat_history,
        "profile": profile,
        "data": data,
        "history_names": [],
        "likes": [],
        "dislikes": [],
        "taste_analysis": "",
        "catalog_results": "",
        "candidates": "",
        "final_response": "",
        "composer_messages": [],
        "phase": "analyze",
    }

    yield "_Analyzing your taste profile..._\n\n"

    result = initial_state
    seen_phases = set()
    async for chunk in graph.astream(initial_state, stream_mode="values"):
        result = chunk
        phase = chunk.get("phase")
        if phase in seen_phases:
            continue
        seen_phases.add(phase)
        if phase == "search":
            yield "_Searching the catalog for candidates..._\n\n"
        elif phase == "compose":
            yield "_Composing your recommendation..._\n\n"

    composer_messages = result.get("composer_messages", [])
    if not composer_messages:
        yield "[ERROR] composer messages were not built"
        return

    async for token in stream_chat(composer_messages):
        yield token
