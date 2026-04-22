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
        "phase": "analyze",
    }

    result = await graph.ainvoke(initial_state)
    final_text = result.get("final_response", "")

    for char in final_text:
        yield char
