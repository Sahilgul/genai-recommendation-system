import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage
from llm import chat, chat_with_tools
from data_loader import get_movie_name, get_user_history_names
from tools import TOOL_SCHEMAS, execute_tool
from multi_agent.prompts import (
    SUPERVISOR_SYSTEM,
    PREFERENCE_ANALYZER_SYSTEM,
    CATALOG_EXPERT_SYSTEM,
    RECOMMENDATION_COMPOSER_SYSTEM,
)

CATALOG_TOOL_SCHEMAS = [s for s in TOOL_SCHEMAS if s['function']['name'] in ('search_catalog', 'get_movie_details')]
MAX_TOOL_ROUNDS = 3


class AgentState(TypedDict):
    user_message: str
    chat_history: list
    profile: dict
    data: dict
    history_names: list
    likes: list
    dislikes: list
    taste_analysis: str
    catalog_results: str
    candidates: str
    final_response: str
    phase: str


def _get_likes_dislikes(profile, item_map, alias_map):
    likes, dislikes = [], []
    for sess in profile['conversations']:
        likes.extend(get_movie_name(a, item_map, alias_map) for a in sess['user_likes'])
        dislikes.extend(get_movie_name(a, item_map, alias_map) for a in sess['user_dislikes'])
    return list(set(likes)), list(set(dislikes))


async def supervisor(state):
    profile = state['profile']
    data = state['data']
    item_map, alias_map = data['item_map'], data['alias_map']

    history_names = state.get('history_names') or get_user_history_names(profile, item_map, alias_map)[:20]
    likes = state.get('likes') or []
    dislikes = state.get('dislikes') or []
    if not likes and not dislikes:
        likes, dislikes = _get_likes_dislikes(profile, item_map, alias_map)

    prompt = SUPERVISOR_SYSTEM.format(
        user_message=state['user_message'],
        taste_analysis=state.get('taste_analysis', 'Not yet analyzed.'),
        candidates=state.get('candidates', 'No candidates yet.'),
        history_count=len(history_names),
        likes_count=len(likes),
    )

    messages = [{"role": "system", "content": prompt}]
    response = await chat(messages)
    decision = response.content.strip().lower()

    if "compose" in decision and state.get('candidates'):
        next_phase = "compose"
    elif "search" in decision and state.get('taste_analysis'):
        next_phase = "search"
    elif state.get('candidates') and state.get('taste_analysis'):
        next_phase = "compose"
    elif state.get('taste_analysis'):
        next_phase = "search"
    else:
        next_phase = "analyze"

    return {
        "history_names": history_names,
        "likes": likes,
        "dislikes": dislikes,
        "phase": next_phase,
    }


async def analyze_preferences(state):
    profile = state['profile']
    data = state['data']
    item_map, alias_map = data['item_map'], data['alias_map']

    history_names = state.get('history_names') or get_user_history_names(profile, item_map, alias_map)[:20]
    likes, dislikes = state.get('likes') or [], state.get('dislikes') or []
    if not likes and not dislikes:
        likes, dislikes = _get_likes_dislikes(profile, item_map, alias_map)

    prompt = PREFERENCE_ANALYZER_SYSTEM.format(
        history=json.dumps(history_names, indent=2),
        likes=json.dumps(likes, indent=2),
        dislikes=json.dumps(dislikes, indent=2),
        user_message=state['user_message'],
    )

    messages = [{"role": "system", "content": prompt}]
    response = await chat(messages)

    return {
        "taste_analysis": response.content,
        "history_names": history_names,
        "likes": likes,
        "dislikes": dislikes,
        "phase": "search",
    }


async def search_catalog(state):
    prompt = CATALOG_EXPERT_SYSTEM.format(
        taste_analysis=state.get('taste_analysis', ''),
        user_message=state['user_message'],
    )

    messages = [{"role": "system", "content": prompt}]
    messages.append({"role": "user", "content": (
        f"Find movies matching this taste profile. The user asked: \"{state['user_message']}\""
    )})

    profile = state['profile']
    data = state['data']
    tool_log = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = await chat_with_tools(messages, CATALOG_TOOL_SCHEMAS)

        if not response.tool_calls:
            break

        messages.append(response)
        for tc in response.tool_calls:
            result = execute_tool(tc["name"], tc["args"], profile, data)
            tool_log.append(f"[{tc['name']}({json.dumps(tc['args'])})] → {result}")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    final_resp = await chat(messages)

    return {
        "candidates": final_resp.content,
        "catalog_results": "\n".join(tool_log) if tool_log else "No tool calls made.",
        "phase": "compose",
    }


async def compose_recommendation(state):
    prompt = RECOMMENDATION_COMPOSER_SYSTEM.format(
        history=json.dumps(state.get('history_names', []), indent=2),
        likes=json.dumps(state.get('likes', []), indent=2),
        dislikes=json.dumps(state.get('dislikes', []), indent=2),
        taste_analysis=state.get('taste_analysis', ''),
        candidates=state.get('candidates', ''),
        user_message=state['user_message'],
    )

    messages = [{"role": "system", "content": prompt}]
    messages.extend(state.get('chat_history', []))
    messages.append({"role": "user", "content": state['user_message']})

    response = await chat(messages)

    return {
        "final_response": response.content,
        "phase": "done",
    }


def route_after_supervisor(state):
    phase = state.get("phase", "analyze")
    if phase == "analyze":
        return "preference_analyzer"
    if phase == "search":
        return "catalog_expert"
    if phase == "compose":
        return "recommendation_composer"
    return "preference_analyzer"


def route_to_supervisor_or_end(state):
    phase = state.get("phase", "")
    if phase == "done":
        return END
    return "supervisor"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor)
    graph.add_node("preference_analyzer", analyze_preferences)
    graph.add_node("catalog_expert", search_catalog)
    graph.add_node("recommendation_composer", compose_recommendation)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges("supervisor", route_after_supervisor, {
        "preference_analyzer": "preference_analyzer",
        "catalog_expert": "catalog_expert",
        "recommendation_composer": "recommendation_composer",
    })

    graph.add_edge("preference_analyzer", "supervisor")
    graph.add_edge("catalog_expert", "supervisor")

    graph.add_conditional_edges("recommendation_composer", route_to_supervisor_or_end, {
        "supervisor": "supervisor",
        END: END,
    })

    return graph.compile()
