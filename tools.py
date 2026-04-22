import json
import re

from data_loader import (
    clean_name,
    get_movie_name,
    get_user_history_names,
    normalize_title,
    resolve,
    search_movies,
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": (
                "Substring search of movie TITLES only. This is NOT semantic search "
                "and does NOT understand genres, moods, actors, or themes. Pass words "
                "or fragments that literally appear in movie titles "
                "(e.g. 'star', 'matrix', 'alien', 'godfather'). "
                "Generic terms like 'sci-fi', 'thriller', 'dark', 'acclaimed' will "
                "return empty results. If unsure, call get_user_taste first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A literal title fragment (e.g. 'matrix', 'star wars', 'alien')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_details",
            "description": (
                "Get details about a specific movie: full title, how many users "
                "liked or disliked it, and sample conversation snippets about it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_name": {
                        "type": "string",
                        "description": "The movie title to look up",
                    },
                },
                "required": ["movie_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_taste",
            "description": (
                "Analyze the current user's taste profile: movies they liked, "
                "disliked, top genre keywords, and total watch count."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


def _get_likes_dislikes(profile, item_map, alias_map):
    likes, dislikes = [], []
    for sess in profile['conversations']:
        likes.extend(get_movie_name(a, item_map, alias_map) for a in sess['user_likes'])
        dislikes.extend(get_movie_name(a, item_map, alias_map) for a in sess['user_dislikes'])
    return list(set(likes)), list(set(dislikes))


def _extract_keywords(names):
    stop = {'with', 'from', 'that', 'this', 'were', 'been', 'have', 'they', 'them', 'their', 'what', 'about'}
    words = {}
    for name in names:
        for w in name.lower().split():
            w = re.sub(r'[^a-z0-9]', '', w)
            if len(w) > 3 and w not in stop:
                words[w] = words.get(w, 0) + 1
    return sorted(words, key=words.get, reverse=True)[:10]


def _find_movie_stats(movie_name, data):
    item_map, alias_map = data['item_map'], data['alias_map']
    norm_q = normalize_title(movie_name)

    target_asin = None
    for asin, name in item_map.items():
        if normalize_title(name) == norm_q:
            target_asin = resolve(asin, alias_map)
            break

    if not target_asin:
        return None

    liked_by, disliked_by = 0, 0
    snippets = []
    for profile in data['profiles']:
        for sess in profile['conversations']:
            resolved_likes = [resolve(a, alias_map) for a in sess['user_likes']]
            resolved_dislikes = [resolve(a, alias_map) for a in sess['user_dislikes']]
            if target_asin in resolved_likes:
                liked_by += 1
            if target_asin in resolved_dislikes:
                disliked_by += 1

            if target_asin in resolved_likes or target_asin in resolved_dislikes:
                cid = sess['conversation_id']
                if cid in data['conversations'] and len(snippets) < 2:
                    text = data['conversations'][cid]
                    name_lower = movie_name.lower()
                    for sentence in re.split(r'(?<=[.!?])\s+', text):
                        if name_lower in sentence.lower() and len(sentence) > 20:
                            cleaned = re.sub(r'^(User|Assistant|Recommender|Seeker):\s*', '', sentence).strip()
                            if cleaned not in snippets:
                                snippets.append(cleaned)
                            if len(snippets) >= 2:
                                break

    return {
        "title": clean_name(item_map.get(target_asin, movie_name)),
        "asin": target_asin,
        "liked_by_users": liked_by,
        "disliked_by_users": disliked_by,
        "snippets": snippets,
    }


def execute_tool(name, arguments, profile, data):
    item_map, alias_map = data['item_map'], data['alias_map']

    if name == "search_catalog":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 10)
        results = search_movies(query, item_map, alias_map, limit)
        return json.dumps([title for _, title in results])

    if name == "get_movie_details":
        movie_name = arguments.get("movie_name", "")
        details = _find_movie_stats(movie_name, data)
        if not details:
            return json.dumps({"error": f"Movie '{movie_name}' not found in catalog"})
        return json.dumps(details)

    if name == "get_user_taste":
        history = get_user_history_names(profile, item_map, alias_map)
        likes, dislikes = _get_likes_dislikes(profile, item_map, alias_map)
        keywords = _extract_keywords(likes + history[:10])
        return json.dumps({
            "watch_count": len(history),
            "liked_movies": likes,
            "disliked_movies": dislikes,
            "top_keywords": keywords,
        })

    return json.dumps({"error": f"Unknown tool: {name}"})
