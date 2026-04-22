import os
import json
from mcp.server.fastmcp import FastMCP
from tools import _get_likes_dislikes, _extract_keywords, _find_movie_stats
from data_loader import search_movies, get_user_history_names, load_all

mcp = FastMCP("MovieCRS")

_profile = None
_data = None


def set_context(profile, data):
    global _profile, _data
    _profile = profile
    _data = data


def _ensure_context():
    global _profile, _data
    if _data is not None and _profile is not None:
        return
    _data = load_all()
    user_id = os.environ.get("MCP_USER_ID", "")
    idx = _data['user_index'].get(user_id)
    if idx is not None:
        _profile = _data['profiles'][idx]
    else:
        _profile = _data['profiles'][0]


@mcp.tool()
def search_catalog(query: str, limit: int = 10) -> str:
    """Search the movie catalog by keyword, genre, actor, or title. Returns up to `limit` matching movie titles."""
    _ensure_context()
    results = search_movies(query, _data['item_map'], _data['alias_map'], limit)
    return json.dumps([title for _, title in results])


@mcp.tool()
def get_movie_details(movie_name: str) -> str:
    """Get details about a specific movie: full title, how many users liked or disliked it, and sample conversation snippets about it."""
    _ensure_context()
    details = _find_movie_stats(movie_name, _data)
    if not details:
        return json.dumps({"error": f"Movie '{movie_name}' not found in catalog"})
    return json.dumps(details)


@mcp.tool()
def get_user_taste() -> str:
    """Analyze the current user's taste profile: movies they liked, disliked, top genre keywords, and total watch count."""
    _ensure_context()
    item_map, alias_map = _data['item_map'], _data['alias_map']
    history = get_user_history_names(_profile, item_map, alias_map)
    likes, dislikes = _get_likes_dislikes(_profile, item_map, alias_map)
    keywords = _extract_keywords(likes + history[:10])
    return json.dumps({
        "watch_count": len(history),
        "liked_movies": likes,
        "disliked_movies": dislikes,
        "top_keywords": keywords,
    })


if __name__ == "__main__":
    mcp.run(transport="stdio")
