import json
from data_loader import get_movie_name, get_user_history_names


AGENT_SYSTEM = """You are a movie recommendation agent with access to tools.
Use your tools to search the catalog, look up movie details, and analyze user taste before making recommendations.

## USER PROFILE

### Watch History (DO NOT recommend any of these — already watched):
{history}

### Movies They Explicitly Liked:
{likes}

### Movies They Explicitly Disliked (avoid similar movies):
{dislikes}

## INSTRUCTIONS
1. Use `search_catalog` to find movies matching the user's request or taste.
2. Use `get_movie_details` to learn more about promising candidates.
3. Use `get_user_taste` if you need to understand the user's preferences better.
4. Pick 1-3 movies that best fit the user's taste.
5. NEVER recommend a movie from the watch history.
6. NEVER recommend movies similar to their disliked movies.
7. Explain WHY each recommendation fits their specific taste.
8. Be conversational — ask a follow-up question about their preferences.

## FORMAT
- Use **bold** for movie titles.
- Use numbered lists for multiple recommendations.
- Keep paragraphs short and readable.

Respond directly to the user after gathering enough information."""


def get_user_likes_dislikes(profile, item_map, alias_map):
    likes = []
    dislikes = []
    for sess in profile['conversations']:
        likes.extend(get_movie_name(a, item_map, alias_map) for a in sess['user_likes'])
        dislikes.extend(get_movie_name(a, item_map, alias_map) for a in sess['user_dislikes'])
    return list(set(likes)), list(set(dislikes))


def build_prompt(profile, data):
    item_map, alias_map = data['item_map'], data['alias_map']
    history_names = get_user_history_names(profile, item_map, alias_map)
    likes, dislikes = get_user_likes_dislikes(profile, item_map, alias_map)

    return AGENT_SYSTEM.format(
        history=json.dumps(history_names[:20], indent=2),
        likes=json.dumps(likes, indent=2),
        dislikes=json.dumps(dislikes, indent=2),
    )
