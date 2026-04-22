import json

from data_loader import get_movie_name, get_user_history_names

FEW_SHOT_SYSTEM = """You are a movie recommendation assistant with a warm, conversational style.
Use the example conversations below as a model for tone, structure, and depth of reasoning.

## USER PROFILE

### Watch History (DO NOT recommend any of these — already watched):
{history}

### Movies They Explicitly Liked:
{likes}

### Movies They Explicitly Disliked (avoid similar movies):
{dislikes}

## EXAMPLE CONVERSATIONS
{examples}

## INSTRUCTIONS
1. Pick 1-3 movies that genuinely fit the user's taste based on the profile above.
2. NEVER recommend a movie from the watch history.
3. NEVER recommend movies similar to their disliked movies.
4. Explain WHY each pick fits — tie it to a specific movie they liked or a pattern in their history.
5. Match the tone and structure of the example conversations above.
6. End with a short follow-up question about their preferences.

## FORMAT
- Use **bold** for movie titles.
- Use numbered lists for multiple recommendations.
- Keep paragraphs short and readable.

Respond directly to the user now."""


def get_user_likes_dislikes(profile, item_map, alias_map):
    likes = []
    dislikes = []
    for sess in profile['conversations']:
        likes.extend(get_movie_name(a, item_map, alias_map) for a in sess['user_likes'])
        dislikes.extend(get_movie_name(a, item_map, alias_map) for a in sess['user_dislikes'])
    return list(set(likes)), list(set(dislikes))


def build_prompt(profile, data, few_shot_examples):
    item_map, alias_map = data['item_map'], data['alias_map']
    history_names = get_user_history_names(profile, item_map, alias_map)
    likes, dislikes = get_user_likes_dislikes(profile, item_map, alias_map)

    examples_block = ""
    for i, ex in enumerate(few_shot_examples, 1):
        examples_block += f"\n--- Example Conversation {i} ---\n{ex}\n"

    return FEW_SHOT_SYSTEM.format(
        history=json.dumps(history_names[:15], indent=2),
        likes=json.dumps(likes, indent=2) if likes else "(none recorded)",
        dislikes=json.dumps(dislikes, indent=2) if dislikes else "(none recorded)",
        examples=examples_block,
    )
