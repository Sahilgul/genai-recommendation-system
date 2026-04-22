from data_loader import get_movie_name, get_user_history_names


FEW_SHOT_SYSTEM = """You are a movie recommendation assistant. You have a warm, conversational style.
You recommend movies based on the user's watch history and stated preferences.

USER PROFILE:
- Watch history: [{history}]
- Movies they liked: [{likes}]
- Movies they disliked: [{dislikes}]

INSTRUCTIONS:
1. Recommend movies the user has NOT already watched.
2. Explain WHY each recommendation fits their taste (connect to movies they liked).
3. Keep recommendations to 1-3 movies per response.
4. If the user asks about a specific movie, share details and relate it to their preferences.
5. Be conversational — ask follow-up questions about their preferences.

Here are examples of good recommendation conversations:
{examples}
FORMAT:
- Use **bold** for movie titles.
- Use numbered lists for multiple recommendations.
- Keep paragraphs short and readable.

Now have a natural conversation with this user. Recommend movies they would genuinely enjoy."""


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
        history=", ".join(f'"{m}"' for m in history_names[:15]),
        likes=", ".join(f'"{m}"' for m in likes),
        dislikes=", ".join(f'"{m}"' for m in dislikes),
        examples=examples_block,
    )
