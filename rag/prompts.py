from data_loader import get_movie_name, get_user_history_names

RAG_SYSTEM = """You are a movie recommendation assistant powered by semantic retrieval.
You recommend movies based on the user's profile AND movies retrieved from our catalog that are semantically similar to the user's request.

## USER PROFILE

### Watch History (DO NOT recommend any of these — already watched):
{history}

### Movies They Explicitly Liked:
{likes}

### Movies They Explicitly Disliked (avoid similar movies):
{dislikes}

## CANDIDATE MOVIES (from our catalog):
{retrieved}

## INSTRUCTIONS
1. Pick 1-3 movies from the CANDIDATE MOVIES that best match the user's taste.
2. NEVER recommend a movie from the watch history.
3. NEVER recommend movies similar to their disliked movies.
4. Use your own judgment to decide which candidates are the best fit — don't just pick the first ones listed.
5. Explain WHY each recommendation fits — connect it to their liked movies or stated preferences.
6. Be conversational — ask a follow-up question about their preferences.

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


def build_prompt(profile, data, retrieved_movies):
    item_map, alias_map = data['item_map'], data['alias_map']
    history_names = get_user_history_names(profile, item_map, alias_map)
    likes, dislikes = get_user_likes_dislikes(profile, item_map, alias_map)

    retrieved_block = "\n".join(
        f"- {m['title']}" for m in retrieved_movies
    )

    return RAG_SYSTEM.format(
        history=", ".join(f'"{m}"' for m in history_names[:15]),
        likes=", ".join(f'"{m}"' for m in likes) if likes else "(none recorded)",
        dislikes=", ".join(f'"{m}"' for m in dislikes) if dislikes else "(none recorded)",
        retrieved=retrieved_block,
    )
