from agent.prompts import build_prompt as build_agent_prompt
from few_shot.prompts import build_prompt as build_few_shot_prompt
from few_shot.prompts import get_user_likes_dislikes
from rag.prompts import build_prompt as build_rag_prompt


def make_profile(likes=None, dislikes=None, history=None):
    return {
        'user_id': 'TEST_USER',
        'history': history or ['A1', 'A2'],
        'might_like': [],
        'conversations': [{
            'conversation_id': 1,
            'user_likes': likes or [],
            'user_dislikes': dislikes or [],
            'rec_item': [],
        }],
    }


def make_data(item_map=None, alias_map=None):
    imap = item_map or {'A1': 'The Matrix', 'A2': 'Inception', 'A3': 'Interstellar'}
    amap = alias_map or {}
    return {
        'item_map': imap,
        'alias_map': amap,
        'primary_names': {k: v for k, v in imap.items()},
        'profiles': [],
        'conversations': {},
        'user_index': {},
    }


# ── get_user_likes_dislikes ──

class TestGetUserLikesDislikes:
    def test_extracts_likes(self):
        profile = make_profile(likes=['A3'])
        data = make_data()
        likes, dislikes = get_user_likes_dislikes(profile, data['item_map'], data['alias_map'])
        assert 'Interstellar' in likes
        assert dislikes == []

    def test_extracts_dislikes(self):
        profile = make_profile(dislikes=['A2'])
        data = make_data()
        likes, dislikes = get_user_likes_dislikes(profile, data['item_map'], data['alias_map'])
        assert likes == []
        assert 'Inception' in dislikes

    def test_deduplicates_across_sessions(self):
        profile = {
            'user_id': 'TEST',
            'history': [],
            'might_like': [],
            'conversations': [
                {'conversation_id': 1, 'user_likes': ['A1'], 'user_dislikes': [], 'rec_item': []},
                {'conversation_id': 2, 'user_likes': ['A1'], 'user_dislikes': [], 'rec_item': []},
            ],
        }
        data = make_data()
        likes, _ = get_user_likes_dislikes(profile, data['item_map'], data['alias_map'])
        assert likes.count('The Matrix') == 1

    def test_empty_conversations(self):
        profile = make_profile()
        profile['conversations'] = []
        data = make_data()
        likes, dislikes = get_user_likes_dislikes(profile, data['item_map'], data['alias_map'])
        assert likes == []
        assert dislikes == []


# ── build_few_shot_prompt ──

class TestBuildFewShotPrompt:
    def test_contains_history(self):
        profile = make_profile()
        data = make_data()
        result = build_few_shot_prompt(profile, data, ["Example convo 1"])
        assert "The Matrix" in result
        assert "Inception" in result

    def test_contains_examples(self):
        profile = make_profile()
        data = make_data()
        result = build_few_shot_prompt(profile, data, ["My example conversation"])
        assert "My example conversation" in result

    def test_contains_instructions(self):
        profile = make_profile()
        data = make_data()
        result = build_few_shot_prompt(profile, data, [])
        assert "recommendation" in result.lower() or "recommend" in result.lower()

    def test_limits_history_to_15(self):
        long_history = [f"A{i}" for i in range(50)]
        profile = make_profile(history=long_history)
        item_map = {f"A{i}": f"Movie {i}" for i in range(50)}
        data = make_data(item_map=item_map)
        result = build_few_shot_prompt(profile, data, [])
        assert "Movie 14" in result
        assert "Movie 20" not in result


# ── build_agent_prompt ──

class TestBuildAgentPrompt:
    def test_contains_history(self):
        profile = make_profile()
        data = make_data()
        result = build_agent_prompt(profile, data)
        assert "The Matrix" in result

    def test_contains_tool_instructions(self):
        profile = make_profile()
        data = make_data()
        result = build_agent_prompt(profile, data)
        assert "search_catalog" in result
        assert "get_movie_details" in result
        assert "get_user_taste" in result

    def test_contains_guard_instructions(self):
        profile = make_profile()
        data = make_data()
        result = build_agent_prompt(profile, data)
        assert "NEVER recommend a movie from the watch history" in result

    def test_no_catalog_results_section(self):
        profile = make_profile()
        data = make_data()
        result = build_agent_prompt(profile, data)
        assert "CATALOG SEARCH RESULTS" not in result


# ── build_rag_prompt ──

class TestBuildRagPrompt:
    def test_contains_retrieved_movies(self):
        profile = make_profile()
        data = make_data()
        retrieved = [{"title": "Blade Runner"}, {"title": "Alien"}]
        result = build_rag_prompt(profile, data, retrieved)
        assert "Blade Runner" in result
        assert "Alien" in result

    def test_contains_history(self):
        profile = make_profile()
        data = make_data()
        result = build_rag_prompt(profile, data, [])
        assert "The Matrix" in result

    def test_no_scores_in_output(self):
        profile = make_profile()
        data = make_data()
        retrieved = [{"title": "Blade Runner", "score": 0.95}]
        result = build_rag_prompt(profile, data, retrieved)
        assert "0.95" not in result

    def test_empty_likes_shows_none_recorded(self):
        profile = make_profile()
        data = make_data()
        result = build_rag_prompt(profile, data, [])
        assert "(none recorded)" in result

    def test_likes_shown_when_present(self):
        profile = make_profile(likes=['A3'])
        data = make_data()
        result = build_rag_prompt(profile, data, [])
        assert "Interstellar" in result
        assert "(none recorded)" not in result or result.count("(none recorded)") == 1
