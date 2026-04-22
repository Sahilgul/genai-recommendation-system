import json
from mcp_server import (
    search_catalog, get_movie_details, get_user_taste, set_context, mcp,
)


MOCK_ITEM_MAP = {
    'A1': 'The Matrix',
    'A2': 'Inception',
    'A3': 'White Christmas',
}
MOCK_ALIAS_MAP = {}

MOCK_PROFILE = {
    'user_id': 'U1',
    'history': ['A1'],
    'might_like': ['A3'],
    'conversations': [
        {
            'conversation_id': 1,
            'user_likes': ['A1'],
            'user_dislikes': ['A2'],
            'rec_item': [],
        },
    ],
}

MOCK_DATA = {
    'item_map': MOCK_ITEM_MAP,
    'alias_map': MOCK_ALIAS_MAP,
    'primary_names': MOCK_ITEM_MAP,
    'profiles': [MOCK_PROFILE],
    'conversations': {1: "User: I love The Matrix. It's a great sci-fi movie about virtual reality."},
    'user_index': {'U1': 0},
}


class TestSetContext:
    def test_sets_profile_and_data(self):
        set_context(MOCK_PROFILE, MOCK_DATA)
        assert search_catalog("matrix") is not None

    def test_context_persists(self):
        set_context(MOCK_PROFILE, MOCK_DATA)
        result1 = search_catalog("matrix")
        result2 = search_catalog("matrix")
        assert result1 == result2


class TestSearchCatalog:
    def setup_method(self):
        set_context(MOCK_PROFILE, MOCK_DATA)

    def test_finds_movie(self):
        result = json.loads(search_catalog("matrix"))
        assert "The Matrix" in result

    def test_returns_empty_for_no_match(self):
        result = json.loads(search_catalog("zzzzz"))
        assert result == []

    def test_respects_limit(self):
        result = json.loads(search_catalog("", limit=1))
        assert len(result) <= 1


class TestGetMovieDetails:
    def setup_method(self):
        set_context(MOCK_PROFILE, MOCK_DATA)

    def test_returns_details_for_known_movie(self):
        result = json.loads(get_movie_details("The Matrix"))
        assert result['title'] == 'The Matrix'
        assert 'liked_by_users' in result

    def test_returns_error_for_unknown_movie(self):
        result = json.loads(get_movie_details("NonexistentMovie123"))
        assert 'error' in result


class TestGetUserTaste:
    def setup_method(self):
        set_context(MOCK_PROFILE, MOCK_DATA)

    def test_returns_taste_profile(self):
        result = json.loads(get_user_taste())
        assert 'watch_count' in result
        assert 'liked_movies' in result
        assert 'disliked_movies' in result
        assert 'top_keywords' in result

    def test_liked_movies_populated(self):
        result = json.loads(get_user_taste())
        assert len(result['liked_movies']) > 0

    def test_disliked_movies_populated(self):
        result = json.loads(get_user_taste())
        assert len(result['disliked_movies']) > 0


class TestMcpServerInstance:
    def test_server_has_name(self):
        assert mcp.name == "MovieCRS"
