import json
from tools import execute_tool, TOOL_SCHEMAS, _get_likes_dislikes, _extract_keywords


def make_profile():
    return {
        'user_id': 'U1',
        'history': ['A1', 'A2'],
        'might_like': [],
        'conversations': [
            {
                'conversation_id': 10,
                'user_likes': ['A1'],
                'user_dislikes': ['A2'],
                'rec_item': [],
            },
        ],
    }


def make_data():
    return {
        'item_map': {'A1': 'The Matrix', 'A2': 'Inception', 'A3': 'Interstellar'},
        'alias_map': {},
        'primary_names': {'A1': 'The Matrix', 'A2': 'Inception', 'A3': 'Interstellar'},
        'profiles': [make_profile()],
        'conversations': {
            10: "User: The Matrix is an amazing sci-fi film. Assistant: I agree, The Matrix changed cinema forever.",
        },
        'user_index': {'U1': 0},
    }


# ── TOOL_SCHEMAS ──

class TestToolSchemas:
    def test_has_three_tools(self):
        assert len(TOOL_SCHEMAS) == 3

    def test_all_have_function_type(self):
        for schema in TOOL_SCHEMAS:
            assert schema['type'] == 'function'
            assert 'function' in schema
            assert 'name' in schema['function']
            assert 'description' in schema['function']
            assert 'parameters' in schema['function']

    def test_tool_names(self):
        names = {s['function']['name'] for s in TOOL_SCHEMAS}
        assert names == {'search_catalog', 'get_movie_details', 'get_user_taste'}

    def test_search_catalog_requires_query(self):
        schema = next(s for s in TOOL_SCHEMAS if s['function']['name'] == 'search_catalog')
        assert 'query' in schema['function']['parameters']['properties']
        assert 'query' in schema['function']['parameters']['required']

    def test_get_movie_details_requires_movie_name(self):
        schema = next(s for s in TOOL_SCHEMAS if s['function']['name'] == 'get_movie_details')
        assert 'movie_name' in schema['function']['parameters']['properties']
        assert 'movie_name' in schema['function']['parameters']['required']


# ── execute_tool: search_catalog ──

class TestSearchCatalog:
    def test_returns_matching_movies(self):
        result = json.loads(execute_tool("search_catalog", {"query": "matrix"}, make_profile(), make_data()))
        assert "The Matrix" in result

    def test_respects_limit(self):
        result = json.loads(execute_tool("search_catalog", {"query": "a", "limit": 2}, make_profile(), make_data()))
        assert len(result) <= 2

    def test_no_results(self):
        result = json.loads(execute_tool("search_catalog", {"query": "zzzzz"}, make_profile(), make_data()))
        assert result == []


# ── execute_tool: get_movie_details ──

class TestGetMovieDetails:
    def test_returns_details(self):
        result = json.loads(execute_tool("get_movie_details", {"movie_name": "The Matrix"}, make_profile(), make_data()))
        assert result['title'] == 'The Matrix'
        assert 'liked_by_users' in result
        assert 'disliked_by_users' in result
        assert 'snippets' in result

    def test_counts_likes(self):
        result = json.loads(execute_tool("get_movie_details", {"movie_name": "The Matrix"}, make_profile(), make_data()))
        assert result['liked_by_users'] >= 1

    def test_not_found(self):
        result = json.loads(execute_tool("get_movie_details", {"movie_name": "Nonexistent Movie"}, make_profile(), make_data()))
        assert 'error' in result


# ── execute_tool: get_user_taste ──

class TestGetUserTaste:
    def test_returns_taste_profile(self):
        result = json.loads(execute_tool("get_user_taste", {}, make_profile(), make_data()))
        assert 'watch_count' in result
        assert 'liked_movies' in result
        assert 'disliked_movies' in result
        assert 'top_keywords' in result

    def test_liked_movies_populated(self):
        result = json.loads(execute_tool("get_user_taste", {}, make_profile(), make_data()))
        assert 'The Matrix' in result['liked_movies']

    def test_disliked_movies_populated(self):
        result = json.loads(execute_tool("get_user_taste", {}, make_profile(), make_data()))
        assert 'Inception' in result['disliked_movies']


# ── execute_tool: unknown ──

class TestUnknownTool:
    def test_returns_error(self):
        result = json.loads(execute_tool("nonexistent_tool", {}, make_profile(), make_data()))
        assert 'error' in result


# ── helpers ──

class TestExtractKeywords:
    def test_extracts_meaningful_words(self):
        keywords = _extract_keywords(["The Matrix", "Interstellar"])
        assert "matrix" in keywords
        assert "interstellar" in keywords

    def test_skips_short_words(self):
        keywords = _extract_keywords(["A Big Movie"])
        assert "big" not in keywords

    def test_returns_top_10(self):
        names = [f"Movie{i} ExtraWord{i}" for i in range(20)]
        keywords = _extract_keywords(names)
        assert len(keywords) <= 10


class TestGetLikesDislikes:
    def test_extracts_correctly(self):
        profile = make_profile()
        data = make_data()
        likes, dislikes = _get_likes_dislikes(profile, data['item_map'], data['alias_map'])
        assert 'The Matrix' in likes
        assert 'Inception' in dislikes
