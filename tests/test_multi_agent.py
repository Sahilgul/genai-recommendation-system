import pytest
from unittest.mock import patch, AsyncMock
from langchain_core.messages import AIMessage
from multi_agent.graph import (
    analyze_preferences, search_catalog,
    compose_recommendation, supervisor,
    route_after_supervisor, route_to_supervisor_or_end,
    build_graph, _get_likes_dislikes,
    CATALOG_TOOL_SCHEMAS,
)
from multi_agent.prompts import (
    SUPERVISOR_SYSTEM,
    PREFERENCE_ANALYZER_SYSTEM,
    CATALOG_EXPERT_SYSTEM,
    RECOMMENDATION_COMPOSER_SYSTEM,
)


def make_profile():
    return {
        'user_id': 'U1',
        'history': ['A1'],
        'might_like': ['A3'],
        'conversations': [
            {'conversation_id': 1, 'user_likes': ['A1'], 'user_dislikes': ['A2'], 'rec_item': []},
        ],
    }


def make_data():
    return {
        'item_map': {'A1': 'The Matrix', 'A2': 'Inception', 'A3': 'Interstellar'},
        'alias_map': {},
        'primary_names': {'A1': 'The Matrix', 'A2': 'Inception', 'A3': 'Interstellar'},
        'profiles': [make_profile()],
        'conversations': {1: "User: I love sci-fi movies like The Matrix."},
        'user_index': {'U1': 0},
    }


def make_state(phase="analyze", taste="", candidates=""):
    return {
        "user_message": "Recommend a sci-fi movie",
        "chat_history": [],
        "profile": make_profile(),
        "data": make_data(),
        "history_names": [],
        "likes": [],
        "dislikes": [],
        "taste_analysis": taste,
        "catalog_results": "",
        "candidates": candidates,
        "final_response": "",
        "phase": phase,
    }


def _ai(content):
    return AIMessage(content=content)


class TestRouteAfterSupervisor:
    def test_routes_to_analyzer_on_analyze(self):
        assert route_after_supervisor({"phase": "analyze"}) == "preference_analyzer"

    def test_routes_to_catalog_on_search(self):
        assert route_after_supervisor({"phase": "search"}) == "catalog_expert"

    def test_routes_to_composer_on_compose(self):
        assert route_after_supervisor({"phase": "compose"}) == "recommendation_composer"

    def test_defaults_to_analyzer(self):
        assert route_after_supervisor({}) == "preference_analyzer"


class TestRouteToSupervisorOrEnd:
    def test_ends_on_done(self):
        from langgraph.graph import END
        assert route_to_supervisor_or_end({"phase": "done"}) == END

    def test_returns_to_supervisor_otherwise(self):
        assert route_to_supervisor_or_end({"phase": "search"}) == "supervisor"
        assert route_to_supervisor_or_end({"phase": "compose"}) == "supervisor"


class TestSupervisor:
    @pytest.mark.asyncio
    async def test_decides_analyze_when_no_taste(self):
        with patch('multi_agent.graph.chat', new_callable=AsyncMock, return_value=_ai("analyze")):
            state = make_state()
            result = await supervisor(state)
            assert result['phase'] == 'analyze'

    @pytest.mark.asyncio
    async def test_decides_search_when_taste_exists(self):
        with patch('multi_agent.graph.chat', new_callable=AsyncMock, return_value=_ai("search")):
            state = make_state(taste="Likes sci-fi")
            result = await supervisor(state)
            assert result['phase'] == 'search'

    @pytest.mark.asyncio
    async def test_decides_compose_when_candidates_exist(self):
        with patch('multi_agent.graph.chat', new_callable=AsyncMock, return_value=_ai("compose")):
            state = make_state(taste="Likes sci-fi", candidates="1. Interstellar")
            result = await supervisor(state)
            assert result['phase'] == 'compose'

    @pytest.mark.asyncio
    async def test_populates_likes_dislikes(self):
        with patch('multi_agent.graph.chat', new_callable=AsyncMock, return_value=_ai("analyze")):
            state = make_state()
            result = await supervisor(state)
            assert len(result['likes']) > 0
            assert len(result['dislikes']) > 0
            assert len(result['history_names']) > 0


class TestGetLikesDislikes:
    def test_extracts_likes_and_dislikes(self):
        profile = make_profile()
        data = make_data()
        likes, dislikes = _get_likes_dislikes(profile, data['item_map'], data['alias_map'])
        assert 'The Matrix' in likes
        assert 'Inception' in dislikes


class TestCatalogToolSchemas:
    def test_has_two_tools(self):
        assert len(CATALOG_TOOL_SCHEMAS) == 2

    def test_tool_names(self):
        names = [s['function']['name'] for s in CATALOG_TOOL_SCHEMAS]
        assert 'search_catalog' in names
        assert 'get_movie_details' in names


class TestAnalyzePreferences:
    @pytest.mark.asyncio
    async def test_produces_taste_analysis(self):
        with patch('multi_agent.graph.chat', new_callable=AsyncMock, return_value=_ai("User enjoys sci-fi and action movies.")):
            state = make_state()
            result = await analyze_preferences(state)
            assert result['taste_analysis'] == "User enjoys sci-fi and action movies."
            assert result['phase'] == 'search'
            assert len(result['history_names']) > 0

    @pytest.mark.asyncio
    async def test_populates_likes_dislikes(self):
        with patch('multi_agent.graph.chat', new_callable=AsyncMock, return_value=_ai("Taste summary.")):
            state = make_state()
            result = await analyze_preferences(state)
            assert len(result['likes']) > 0
            assert len(result['dislikes']) > 0


class TestSearchCatalog:
    @pytest.mark.asyncio
    async def test_uses_tool_calling(self):
        tool_resp = AIMessage(content="", tool_calls=[
            {"name": "search_catalog", "args": {"query": "sci-fi"}, "id": "tc_1"},
        ])
        no_tool_resp = AIMessage(content="no more tools")
        final_resp = _ai("1. **The Matrix** - Classic sci-fi")

        with patch('multi_agent.graph.chat_with_tools', new_callable=AsyncMock, side_effect=[tool_resp, no_tool_resp]), \
             patch('multi_agent.graph.chat', new_callable=AsyncMock, return_value=final_resp):
            state = make_state(phase="search", taste="Likes sci-fi")
            result = await search_catalog(state)
            assert "Matrix" in result['candidates']
            assert result['phase'] == 'compose'
            assert "search_catalog" in result['catalog_results']

    @pytest.mark.asyncio
    async def test_handles_no_tool_calls(self):
        no_tool_resp = AIMessage(content="No tools needed")
        final_resp = _ai("No good candidates found.")

        with patch('multi_agent.graph.chat_with_tools', new_callable=AsyncMock, return_value=no_tool_resp), \
             patch('multi_agent.graph.chat', new_callable=AsyncMock, return_value=final_resp):
            state = make_state(phase="search", taste="Likes sci-fi")
            result = await search_catalog(state)
            assert result['candidates'] == "No good candidates found."
            assert "No tool calls" in result['catalog_results']


class TestComposeRecommendation:
    @pytest.mark.asyncio
    async def test_produces_final_response(self):
        with patch('multi_agent.graph.chat', new_callable=AsyncMock, return_value=_ai("I recommend Interstellar!")):
            state = make_state(phase="compose")
            state['taste_analysis'] = "Loves sci-fi"
            state['candidates'] = "1. Interstellar - Great sci-fi"
            state['history_names'] = ["The Matrix"]
            state['likes'] = ["The Matrix"]
            state['dislikes'] = ["Inception"]
            result = await compose_recommendation(state)
            assert result['final_response'] == "I recommend Interstellar!"
            assert result['phase'] == 'done'


class TestBuildGraph:
    def test_compiles_without_error(self):
        graph = build_graph()
        assert graph is not None


class TestPromptTemplates:
    def test_supervisor_has_placeholders(self):
        assert '{user_message}' in SUPERVISOR_SYSTEM
        assert '{taste_analysis}' in SUPERVISOR_SYSTEM
        assert '{candidates}' in SUPERVISOR_SYSTEM

    def test_preference_analyzer_has_placeholders(self):
        assert '{history}' in PREFERENCE_ANALYZER_SYSTEM
        assert '{likes}' in PREFERENCE_ANALYZER_SYSTEM
        assert '{dislikes}' in PREFERENCE_ANALYZER_SYSTEM
        assert '{user_message}' in PREFERENCE_ANALYZER_SYSTEM

    def test_catalog_expert_has_placeholders(self):
        assert '{taste_analysis}' in CATALOG_EXPERT_SYSTEM
        assert '{user_message}' in CATALOG_EXPERT_SYSTEM

    def test_recommendation_composer_has_placeholders(self):
        assert '{history}' in RECOMMENDATION_COMPOSER_SYSTEM
        assert '{taste_analysis}' in RECOMMENDATION_COMPOSER_SYSTEM
        assert '{candidates}' in RECOMMENDATION_COMPOSER_SYSTEM
        assert '{user_message}' in RECOMMENDATION_COMPOSER_SYSTEM


class TestStreamRecommendation:
    @pytest.mark.asyncio
    async def test_yields_characters(self):
        with patch('multi_agent.crs._get_graph') as mock_get_graph:
            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(return_value={
                "final_response": "Watch Interstellar!",
                "phase": "done",
            })
            mock_get_graph.return_value = mock_graph

            from multi_agent.crs import stream_recommendation
            tokens = []
            async for char in stream_recommendation(make_profile(), make_data(), [], "sci-fi"):
                tokens.append(char)

            assert ''.join(tokens) == "Watch Interstellar!"
