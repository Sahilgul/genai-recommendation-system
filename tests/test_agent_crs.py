import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import AIMessage, ToolMessage
from agent.crs import _run_mcp_tool_loop, _mcp_to_groq_schema, MAX_TOOL_ROUNDS


def _make_mcp_tool(name, description, properties=None, required=None):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
    }
    return tool


def _make_ai_message(content="", tool_calls=None):
    return AIMessage(content=content, tool_calls=tool_calls or [])


def _make_tool_call_msg(name, args, tc_id="tc_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": tc_id}])


class TestMcpToGroqSchema:
    def test_converts_single_tool(self):
        mcp_tools = [_make_mcp_tool(
            "search_catalog",
            "Search movies",
            {"query": {"type": "string", "description": "Search query"}},
            ["query"],
        )]
        schemas = _mcp_to_groq_schema(mcp_tools)
        assert len(schemas) == 1
        assert schemas[0]['type'] == 'function'
        assert schemas[0]['function']['name'] == 'search_catalog'
        assert 'query' in schemas[0]['function']['parameters']['properties']
        assert 'query' in schemas[0]['function']['parameters']['required']

    def test_converts_multiple_tools(self):
        mcp_tools = [
            _make_mcp_tool("tool_a", "desc a"),
            _make_mcp_tool("tool_b", "desc b"),
            _make_mcp_tool("tool_c", "desc c"),
        ]
        schemas = _mcp_to_groq_schema(mcp_tools)
        assert len(schemas) == 3
        names = [s['function']['name'] for s in schemas]
        assert names == ['tool_a', 'tool_b', 'tool_c']

    def test_handles_empty_schema(self):
        tool = _make_mcp_tool("get_user_taste", "Get taste", {}, [])
        schemas = _mcp_to_groq_schema([tool])
        assert schemas[0]['function']['parameters']['properties'] == {}


class TestRunMcpToolLoop:
    @pytest.mark.asyncio
    async def test_returns_messages_when_no_tool_calls(self):
        mock_resp = _make_ai_message(content="No tools needed")

        session = AsyncMock()
        schemas = []

        with patch('agent.crs.chat_with_tools', new_callable=AsyncMock, return_value=mock_resp):
            messages = [{"role": "user", "content": "hi"}]
            result = await _run_mcp_tool_loop(session, schemas, messages)
            assert result == messages
            session.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_mcp_session_for_tool(self):
        tool_resp = _make_tool_call_msg("search_catalog", {"query": "matrix"})
        final_resp = _make_ai_message(content="Done")

        mcp_result = MagicMock()
        mcp_content = MagicMock()
        mcp_content.text = '["The Matrix"]'
        mcp_result.content = [mcp_content]

        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=mcp_result)

        with patch('agent.crs.chat_with_tools', new_callable=AsyncMock, side_effect=[tool_resp, final_resp]):
            messages = [{"role": "user", "content": "recommend sci-fi"}]
            result = await _run_mcp_tool_loop(session, [], messages)

            session.call_tool.assert_called_once_with("search_catalog", {"query": "matrix"})
            tool_messages = [m for m in result if isinstance(m, ToolMessage)]
            assert len(tool_messages) == 1
            assert tool_messages[0].content == '["The Matrix"]'

    @pytest.mark.asyncio
    async def test_respects_max_rounds(self):
        tool_resp = _make_tool_call_msg("get_user_taste", {})

        mcp_result = MagicMock()
        mcp_content = MagicMock()
        mcp_content.text = '{"watch_count": 5}'
        mcp_result.content = [mcp_content]

        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=mcp_result)

        with patch('agent.crs.chat_with_tools', new_callable=AsyncMock, return_value=tool_resp):
            messages = [{"role": "user", "content": "hi"}]
            result = await _run_mcp_tool_loop(session, [], messages)

            tool_messages = [m for m in result if isinstance(m, ToolMessage)]
            assert len(tool_messages) == MAX_TOOL_ROUNDS

    @pytest.mark.asyncio
    async def test_handles_empty_mcp_content(self):
        tool_resp = _make_tool_call_msg("search_catalog", {"query": "x"})
        final_resp = _make_ai_message(content="Done")

        mcp_result = MagicMock()
        mcp_result.content = []

        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=mcp_result)

        with patch('agent.crs.chat_with_tools', new_callable=AsyncMock, side_effect=[tool_resp, final_resp]):
            messages = [{"role": "user", "content": "hi"}]
            result = await _run_mcp_tool_loop(session, [], messages)

            tool_messages = [m for m in result if isinstance(m, ToolMessage)]
            assert tool_messages[0].content == '{}'
