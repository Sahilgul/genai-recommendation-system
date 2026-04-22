import sys
from pathlib import Path

from langchain_core.messages import ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.prompts import build_prompt
from llm import chat_with_tools, stream_chat

MAX_TOOL_ROUNDS = 3
MCP_SERVER_PATH = str(Path(__file__).parent.parent / "mcp_server.py")


def _mcp_to_openai_schema(mcp_tools):
    schemas = []
    for t in mcp_tools:
        props = {}
        input_schema = t.inputSchema or {}
        for pname, pdef in input_schema.get("properties", {}).items():
            props[pname] = {
                "type": pdef.get("type", "string"),
                "description": pdef.get("description", ""),
            }
        required = input_schema.get("required", [])
        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        })
    return schemas


async def _run_mcp_tool_loop(session, tool_schemas, messages):
    for _ in range(MAX_TOOL_ROUNDS):
        response = await chat_with_tools(messages, tool_schemas)

        if not response.tool_calls:
            return messages

        messages.append(response)

        for tc in response.tool_calls:
            result = await session.call_tool(tc["name"], tc["args"])
            content = result.content[0].text if result.content else "{}"
            messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))

    return messages


async def stream_recommendation(profile, data, chat_history, user_message):
    from mcp_server import set_context
    set_context(profile, data)

    system = build_prompt(profile, data)
    messages = [{"role": "system", "content": system}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    python_exe = sys.executable
    server_params = StdioServerParameters(
        command=python_exe,
        args=[MCP_SERVER_PATH],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            tool_schemas = _mcp_to_openai_schema(mcp_tools.tools)

            messages = await _run_mcp_tool_loop(session, tool_schemas, messages)

    messages.append({
        "role": "user",
        "content": (
            "You have gathered enough information from the tools above. "
            "Now write the final movie recommendation directly. "
            "Do not propose any more searches or tool calls. "
            "Just give the user 2-3 specific movie titles with a brief reason for each, "
            "tied to their watch history."
        ),
    })

    async for token in stream_chat(messages):
        yield token
