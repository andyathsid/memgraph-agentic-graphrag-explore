from __future__ import annotations

import json
import os
import sys
from typing import Any

import pytest
from dotenv import load_dotenv
from langchain.messages import AIMessage, ToolMessage

load_dotenv()

pytestmark = pytest.mark.integration

_MAX_TOOL_CONTENT_CHARS = 800
_requires_live_agent = pytest.mark.skipif(
    os.getenv("RUN_LIVE_AGENT_TESTS") != "1"
    or not os.getenv("OPENROUTER_API_KEY"),
    reason="Set RUN_LIVE_AGENT_TESTS=1 and OPENROUTER_API_KEY, and start Memgraph.",
)


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="backslashreplace").decode(encoding))


def _render(message: Any) -> str:
    if isinstance(message, ToolMessage):
        content = message.content
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
            content = json.dumps(parsed, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            content = str(content)

        if len(content) > _MAX_TOOL_CONTENT_CHARS:
            content = (
                content[:_MAX_TOOL_CONTENT_CHARS]
                + f"\n... [truncated, {len(content)} chars total]"
            )

        header = (
            "================================= Tool Message "
            "=================================\n"
            f"Name: {message.name}"
        )
        return f"{header}\n{content}"

    return message.pretty_repr()


def _run_agent(agent: Any, agent_input: dict[str, Any]) -> dict[str, Any]:
    """Run the agent and print each new message as the graph advances."""
    final_state = None
    seen = 0

    for state in agent.stream(agent_input, stream_mode="values"):
        final_state = state
        messages = state["messages"]
        for message in messages[seen:]:
            _safe_print("")
            _safe_print(_render(message))
            _safe_print("")
        seen = len(messages)

    assert final_state is not None
    return final_state


@_requires_live_agent
def test_agent_uses_semantic_search_and_returns_an_answer() -> None:
    from agent.graph import graph

    result = _run_agent(
        graph,
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use the semantic movie search tool to find movies about "
                        "toys coming alive. Base your answer on the tool results."
                    ),
                }
            ],
        },
    )

    messages = result["messages"]
    search_results = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.name == "search_movies"
    ]

    assert search_results, "The agent did not call search_movies."
    assert any("Toy Story" in str(message.content) for message in search_results)
    assert isinstance(messages[-1], AIMessage)
    assert str(messages[-1].content).strip()


@_requires_live_agent
def test_agent_uses_text_to_cypher_and_returns_an_answer() -> None:
    from agent.graph import graph

    result = _run_agent(
        graph,
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use the structured movie graph query tool to list movies "
                        "in which Hugo Weaving acted. Base your answer on the tool "
                        "results."
                    ),
                }
            ],
        },
    )

    messages = result["messages"]
    graph_results = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.name == "query_movie_graph"
    ]

    assert graph_results, "The agent did not call query_movie_graph."
    assert any("V for Vendetta" in str(message.content) for message in graph_results)
    assert isinstance(messages[-1], AIMessage)
    assert str(messages[-1].content).strip()
