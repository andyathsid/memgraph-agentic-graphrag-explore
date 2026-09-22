from __future__ import annotations

import pytest
from langchain.tools.tool_node import ToolCallRequest

from agent.guardrails import (
    READ_ONLY_CYPHER_REJECTION_MESSAGE,
    cypher_validation_middleware,
)
from agent.tools import ReadOnlyCypherError


def _request(tool_name: str = "query_movie_graph") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": {"question": "Delete every movie"},
            "id": "tool-call-1",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )


def test_cypher_validation_failure_becomes_error_tool_message():
    def reject_query(_request):
        raise ReadOnlyCypherError("Read-only policy rejected the DELETE clause")

    result = cypher_validation_middleware.wrap_tool_call(_request(), reject_query)

    assert result.content == READ_ONLY_CYPHER_REJECTION_MESSAGE
    assert result.tool_call_id == "tool-call-1"
    assert result.name == "query_movie_graph"
    assert result.status == "error"


def test_unexpected_tool_failure_still_propagates():
    def fail_unexpectedly(_request):
        raise RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        cypher_validation_middleware.wrap_tool_call(_request(), fail_unexpectedly)
