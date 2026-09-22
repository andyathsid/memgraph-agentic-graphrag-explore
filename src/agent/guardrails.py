"""LangChain middleware for recoverable agent guardrail failures."""

from __future__ import annotations

from langchain.agents.middleware import ToolErrorMiddleware
from langchain.tools.tool_node import ToolCallRequest

from agent.tools import ReadOnlyCypherError

READ_ONLY_CYPHER_REJECTION_MESSAGE = (
    "The generated Cypher query was rejected by the read-only policy. "
    "Try again with a question that can be answered without changing the graph."
)


def _handle_cypher_validation_error(
    error: Exception,
    _request: ToolCallRequest,
) -> str | None:
    if isinstance(error, ReadOnlyCypherError):
        return READ_ONLY_CYPHER_REJECTION_MESSAGE
    return None


cypher_validation_middleware = ToolErrorMiddleware(
    on_error=_handle_cypher_validation_error,
    tools=["query_movie_graph"],
)
