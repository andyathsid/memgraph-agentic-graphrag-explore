"""LangChain tools for native Memgraph retrieval and text-to-Cypher."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from langchain_core.tools import BaseTool, tool

from .prompts import TEXT_TO_CYPHER_PROMPT

SEARCH_MOVIES_QUERY = """
CALL vector_search.search($index_name, $top_k, $query_vector)
YIELD node, similarity
OPTIONAL MATCH (node)-[relationship]-(connected)
WITH node, similarity,
     collect(DISTINCT {
         relationship: type(relationship),
         labels: labels(connected),
         name: coalesce(connected.name, connected.title),
         rating: relationship.rating
     }) AS connected_context
RETURN node.movieId AS movie_id,
       node.title AS title,
       node.plot AS plot,
       similarity,
       connected_context
ORDER BY similarity DESC
"""

_MUTATING_CLAUSES = (
    "CREATE",
    "MERGE",
    "SET",
    "REMOVE",
    "DELETE",
    "DETACH",
    "DROP",
    "LOAD CSV",
    "FOREACH",
    "CALL",
    "GRANT",
    "DENY",
    "REVOKE",
    "ALTER",
    "RENAME",
    "TERMINATE",
    "START",
    "STOP",
)
_READ_PREFIX = re.compile(
    r"^\s*(?:OPTIONAL\s+MATCH|MATCH|WITH|UNWIND|RETURN)\b", re.IGNORECASE
)


def _strip_literals_and_comments(query: str) -> str:
    """Remove content that should not participate in clause checks."""
    result: list[str] = []
    index = 0
    while index < len(query):
        if query.startswith("//", index):
            end = query.find("\n", index + 2)
            index = len(query) if end < 0 else end
            result.append(" ")
            continue
        if query.startswith("/*", index):
            end = query.find("*/", index + 2)
            index = len(query) if end < 0 else end + 2
            result.append(" ")
            continue
        character = query[index]
        if character in ("'", '"', "`"):
            quote = character
            index += 1
            while index < len(query):
                if query[index] == "\\":
                    index += 2
                    continue
                if query[index] == quote:
                    if index + 1 < len(query) and query[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            result.append(" ")
            continue
        result.append(character)
        index += 1
    return "".join(result)


def validate_read_only_cypher(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("The model generated an empty Cypher query")
    normalized = _strip_literals_and_comments(query)
    if not _READ_PREFIX.match(normalized):
        raise ValueError("Only read-only graph queries are allowed")
    statements = [part for part in normalized.split(";") if part.strip()]
    if len(statements) != 1:
        raise ValueError("Only one Cypher statement is allowed")
    for clause in _MUTATING_CLAUSES:
        pattern = r"\b" + r"\s+".join(map(re.escape, clause.split())) + r"\b"
        if re.search(pattern, normalized, re.IGNORECASE):
            raise ValueError(f"Read-only policy rejected the {clause} clause")
    return query.rstrip("; ")


def extract_cypher(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            str(block.get("text", "")) if isinstance(block, Mapping) else str(block)
            for block in content
        )
    text = str(content).strip()
    fenced = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(r"^\s*cypher\s*:\s*", "", text, flags=re.IGNORECASE)
    return text


def _without_embeddings(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_embeddings(item)
            for key, item in value.items()
            if "embedding" not in str(key).lower()
        }
    if isinstance(value, (list, tuple)):
        return [_without_embeddings(item) for item in value]
    return value


def create_memgraph_tools(
    graph: Any,
    embedder: Any,
    cypher_model: Any,
    *,
    vector_index: str,
    embedding_dimensions: int,
    default_top_k: int,
) -> list[BaseTool]:
    """Create tools with injectable boundaries for unit testing."""

    @tool("search_movies")
    def search_movies(query: str, top_k: int = default_top_k) -> list[dict[str, Any]]:
        """Find movies with plots semantically similar to a description or theme."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        query_vector = embedder.embed_query(query)
        if len(query_vector) != embedding_dimensions:
            raise RuntimeError(
                f"Expected a {embedding_dimensions}-dimension query embedding, "
                f"got {len(query_vector)}"
            )
        rows = graph.query(
            SEARCH_MOVIES_QUERY,
            {
                "index_name": vector_index,
                "top_k": top_k,
                "query_vector": query_vector,
            },
        )
        return [_without_embeddings(dict(row)) for row in rows]

    @tool("query_movie_graph")
    def query_movie_graph(question: str) -> dict[str, Any]:
        """Answer a structured movie-graph question with generated read-only Cypher."""
        if not question.strip():
            raise ValueError("question must not be empty")
        graph.refresh_schema()
        prompt = TEXT_TO_CYPHER_PROMPT.format(
            schema=graph.get_schema,
            question=question,
        )
        cypher = validate_read_only_cypher(extract_cypher(cypher_model.invoke(prompt)))
        rows = graph.query(cypher)
        return {
            "cypher": cypher,
            "rows": [_without_embeddings(dict(row)) for row in rows[:100]],
        }

    return [search_movies, query_movie_graph]
