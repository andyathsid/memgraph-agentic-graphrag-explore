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
_READ_PREFIX_PATTERN = re.compile(
    r"^\s*(?:OPTIONAL\s+MATCH|MATCH|WITH|UNWIND|RETURN)\b", re.IGNORECASE
)
_MUTATING_CLAUSE_PATTERNS = {
    clause: re.compile(
        r"\b" + r"\s+".join(map(re.escape, clause.split())) + r"\b",
        re.IGNORECASE,
    )
    for clause in _MUTATING_CLAUSES
}


def _strip_literals_and_comments(query: str) -> str:
    """Remove content that should not participate in clause checks."""
    visible_characters: list[str] = []
    position = 0

    while position < len(query):
        if query.startswith("//", position):
            comment_end = query.find("\n", position + 2)
            position = len(query) if comment_end < 0 else comment_end
            visible_characters.append(" ")
            continue

        if query.startswith("/*", position):
            comment_end = query.find("*/", position + 2)
            position = len(query) if comment_end < 0 else comment_end + 2
            visible_characters.append(" ")
            continue

        character = query[position]
        if character in ("'", '"', "`"):
            quote = character
            position += 1

            while position < len(query):
                if query[position] == "\\":
                    position += 2
                    continue
                if query[position] == quote:
                    if position + 1 < len(query) and query[position + 1] == quote:
                        position += 2
                        continue
                    position += 1
                    break
                position += 1

            visible_characters.append(" ")
            continue

        visible_characters.append(character)
        position += 1

    return "".join(visible_characters)


def validate_read_only_cypher(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("The model generated an empty Cypher query")

    query_without_literals = _strip_literals_and_comments(query)
    if not _READ_PREFIX_PATTERN.match(query_without_literals):
        raise ValueError("Only read-only graph queries are allowed")

    statements = [
        statement
        for statement in query_without_literals.split(";")
        if statement.strip()
    ]
    if len(statements) != 1:
        raise ValueError("Only one Cypher statement is allowed")

    for clause, pattern in _MUTATING_CLAUSE_PATTERNS.items():
        if pattern.search(query_without_literals):
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
    fenced = re.search(
        r"```(?:cypher)?\s*(.*?)```",
        text,
        re.IGNORECASE | re.DOTALL,
    )
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
        parameters = {
            "index_name": vector_index,
            "top_k": top_k,
            "query_vector": query_vector,
        }
        rows = graph.query(SEARCH_MOVIES_QUERY, parameters)
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
        response = cypher_model.invoke(prompt)
        cypher = validate_read_only_cypher(extract_cypher(response))
        rows = graph.query(cypher)
        return {
            "cypher": cypher,
            "rows": [_without_embeddings(dict(row)) for row in rows[:100]],
        }

    return [search_movies, query_movie_graph]
