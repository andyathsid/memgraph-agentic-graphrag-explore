from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    SEARCH_MOVIES_QUERY,
    create_memgraph_tools,
    validate_read_only_cypher,
)


class FakeGraph:
    get_schema = "(:Person)-[:ACTED_IN]->(:Movie)"

    def __init__(self):
        self.calls = []
        self.refreshed = False

    def refresh_schema(self):
        self.refreshed = True

    def query(self, query, params=None):
        self.calls.append((query, params))
        if "vector_search.search" in query:
            return [
                {
                    "movie_id": "1",
                    "title": "Toy Story",
                    "plot": "Toys come alive.",
                    "similarity": 0.9,
                    "plotEmbeddingNemotron": [1, 2, 3],
                }
            ]
        return [{"title": "The Matrix", "plotEmbedding": [1, 2]}]


class FakeEmbedder:
    def embed_query(self, query):
        assert query
        return [0.1, 0.2, 0.3]


class FakeModel:
    def invoke(self, prompt):
        assert "ACTED_IN" in prompt
        return SimpleNamespace(
            content="```cypher\nMATCH (m:Movie) RETURN m.title AS title\n```"
        )


def make_tools():
    graph = FakeGraph()
    tools = create_memgraph_tools(
        graph,
        FakeEmbedder(),
        FakeModel(),
        vector_index="moviePlotsNemotron",
        embedding_dimensions=3,
        default_top_k=5,
    )
    return graph, {tool.name: tool for tool in tools}


def test_tools_are_registered_for_both_retrieval_routes():
    _, tools = make_tools()
    assert set(tools) == {"search_movies", "query_movie_graph"}
    assert "semantic" in SYSTEM_PROMPT
    assert "structured" in SYSTEM_PROMPT


def test_vector_search_uses_native_memgraph_query_and_hides_vectors():
    graph, tools = make_tools()
    result = tools["search_movies"].invoke({"query": "toys coming alive", "top_k": 1})

    assert graph.calls[0][0] == SEARCH_MOVIES_QUERY
    assert graph.calls[0][1]["index_name"] == "moviePlotsNemotron"
    assert graph.calls[0][1]["query_vector"] == [0.1, 0.2, 0.3]
    assert result[0]["title"] == "Toy Story"
    assert "plotEmbeddingNemotron" not in result[0]


def test_text_to_cypher_refreshes_schema_returns_query_and_hides_vectors():
    graph, tools = make_tools()
    result = tools["query_movie_graph"].invoke({"question": "Which movies exist?"})

    assert graph.refreshed
    assert result["cypher"] == "MATCH (m:Movie) RETURN m.title AS title"
    assert result["rows"] == [{"title": "The Matrix"}]


@pytest.mark.parametrize(
    "query",
    [
        "CREATE (n)",
        "MATCH (n) SET n.name = 'x' RETURN n",
        "MATCH (n) DETACH DELETE n",
        "MATCH (n) REMOVE n.name RETURN n",
        "MATCH (n) CALL db.labels() YIELD label RETURN label",
        "LOAD CSV FROM 'file:///x' AS row RETURN row",
        "MATCH (n) RETURN n; MATCH (m) RETURN m",
    ],
)
def test_read_only_validator_rejects_mutation_and_multiple_statements(query):
    with pytest.raises(ValueError):
        validate_read_only_cypher(query)


def test_read_only_validator_ignores_keywords_inside_strings_and_comments():
    query = (
        "MATCH (m:Movie) WHERE m.title = 'Set It Off' // CREATE is text\nRETURN m.title"
    )
    assert validate_read_only_cypher(query) == query


def test_vector_dimension_mismatch_fails_before_database_query():
    graph = FakeGraph()
    tools = create_memgraph_tools(
        graph,
        FakeEmbedder(),
        FakeModel(),
        vector_index="moviePlotsNemotron",
        embedding_dimensions=2048,
        default_top_k=5,
    )
    with pytest.raises(RuntimeError, match="2048-dimension"):
        tools[0].invoke({"query": "space travel"})
    assert graph.calls == []
