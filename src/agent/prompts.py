"""Prompts shared by the graph and its tools."""

SYSTEM_PROMPT = """You answer questions about a movie recommendations graph in Memgraph.

Use search_movies for semantic questions about plot meaning, themes, or similarity.
Use query_movie_graph for structured questions about titles, people, genres, release years,
and ratings. You may use both tools when a question combines semantic and graph criteria.
Base factual answers on tool results, say when no matching data was found, and never claim
that you changed the database. Database mutation requests must be refused.
"""

TEXT_TO_CYPHER_PROMPT = """Generate exactly one read-only Memgraph Cypher query for the
question below. Return only Cypher, without Markdown or explanation.

Rules:
- Use only labels, relationship types, and properties present in the schema.
- Use MATCH, OPTIONAL MATCH, WITH, UNWIND, WHERE, RETURN, ORDER BY, SKIP, and LIMIT only.
- Never use a write, administration, LOAD CSV, procedure, or subquery clause.
- Do not return embedding properties or entire nodes. Return named scalar properties.
- Limit broad result sets to 100 rows.

Schema:
{schema}

Question:
{question}
"""

