"""Stream the Neo4j recommendations graph into Memgraph over Bolt."""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

from .config import MigrationSettings

MIGRATION_ID_PROPERTY = "__neo4j_migration_element_id__"
NODE_STREAM_QUERY = """
MATCH (node)
RETURN elementId(node) AS source_id,
       labels(node) AS labels,
       properties(node) AS properties
"""
RELATIONSHIP_STREAM_QUERY = """
MATCH (start)-[relationship]->(end)
RETURN elementId(start) AS start_id,
       elementId(end) AS end_id,
       type(relationship) AS relationship_type,
       properties(relationship) AS properties
"""


def escape_identifier(identifier: str) -> str:
    """Quote a Cypher identifier after rejecting control characters."""
    if not identifier or "\x00" in identifier:
        raise ValueError(f"Invalid Cypher identifier: {identifier!r}")
    return f"`{identifier.replace('`', '``')}`"


def node_write_query(labels: Iterable[str]) -> str:
    label_expression = "".join(f":{escape_identifier(label)}" for label in labels)
    migration_property = escape_identifier(MIGRATION_ID_PROPERTY)
    return f"""
UNWIND $rows AS row
CREATE (node{label_expression})
SET node = row.properties
SET node.{migration_property} = row.source_id
"""


def relationship_write_query(relationship_type: str) -> str:
    migration_property = escape_identifier(MIGRATION_ID_PROPERTY)
    relationship_identifier = escape_identifier(relationship_type)
    return f"""
UNWIND $rows AS row
MATCH (start {{{migration_property}: row.start_id}})
MATCH (end {{{migration_property}: row.end_id}})
CREATE (start)-[relationship:{relationship_identifier}]->(end)
SET relationship = row.properties
"""


def _as_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "data"):
        return record.data()
    return dict(record)


def grouped_batches(
    records: Iterable[Any],
    *,
    group_key: Callable[[dict[str, Any]], Any],
    batch_size: int,
) -> Iterator[tuple[Any, list[dict[str, Any]]]]:
    """Group records by query shape while keeping total buffered rows bounded."""
    groups: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
    buffered = 0
    for raw_record in records:
        record = _as_dict(raw_record)
        groups[group_key(record)].append(record)
        buffered += 1
        if buffered >= batch_size:
            yield from groups.items()
            groups.clear()
            buffered = 0
    yield from groups.items()


@dataclass(frozen=True, slots=True)
class GraphStats:
    nodes: int
    relationships: int
    labels: dict[str, int]
    relationship_types: dict[str, int]
    movie_plots: int
    nemotron_embeddings: int
    embedding_dimensions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MigrationReport:
    stats: GraphStats
    vector_capacity: int
    node_batches: int
    relationship_batches: int


def _execute(driver: Any, database: str, query: str, **parameters: Any) -> list[Any]:
    records, _, _ = driver.execute_query(
        query,
        parameters_=parameters or None,
        database_=database,
    )
    return list(records)


def _execute_autocommit(
    driver: Any, database: str, query: str, **parameters: Any
) -> list[Any]:
    """Execute commands that Memgraph does not allow in explicit transactions."""
    with driver.session(database=database) as session:
        return list(session.run(query, parameters or None))


def collect_stats(driver: Any, database: str) -> GraphStats:
    node_record = _as_dict(
        _execute(driver, database, "MATCH (n) RETURN count(n) AS count")[0]
    )
    relationship_record = _as_dict(
        _execute(driver, database, "MATCH ()-[r]->() RETURN count(r) AS count")[0]
    )
    label_records = _execute(
        driver,
        database,
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count",
    )
    relationship_type_records = _execute(
        driver,
        database,
        "MATCH ()-[r]->() RETURN type(r) AS relationship_type, count(*) AS count",
    )
    movie_record = _as_dict(
        _execute(
            driver,
            database,
            """
            MATCH (movie:Movie)
            RETURN count(CASE WHEN movie.plot IS NOT NULL THEN 1 END) AS plots,
                   count(CASE WHEN movie.plotEmbeddingNemotron IS NOT NULL THEN 1 END)
                       AS embeddings,
                   collect(DISTINCT CASE
                       WHEN movie.plotEmbeddingNemotron IS NOT NULL
                       THEN size(movie.plotEmbeddingNemotron)
                   END) AS dimensions
            """,
        )[0]
    )
    dimensions = tuple(
        sorted(dimension for dimension in movie_record["dimensions"] if dimension)
    )
    return GraphStats(
        nodes=int(node_record["count"]),
        relationships=int(relationship_record["count"]),
        labels={
            str(row["label"]): int(row["count"]) for row in map(_as_dict, label_records)
        },
        relationship_types={
            str(row["relationship_type"]): int(row["count"])
            for row in map(_as_dict, relationship_type_records)
        },
        movie_plots=int(movie_record["plots"]),
        nemotron_embeddings=int(movie_record["embeddings"]),
        embedding_dimensions=dimensions,
    )


def validate_stats(
    source: GraphStats, target: GraphStats, expected_dimensions: int
) -> None:
    mismatches: list[str] = []
    for field in (
        "nodes",
        "relationships",
        "labels",
        "relationship_types",
        "movie_plots",
        "nemotron_embeddings",
        "embedding_dimensions",
    ):
        if getattr(source, field) != getattr(target, field):
            mismatches.append(
                f"{field}: source={getattr(source, field)!r}, target={getattr(target, field)!r}"
            )
    invalid_dimensions = [
        dimension
        for dimension in source.embedding_dimensions
        if dimension != expected_dimensions
    ]
    if invalid_dimensions:
        mismatches.append(
            f"Nemotron embeddings must have {expected_dimensions} dimensions; "
            f"found {invalid_dimensions}"
        )
    if mismatches:
        raise RuntimeError("Migration validation failed:\n- " + "\n- ".join(mismatches))


def vector_capacity(
    embedding_count: int,
    *,
    override: int,
    headroom: float,
) -> int:
    required = max(1, embedding_count)
    if override:
        if override < required:
            raise ValueError(
                f"MEMGRAPH_VECTOR_CAPACITY={override} is below the required {required}"
            )
        return override
    return max(required, math.ceil(required * headroom))


def _drop_vector_index_if_present(driver: Any, database: str, index_name: str) -> bool:
    records = _execute_autocommit(driver, database, "SHOW INDEX INFO")
    present = any(
        index_name in {str(value) for value in _as_dict(record).values()}
        and any("vector" in str(value).lower() for value in _as_dict(record).values())
        for record in records
    )
    if present:
        _execute_autocommit(
            driver, database, f"DROP VECTOR INDEX {escape_identifier(index_name)}"
        )
    return present


def create_vector_index(
    driver: Any,
    database: str,
    *,
    index_name: str,
    dimensions: int,
    capacity: int,
) -> None:
    _drop_vector_index_if_present(driver, database, index_name)
    query = f"""
CREATE VECTOR INDEX {escape_identifier(index_name)}
ON :Movie(plotEmbeddingNemotron)
WITH CONFIG {{'dimension': {dimensions}, 'capacity': {capacity}, 'metric': 'cos'}}
"""
    _execute_autocommit(driver, database, query)


def _stream(driver: Any, database: str, query: str, fetch_size: int) -> Iterator[Any]:
    with driver.session(database=database, fetch_size=fetch_size) as session:
        yield from session.run(query)


def migrate(
    source_driver: Any,
    target_driver: Any,
    settings: MigrationSettings,
    *,
    replace: bool = False,
) -> MigrationReport:
    target_count = int(
        _as_dict(
            _execute(
                target_driver,
                settings.memgraph.database,
                "MATCH (n) RETURN count(n) AS count",
            )[0]
        )["count"]
    )
    if target_count and not replace:
        raise RuntimeError(
            "Memgraph is not empty. Re-run with --replace to explicitly replace its graph data."
        )
    source_stats = collect_stats(source_driver, settings.source_database)
    if replace:
        _drop_vector_index_if_present(
            target_driver, settings.memgraph.database, settings.vector_index
        )
        _execute(target_driver, settings.memgraph.database, "MATCH (n) DETACH DELETE n")

    node_batches = 0
    node_records = _stream(
        source_driver,
        settings.source_database,
        NODE_STREAM_QUERY,
        settings.batch_size,
    )
    with tqdm(
        total=source_stats.nodes,
        desc="Migrating nodes",
        unit="node",
        dynamic_ncols=True,
    ) as progress:
        for labels, rows in grouped_batches(
            node_records,
            group_key=lambda row: tuple(sorted(row["labels"])),
            batch_size=settings.batch_size,
        ):
            for row in rows:
                if MIGRATION_ID_PROPERTY in row["properties"]:
                    raise RuntimeError(
                        f"Source data already uses reserved property {MIGRATION_ID_PROPERTY!r}"
                    )
            _execute(
                target_driver,
                settings.memgraph.database,
                node_write_query(labels),
                rows=rows,
            )
            node_batches += 1
            progress.update(len(rows))
            progress.set_postfix(batches=node_batches, refresh=False)

    relationship_batches = 0
    relationship_records = _stream(
        source_driver,
        settings.source_database,
        RELATIONSHIP_STREAM_QUERY,
        settings.batch_size,
    )
    with tqdm(
        total=source_stats.relationships,
        desc="Migrating relationships",
        unit="relationship",
        dynamic_ncols=True,
    ) as progress:
        for relationship_type, rows in grouped_batches(
            relationship_records,
            group_key=lambda row: row["relationship_type"],
            batch_size=settings.batch_size,
        ):
            _execute(
                target_driver,
                settings.memgraph.database,
                relationship_write_query(relationship_type),
                rows=rows,
            )
            relationship_batches += 1
            progress.update(len(rows))
            progress.set_postfix(batches=relationship_batches, refresh=False)
    target_stats = collect_stats(target_driver, settings.memgraph.database)
    validate_stats(source_stats, target_stats, settings.embedding_dimensions)
    capacity = vector_capacity(
        target_stats.nemotron_embeddings,
        override=settings.vector_capacity,
        headroom=settings.vector_capacity_headroom,
    )
    create_vector_index(
        target_driver,
        settings.memgraph.database,
        index_name=settings.vector_index,
        dimensions=settings.embedding_dimensions,
        capacity=capacity,
    )
    _execute(
        target_driver,
        settings.memgraph.database,
        f"MATCH (n) REMOVE n.{escape_identifier(MIGRATION_ID_PROPERTY)}",
    )
    return MigrationReport(target_stats, capacity, node_batches, relationship_batches)


def _rate_limited_call(
    function: Callable[[], list[list[float]]],
    *,
    attempts: int = 4,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[list[float]]:
    for attempt in range(attempts):
        try:
            return function()
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            response = getattr(exc, "response", None)
            if status_code is None and response is not None:
                status_code = getattr(response, "status_code", None)
            if status_code != 429 or attempt == attempts - 1:
                raise
            headers: Mapping[str, Any] = getattr(response, "headers", {}) or {}
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            try:
                delay = max(1.0, float(retry_after))
            except (TypeError, ValueError):
                delay = min(60.0, 2.0**attempt)
            sleeper(delay)
    raise AssertionError("unreachable")


def backfill_embeddings(
    target_driver: Any,
    embedder: Any,
    settings: MigrationSettings,
    *,
    embedding_model: str,
    batch_size: int,
    request_interval: float = 3.2,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    del embedding_model  # The configured embedder already owns the model name.
    _drop_vector_index_if_present(
        target_driver, settings.memgraph.database, settings.vector_index
    )
    embedded = 0
    while True:
        records = _execute(
            target_driver,
            settings.memgraph.database,
            """
            MATCH (movie:Movie)
            WHERE movie.plotEmbeddingNemotron IS NULL
              AND movie.plot IS NOT NULL
              AND trim(movie.plot) <> ''
            RETURN movie.movieId AS movie_id, movie.plot AS plot
            LIMIT $batch_size
            """,
            batch_size=batch_size,
        )
        rows = [_as_dict(record) for record in records]
        if not rows:
            break
        vectors = _rate_limited_call(
            lambda: embedder.embed_documents([row["plot"] for row in rows]),
            sleeper=sleeper,
        )
        if len(vectors) != len(rows):
            raise RuntimeError("OpenRouter returned the wrong number of embeddings")
        updates = []
        for row, vector in zip(rows, vectors, strict=True):
            if len(vector) != settings.embedding_dimensions:
                raise RuntimeError(
                    f"Expected {settings.embedding_dimensions} embedding dimensions, "
                    f"got {len(vector)}"
                )
            updates.append({"movie_id": row["movie_id"], "embedding": vector})
        _execute(
            target_driver,
            settings.memgraph.database,
            """
            UNWIND $rows AS row
            MATCH (movie:Movie {movieId: row.movie_id})
            SET movie.plotEmbeddingNemotron = row.embedding
            """,
            rows=updates,
        )
        embedded += len(rows)
        if request_interval:
            sleeper(request_interval)

    stats = collect_stats(target_driver, settings.memgraph.database)
    invalid_dimensions = [
        dimension
        for dimension in stats.embedding_dimensions
        if dimension != settings.embedding_dimensions
    ]
    if invalid_dimensions:
        raise RuntimeError(f"Invalid stored embedding dimensions: {invalid_dimensions}")
    capacity = vector_capacity(
        stats.nemotron_embeddings,
        override=settings.vector_capacity,
        headroom=settings.vector_capacity_headroom,
    )
    create_vector_index(
        target_driver,
        settings.memgraph.database,
        index_name=settings.vector_index,
        dimensions=settings.embedding_dimensions,
        capacity=capacity,
    )
    return embedded


def _drivers(settings: MigrationSettings) -> tuple[Any, Any]:
    source = GraphDatabase.driver(
        settings.source_uri,
        auth=(settings.source_username, settings.source_password),
    )
    target = GraphDatabase.driver(settings.memgraph.uri, auth=settings.memgraph.auth)
    return source, target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly clear Memgraph before migrating.",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    settings = MigrationSettings.from_env()
    source, target = _drivers(settings)
    try:
        source.verify_connectivity()
        target.verify_connectivity()
        report = migrate(source, target, settings, replace=args.replace)
    finally:
        source.close()
        target.close()
    print(
        f"Migrated {report.stats.nodes} nodes and {report.stats.relationships} relationships "
        f"in {report.node_batches + report.relationship_batches} batches. "
        f"Vector index capacity: {report.vector_capacity}."
    )
    return 0


def backfill_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill missing Memgraph movie embeddings."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="OpenRouter embedding request batch size.",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    settings = MigrationSettings.from_env()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY before backfilling embeddings.")

    from langchain_openai import OpenAIEmbeddings

    embedding_model = os.getenv("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free")
    embedder = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        max_retries=0,
        check_embedding_ctx_length=False,
    )
    target = GraphDatabase.driver(settings.memgraph.uri, auth=settings.memgraph.auth)
    try:
        target.verify_connectivity()
        count = backfill_embeddings(
            target,
            embedder,
            settings,
            embedding_model=embedding_model,
            batch_size=args.batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        )
    finally:
        target.close()
    print(f"Backfilled {count} movie embeddings and rebuilt {settings.vector_index}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
