from __future__ import annotations

import pytest

from agent import migration
from agent.config import MemgraphSettings, MigrationSettings


def settings() -> MigrationSettings:
    return MigrationSettings(
        source_uri="bolt://source",
        source_database="neo4j",
        source_username="neo4j",
        source_password="secret",
        memgraph=MemgraphSettings("bolt://target", "memgraph", "", ""),
        batch_size=2,
        vector_index="moviePlotsNemotron",
        embedding_dimensions=3,
        vector_capacity=0,
        vector_capacity_headroom=1.2,
    )


def stats(
    *, nodes: int = 2, dimensions: tuple[int, ...] = (3,)
) -> migration.GraphStats:
    return migration.GraphStats(
        nodes=nodes,
        relationships=1,
        labels={"Movie": 1, "Person": 1},
        relationship_types={"ACTED_IN": 1},
        movie_plots=1,
        nemotron_embeddings=1,
        embedding_dimensions=dimensions,
    )


class RecordingDriver:
    def __init__(self, target_count: int = 0):
        self.target_count = target_count
        self.calls: list[tuple[str, dict]] = []
        self.autocommit_calls: list[tuple[str, dict]] = []

    class Session:
        def __init__(self, driver):
            self.driver = driver

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def run(self, query, parameters=None):
            parameters = parameters or {}
            self.driver.calls.append((query, parameters))
            self.driver.autocommit_calls.append((query, parameters))
            if "SHOW INDEX INFO" in query:
                return []
            return []

    def session(self, database=None):
        return self.Session(self)

    def execute_query(self, query, parameters_=None, database_=None):
        parameters = parameters_ or {}
        self.calls.append((query, parameters))
        if "MATCH (n) RETURN count(n) AS count" in query:
            return ([{"count": self.target_count}], None, None)
        if "SHOW INDEX INFO" in query:
            return ([], None, None)
        return ([], None, None)


def test_grouped_node_batches_are_bounded_and_preserve_multiple_labels():
    records = [
        {"source_id": "1", "labels": ["Movie", "Featured"], "properties": {"x": 1}},
        {"source_id": "2", "labels": ["Person"], "properties": {"x": 2}},
        {"source_id": "3", "labels": ["Movie", "Featured"], "properties": {"x": 3}},
    ]
    batches = list(
        migration.grouped_batches(
            records,
            group_key=lambda row: tuple(sorted(row["labels"])),
            batch_size=2,
        )
    )

    assert sum(len(rows) for _, rows in batches) == 3
    assert all(len(rows) <= 2 for _, rows in batches)
    query = migration.node_write_query(("Featured", "Movie"))
    assert "(node:`Featured`:`Movie`)" in query
    assert "$rows" in query
    assert "x" not in query


def test_dynamic_relationship_type_is_escaped_and_properties_are_parameterized():
    query = migration.relationship_write_query("ACTED`IN")

    assert "[relationship:`ACTED``IN`]" in query
    assert "SET relationship = row.properties" in query


def test_nonempty_target_requires_replace():
    with pytest.raises(RuntimeError, match="not empty"):
        migration.migrate(object(), RecordingDriver(target_count=4), settings())


def test_replace_migrates_properties_and_cleans_up_only_after_validation(monkeypatch):
    target = RecordingDriver(target_count=2)
    node_rows = [
        {
            "source_id": "n1",
            "labels": ["Movie", "Featured"],
            "properties": {
                "movieId": "1",
                "plotEmbedding": [0.1, 0.2],
                "plotEmbeddingNemotron": [0.1, 0.2, 0.3],
            },
        },
        {"source_id": "n2", "labels": ["Person"], "properties": {"name": "Actor"}},
    ]
    relationship_rows = [
        {
            "start_id": "n2",
            "end_id": "n1",
            "relationship_type": "ACTED_IN",
            "properties": {"role": "Hero"},
        }
    ]
    streams = iter((node_rows, relationship_rows))
    monkeypatch.setattr(
        migration, "_stream", lambda *args, **kwargs: iter(next(streams))
    )
    collected = iter((stats(), stats()))
    monkeypatch.setattr(migration, "collect_stats", lambda *args: next(collected))
    vector_calls = []
    monkeypatch.setattr(
        migration,
        "create_vector_index",
        lambda *args, **kwargs: vector_calls.append(kwargs),
    )

    report = migration.migrate(object(), target, settings(), replace=True)

    assert report.stats.nodes == 2
    assert vector_calls[0]["capacity"] == 2
    assert any("DETACH DELETE" in query for query, _ in target.calls)
    assert any("REMOVE n." in query for query, _ in target.calls)
    written_rows = [
        parameters["rows"]
        for query, parameters in target.calls
        if "UNWIND $rows" in query
    ]
    assert node_rows[0]["properties"] in [
        row["properties"] for batch in written_rows for row in batch
    ]
    assert relationship_rows[0]["properties"] in [
        row["properties"] for batch in written_rows for row in batch
    ]


def test_validation_failure_leaves_migration_ids_for_diagnosis(monkeypatch):
    target = RecordingDriver()
    streams = iter(([], []))
    monkeypatch.setattr(
        migration, "_stream", lambda *args, **kwargs: iter(next(streams))
    )
    collected = iter((stats(nodes=2), stats(nodes=1)))
    monkeypatch.setattr(migration, "collect_stats", lambda *args: next(collected))

    with pytest.raises(RuntimeError, match="nodes"):
        migration.migrate(object(), target, settings())

    assert not any("REMOVE n." in query for query, _ in target.calls)


def test_validation_rejects_wrong_embedding_dimension():
    invalid = stats(dimensions=(2, 3))
    with pytest.raises(RuntimeError, match="must have 3 dimensions"):
        migration.validate_stats(invalid, invalid, 3)


def test_vector_capacity_override_and_index_configuration():
    assert migration.vector_capacity(10, override=0, headroom=1.2) == 12
    with pytest.raises(ValueError, match="below"):
        migration.vector_capacity(10, override=9, headroom=1.2)

    driver = RecordingDriver()
    migration.create_vector_index(
        driver,
        "memgraph",
        index_name="moviePlotsNemotron",
        dimensions=2048,
        capacity=1200,
    )
    query = driver.calls[-1][0]
    assert "CREATE VECTOR INDEX `moviePlotsNemotron`" in query
    assert "'dimension': 2048" in query
    assert "'capacity': 1200" in query
    assert "'metric': 'cos'" in query
    assert driver.autocommit_calls[-1][0] == query


def test_rate_limit_retry_uses_server_delay():
    attempts = 0
    delays = []

    class RateLimited(Exception):
        status_code = 429
        response = type("Response", (), {"headers": {"Retry-After": "2"}})()

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimited()
        return [[0.1, 0.2, 0.3]]

    result = migration._rate_limited_call(operation, sleeper=delays.append)

    assert result == [[0.1, 0.2, 0.3]]
    assert attempts == 2
    assert delays == [2.0]


def test_reserved_migration_property_is_rejected(monkeypatch):
    target = RecordingDriver()
    monkeypatch.setattr(migration, "collect_stats", lambda *args: stats(nodes=1))
    streams = iter(
        (
            [
                {
                    "source_id": "n1",
                    "labels": ["Movie"],
                    "properties": {migration.MIGRATION_ID_PROPERTY: "collision"},
                }
            ],
            [],
        )
    )
    monkeypatch.setattr(
        migration, "_stream", lambda *args, **kwargs: iter(next(streams))
    )

    with pytest.raises(RuntimeError, match="reserved property"):
        migration.migrate(object(), target, settings())
