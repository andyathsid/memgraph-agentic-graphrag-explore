"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _integer(name: str, default: int, *, minimum: int = 0) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _floating(name: str, default: float, *, minimum: float = 0.0) -> float:
    value = float(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class MemgraphSettings:
    uri: str
    database: str
    username: str
    password: str

    @classmethod
    def from_env(cls) -> MemgraphSettings:
        return cls(
            uri=os.getenv("MEMGRAPH_URI", "bolt://localhost:7688"),
            database=os.getenv("MEMGRAPH_DATABASE", "memgraph"),
            username=os.getenv("MEMGRAPH_USERNAME", ""),
            password=os.getenv("MEMGRAPH_PASSWORD", ""),
        )

    @property
    def auth(self) -> tuple[str, str] | None:
        return (self.username, self.password) if self.username or self.password else None


@dataclass(frozen=True, slots=True)
class AgentSettings:
    memgraph: MemgraphSettings
    openrouter_api_key: str
    openrouter_base_url: str
    llm_model: str
    embedding_model: str
    embedding_dimensions: int
    vector_index: str
    retrieval_top_k: int

    @classmethod
    def from_env(cls) -> AgentSettings:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY before loading the agent graph.")
        return cls(
            memgraph=MemgraphSettings.from_env(),
            openrouter_api_key=api_key,
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            llm_model=os.getenv("LLM_MODEL", "poolside/laguna-s-2.1:free"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free"
            ),
            embedding_dimensions=_integer("EMBEDDING_DIMENSIONS", 2048, minimum=1),
            vector_index=os.getenv("MEMGRAPH_VECTOR_INDEX", "moviePlotsNemotron"),
            retrieval_top_k=_integer("RETRIEVAL_TOP_K", 5, minimum=1),
        )


@dataclass(frozen=True, slots=True)
class MigrationSettings:
    source_uri: str
    source_database: str
    source_username: str
    source_password: str
    memgraph: MemgraphSettings
    batch_size: int
    vector_index: str
    embedding_dimensions: int
    vector_capacity: int
    vector_capacity_headroom: float

    @classmethod
    def from_env(cls) -> MigrationSettings:
        return cls(
            source_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            source_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            source_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            source_password=os.getenv("NEO4J_PASSWORD", ""),
            memgraph=MemgraphSettings.from_env(),
            batch_size=_integer("MIGRATION_BATCH_SIZE", 500, minimum=1),
            vector_index=os.getenv("MEMGRAPH_VECTOR_INDEX", "moviePlotsNemotron"),
            embedding_dimensions=_integer("EMBEDDING_DIMENSIONS", 2048, minimum=1),
            vector_capacity=_integer("MEMGRAPH_VECTOR_CAPACITY", 0),
            vector_capacity_headroom=_floating(
                "MEMGRAPH_VECTOR_CAPACITY_HEADROOM", 1.2, minimum=1.0
            ),
        )
