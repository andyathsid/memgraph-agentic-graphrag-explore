"""Aegra-compatible LangChain agent backed by Memgraph."""

from __future__ import annotations

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_memgraph.graphs.memgraph import MemgraphLangChain
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter

from agent.config import AgentSettings
from agent.prompts import SYSTEM_PROMPT
from agent.tools import create_memgraph_tools


def build_agent(settings: AgentSettings):
    """Build the compiled agent graph from explicit settings."""
    model = ChatOpenRouter(
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
    )
    embedder = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        check_embedding_ctx_length=False,
    )
    memgraph = MemgraphLangChain(
        url=settings.memgraph.uri,
        username=settings.memgraph.username,
        password=settings.memgraph.password,
        database=settings.memgraph.database,
        refresh_schema=False,
    )
    tools = create_memgraph_tools(
        memgraph,
        embedder,
        model,
        vector_index=settings.vector_index,
        embedding_dimensions=settings.embedding_dimensions,
        default_top_k=settings.retrieval_top_k,
    )
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        name="agent"
    )


load_dotenv()
graph = build_agent(AgentSettings.from_env())
