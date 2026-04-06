"""Shared application factory. Assembles provider, tools, and agent for use by server.py, cli.py, etc."""

from agent import Agent
from prompt import build_agent_system_prompt
from providers.lm_studio import LMStudioProvider
from tools import ToolRegistry
from tools.calculate import CalculateTool
from tools.get_page import GetPageTool
from tools.search_handbook import SearchHandbookTool
from vector_store import load_vector_store


def create_agent(
    base_url: str = "http://localhost:1234/v1",
    api_key: str = "lm-studio",
    llm_model: str = "gemma-3-4b-it-qat",
    vlm_model: str = "gemma-3-12b-it-qat",
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5",
    max_iterations: int = 5,
) -> Agent:
    provider = LMStudioProvider(
        base_url=base_url,
        api_key=api_key,
        llm_model=llm_model,
        vlm_model=vlm_model,
        embedding_model=embedding_model,
    )

    collection = load_vector_store()

    registry = ToolRegistry()
    registry.register(SearchHandbookTool(provider=provider, collection=collection))
    registry.register(GetPageTool())
    registry.register(CalculateTool())

    return Agent(
        provider=provider,
        registry=registry,
        system_prompt=build_agent_system_prompt(),
        max_iterations=max_iterations,
    )
