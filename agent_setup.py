"""Shared application factory. Assembles tools and agent for use by server.py, cli.py, etc."""

from agent import Agent
from llm import provider
from prompt import build_agent_system_prompt
from tools import ToolRegistry
from tools.calculate import CalculateTool
from tools.get_page import GetPageTool
from tools.search_handbook import SearchHandbookTool
from vector_store import load_vector_store


def create_agent(max_iterations: int = 5) -> Agent:
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
