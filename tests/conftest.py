"""Shared test fixtures."""

import pytest


@pytest.fixture
def sample_config():
    """Return a Configuration with defaults for testing."""
    from agent_framework_researcher.configuration import Configuration

    return Configuration(
        llm_provider="openai",
        search_api="tavily",
        allow_clarification=True,
        max_researcher_iterations=3,
        max_react_tool_calls=5,
    )
