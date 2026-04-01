"""Configuration management for the Deep Research agent.

Ported from langchain-ai/open_deep_research — LangChain RunnableConfig replaced with
env-var-based Configuration.from_env().
"""

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Available search API providers."""

    WEB_SEARCH = "web_search"
    NONE = "none"


class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""

    url: str | None = Field(default=None)
    """The URL of the MCP server."""
    tools: list[str] | None = Field(default=None)
    """The tools to make available to the LLM."""


class Configuration(BaseModel):
    """Main configuration for the Deep Research agent."""

    # General
    max_structured_output_retries: int = Field(default=3)
    allow_clarification: bool = Field(default=True)
    max_concurrent_research_units: int = Field(default=5)

    # Research
    search_api: SearchAPI = Field(default=SearchAPI.WEB_SEARCH)
    max_researcher_iterations: int = Field(default=6)
    max_react_tool_calls: int = Field(default=10)

    # Model names (AF format — no provider prefix, e.g. "gpt-4.1" not "openai:gpt-4.1")
    research_model: str = Field(default="gpt-4.1")
    research_model_max_tokens: int = Field(default=10000)
    compression_model: str = Field(default="gpt-4.1")
    compression_model_max_tokens: int = Field(default=8192)
    final_report_model: str = Field(default="gpt-4.1")
    final_report_model_max_tokens: int = Field(default=10000)

    # MCP
    mcp_config: MCPConfig | None = Field(default=None)
    mcp_prompt: str | None = Field(default=None)

    # Provider: "openai" or "azure"
    llm_provider: str = Field(default="openai")

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> Configuration:
        """Create Configuration from environment variables and optional overrides."""
        values: dict[str, Any] = {}
        for field_name in cls.model_fields:
            env_val = os.environ.get(field_name.upper())
            if env_val is not None:
                values[field_name] = env_val
        if overrides:
            values.update(overrides)
        return cls(**{k: v for k, v in values.items() if v is not None})
