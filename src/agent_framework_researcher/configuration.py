"""Configuration management for the Deep Research agent.

Uses pydantic-settings BaseSettings for automatic environment variable loading.
All fields are read from env vars matching their uppercase field names
(e.g. ``DEFAULT_MODEL``, ``LLM_ENDPOINT``).
"""

from enum import Enum

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class Configuration(BaseSettings):
    """Main configuration for the Deep Research agent.

    All fields are automatically loaded from environment variables.
    Just call ``Configuration()`` — no explicit env loading needed.

    Per-task model fields (``research_model``, ``compression_model``,
    ``final_report_model``) fall back to ``default_model`` when not set.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    max_structured_output_retries: int = Field(default=3)
    allow_clarification: bool = Field(default=True)
    max_concurrent_research_units: int = Field(default=5)

    # Research
    search_api: SearchAPI = Field(default=SearchAPI.WEB_SEARCH)
    max_researcher_iterations: int = Field(default=6)
    max_react_tool_calls: int = Field(default=10)

    # Models — per-task models fall back to default_model when not set
    default_model: str = Field(default="gpt-4.1")
    research_model: str | None = Field(default=None)
    research_model_max_tokens: int = Field(default=10000)
    compression_model: str | None = Field(default=None)
    compression_model_max_tokens: int = Field(default=10000)
    final_report_model: str | None = Field(default=None)
    final_report_model_max_tokens: int = Field(default=50000)

    # MCP
    mcp_config: MCPConfig | None = Field(default=None)
    mcp_prompt: str | None = Field(default=None)

    # LLM provider: "openai" or "azure"
    llm_provider: str = Field(default="openai")

    # LLM credentials
    llm_api_key: str | None = Field(default=None)
    llm_endpoint: str | None = Field(default=None)

    @model_validator(mode="after")
    def _apply_model_defaults(self) -> Configuration:
        """Fill per-task models from default_model when not explicitly set."""
        if self.research_model is None:
            self.research_model = self.default_model
        if self.compression_model is None:
            self.compression_model = self.default_model
        if self.final_report_model is None:
            self.final_report_model = self.default_model
        return self
