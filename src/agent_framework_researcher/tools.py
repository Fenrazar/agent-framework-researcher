"""Research tools: web search, think tool, MCP loader, and helpers.

Ported from open_deep_research/utils.py — all LangChain dependencies removed.
Web search uses AF's native OpenAIChatClient.get_web_search_tool().
"""

import logging
from datetime import datetime

from agent_framework.openai import OpenAIChatClient

from agent_framework_researcher.configuration import Configuration, SearchAPI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Think tool
# ---------------------------------------------------------------------------

def think(reflection: str) -> str:
    """Strategic reflection tool for research planning.

    Use after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    Args:
        reflection: Detailed reflection on research progress, findings, gaps, and next steps.

    Returns:
        Confirmation that reflection was recorded.
    """
    return f"Reflection recorded: {reflection}"


# ---------------------------------------------------------------------------
# MCP tool loading
# ---------------------------------------------------------------------------

async def load_mcp_tools(client: OpenAIChatClient, config: Configuration) -> list:
    """Load MCP tools using AF's native MCP support.

    Returns:
        List of MCP tool objects, or empty list if not configured.
    """
    if not config.mcp_config or not config.mcp_config.url:
        return []

    try:
        mcp_tools = await client.get_mcp_tool(
            name="Research MCP",
            url=config.mcp_config.url,
            approval_mode="never_require",
        )
        return [mcp_tools] if mcp_tools else []
    except Exception as e:
        logger.warning("Failed to load MCP tools: %s", e)
        return []


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_today_str() -> str:
    """Get current date formatted for prompts."""
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


# Token limits map (AF uses model names without provider prefix)
MODEL_TOKEN_LIMITS = {
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "gpt-4.1": 1_047_576,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "o4-mini": 200_000,
    "o3-mini": 200_000,
    "o3": 200_000,
    "o1": 200_000,
}


def get_model_token_limit(model_name: str) -> int | None:
    """Look up the token limit for a model."""
    for key, limit in MODEL_TOKEN_LIMITS.items():
        if key in model_name:
            return limit
    return None


def is_token_limit_exceeded(exception: Exception, model_name: str = "") -> bool:
    """Detect if an exception indicates a token/context limit was exceeded."""
    error_str = str(exception).lower()
    token_keywords = ["token", "context", "length", "maximum context", "reduce", "too long"]
    return any(kw in error_str for kw in token_keywords)


def get_search_tools(config: Configuration) -> list:
    """Return search tools based on config.

    Uses AF's native web search (OpenAI Responses API web_search tool).
    No external API key required — search is handled by the model provider.
    """
    if config.search_api == SearchAPI.WEB_SEARCH:
        return [OpenAIChatClient.get_web_search_tool(search_context_size="medium")]
    return []
