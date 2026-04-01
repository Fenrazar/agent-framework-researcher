"""Research tools: Tavily search, think tool, MCP loader, and helpers.

Ported from open_deep_research/utils.py — all LangChain dependencies removed.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Annotated

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from pydantic import Field
from tavily import AsyncTavilyClient

from agent_framework_researcher.configuration import Configuration, SearchAPI
from agent_framework_researcher.models import Summary
from agent_framework_researcher.prompts import summarize_webpage_prompt

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
# Tavily search
# ---------------------------------------------------------------------------

async def tavily_search(
    queries: Annotated[list[str], Field(description="List of search queries to execute")],
) -> str:
    """Search the web using Tavily and summarize results.

    A search engine optimized for comprehensive, accurate, and trusted results.
    Useful for when you need to answer questions about current events.

    Args:
        queries: List of search queries to execute.

    Returns:
        Formatted string containing summarized search results.
    """
    config = Configuration.from_env()
    max_results = 5
    topic = "general"

    # Execute search queries
    tavily_client = AsyncTavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=True,
            topic=topic,
        )
        for query in queries
    ]
    search_results = await asyncio.gather(*search_tasks)

    # Deduplicate by URL
    unique_results: dict = {}
    for response in search_results:
        for result in response["results"]:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = {**result, "query": response["query"]}

    # Summarize each result in parallel
    max_chars = config.max_content_length

    async def _summarize_or_skip(result: dict) -> str | None:
        raw = result.get("raw_content")
        if not raw:
            return None
        return await _summarize_webpage(raw[:max_chars], config)

    summaries = await asyncio.gather(*[_summarize_or_skip(r) for r in unique_results.values()])

    # Format output
    if not unique_results:
        return "No valid search results found. Please try different search queries."

    output = "Search results:\n\n"
    for i, (url, result) in enumerate(unique_results.items()):
        summary = summaries[i]
        content = summary if summary else result.get("content", "")
        output += f"\n--- SOURCE {i + 1}: {result['title']} ---\n"
        output += f"URL: {url}\n\n"
        output += f"SUMMARY:\n{content}\n\n"
        output += "-" * 80 + "\n"

    return output


async def _summarize_webpage(webpage_content: str, config: Configuration) -> str:
    """Summarize webpage content using an AF Agent with JSON-mode parsing."""
    try:
        client = OpenAIChatClient(model=config.summarization_model)
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content,
            date=get_today_str(),
        )

        agent = Agent(
            client=client,
            instructions="You are a summarization assistant. Always respond with valid JSON matching the schema: {\"summary\": \"...\", \"key_excerpts\": \"...\"}",
        )

        result = await asyncio.wait_for(agent.run(prompt_content), timeout=60.0)
        parsed = json.loads(result.text)
        summary = Summary.model_validate(parsed)

        return (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
    except TimeoutError:
        logger.warning("Summarization timed out after 60s, returning raw content")
        return webpage_content
    except Exception as e:
        logger.warning("Summarization failed: %s, returning raw content", e)
        return webpage_content


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
    """Return the list of search tool functions based on config."""
    if config.search_api == SearchAPI.TAVILY:
        return [tavily_search]
    return []
