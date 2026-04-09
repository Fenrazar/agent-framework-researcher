"""Agent factories and tool-bridge functions for supervisor and researcher agents.

Key design: AF's Agent IS the ReAct loop. The supervisor's `conduct_research` tool
spawns a researcher Agent internally — no dynamic fan-out at the workflow level.
"""

import logging

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from agent_framework_researcher.client_factory import create_client
from agent_framework_researcher.configuration import Configuration
from agent_framework_researcher.prompts import (
    compress_research_simple_human_message,
    compress_research_system_prompt,
    lead_researcher_prompt,
    research_system_prompt,
)
from agent_framework_researcher.tools import get_search_tools, get_today_str, load_mcp_tools, think

logger = logging.getLogger(__name__)


def _reasoning_options(config: Configuration) -> dict | None:
    """Build default_options with reasoning effort if configured."""
    if not config.reasoning_effort:
        return None
    return {"reasoning": {"effort": config.reasoning_effort}}


def research_complete() -> str:
    """Signal that all research is complete.

    Call this when you are satisfied with all research findings
    and ready to move on to report generation.
    """
    return "Research phase complete. Findings will now be compiled into a report."


def _build_conduct_research(client: OpenAIChatClient, config: Configuration):
    """Build a conduct_research tool function closed over client/config."""

    async def conduct_research(research_topic: str) -> str:
        """Conduct focused research on a specific topic using a dedicated researcher agent.

        This spawns a researcher agent that searches the web for information,
        then compresses the findings into a concise summary.

        Args:
            research_topic: The topic to research, described in detail.

        Returns:
            Compressed research findings with source citations.
        """
        # Build researcher tools
        tools: list = [think]
        tools.extend(get_search_tools(config))

        mcp_tools = await load_mcp_tools(client, config)
        tools.extend(mcp_tools)

        if not tools:
            return "Error: No search tools configured. Please set search_api to 'web_search' or configure MCP."

        researcher_prompt = research_system_prompt.format(
            mcp_prompt=config.mcp_prompt or "",
            date=get_today_str(),
        )

        research_client = create_client(config, model=config.research_model)
        researcher = Agent(
            client=research_client,
            name="Researcher",
            instructions=researcher_prompt,
            tools=tools,
            default_options=_reasoning_options(config),
        )

        try:
            result = await researcher.run(research_topic)
        except Exception as e:
            logger.error("Researcher failed: %s", e)
            return f"Research failed: {e}"

        # Compress the research output
        compress_prompt = compress_research_system_prompt.format(date=get_today_str())
        compress_client = create_client(config, model=config.compression_model)
        compressor = Agent(
            client=compress_client,
            name="Compressor",
            instructions=compress_prompt,
        )

        try:
            compressed = await compressor.run(f"{result.text}\n\n{compress_research_simple_human_message}")
            return compressed.text
        except Exception as e:
            logger.warning("Compression failed, returning raw research: %s", e)
            return result.text

    return conduct_research


def create_supervisor_agent(client: OpenAIChatClient, config: Configuration) -> Agent:
    """Create the lead research supervisor agent.

    The supervisor delegates research by calling `conduct_research` (which
    internally spawns researcher agents) and signals completion with
    `research_complete`.
    """
    supervisor_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=config.max_concurrent_research_units,
        max_researcher_iterations=config.max_researcher_iterations,
    )

    conduct_research = _build_conduct_research(client, config)

    return Agent(
        client=client,
        name="Supervisor",
        instructions=supervisor_prompt,
        tools=[conduct_research, research_complete, think],
        default_options=_reasoning_options(config),
    )
